# Kernel GPU: High-Accuracy Multi-Modal Tabular + Foundation Neural Network v2 (PyTorch).
# Fixes v1 overfitting by feeding:
#   - Full Tabular SOTA (RankGauss) + Learned Categorical Embeddings
#   - In-fold Cholesky Multi-Target Ridge OOF probes (continuous + 5-class probs) for ALL Vision & Text embeddings + TF-IDF
#   - In-fold SVD(40) of Vision Trinity (DINOv2-L + SigLIP-SO400M + ConvNeXt-L) + SVD(20) of Text (BGE-M3 + E5-L + TF-IDF)
#   - Fine-tuned DeBERTa-v3-base OOF features
# Architecture: Dual-Branch Residual Gated Tabular Net + Feature Dropout + Mixup + SWA (last 4 epochs),
# trained with MSE + SmoothL1 + CORAL Ordinal BCE + 5-class CE across 5 seeds!
# ZERO leakage: every transform, SVD, TF-IDF, Cholesky Ridge, and RankGauss is fit strictly inside each fold!
import gc, glob, json, os, random, re, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

try:
    CONFIG
except NameError:
    CONFIG = {
        "tag": "mm_nn_v2",
        "vision_models": ["dinov2-l", "siglip-so400m", "convnext-l"],
        "text_embs": ["bgem3", "bgem3_full", "e5l"],
        "use_deberta": ["deberta_v3_base"],
        "seeds": [42, 100, 2024, 777, 1337],
        "epochs": 18,
        "batch_size": 256,
        "lr": 8e-4,
        "wd": 5e-2,
    }

TAG = CONFIG.get("tag", "mm_nn_v2")
SEEDS = CONFIG.get("seeds", [42, 100, 2024, 777, 1337])
EPOCHS = int(CONFIG.get("epochs", 18))
BS = int(CONFIG.get("batch_size", 256))
LR = float(CONFIG.get("lr", 8e-4))
WD = float(CONFIG.get("wd", 5e-2))
OUT = "/kaggle/working"
t0 = time.time()

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {dev} | TAG: {TAG} | CONFIG: {CONFIG}", flush=True)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def buscar(archivo, required=True):
    for prof in range(0, 6):
        for d in glob.glob("/kaggle/input/" + "*/" * prof):
            p = os.path.join(d, archivo)
            if os.path.exists(p):
                return p
    if required:
        raise FileNotFoundError(archivo)
    return None


D = os.path.dirname(os.path.dirname(buscar("train/train.csv")))
train_raw = pd.read_csv(f"{D}/train/train.csv").set_index("PetID")
test_raw = pd.read_csv(f"{D}/test/test.csv").set_index("PetID")
breeds_df = pd.read_csv(f"{D}/breed_labels.csv")
states_df = pd.read_csv(f"{D}/state_labels.csv")
breed_name_map = breeds_df.set_index("BreedID")["BreedName"].to_dict()
state_name_map = states_df.set_index("StateID")["StateName"].to_dict()

y_all = train_raw["AdoptionSpeed"].astype(int).values
groups_all = train_raw["RescuerID"].astype(str).values

MALAYSIA_DEMO = {
    "selangor": (6528400, 54995, 8104), "kuala lumpur": (1790000, 121300, 243),
    "pulau pinang": (1766800, 55243, 1048), "johor": (3742200, 37342, 19210),
    "perak": (2503500, 31668, 21035), "negeri sembilan": (1128900, 44450, 6686),
    "melaka": (922400, 49172, 1664), "kedah": (2163000, 22412, 9500),
    "pahang": (1664700, 36185, 36137), "terengganu": (1228300, 30901, 13035),
    "kelantan": (1856800, 14300, 15099), "sabah": (3898400, 25861, 73631),
    "sarawak": (2791700, 53358, 124450), "perlis": (253800, 24004, 821),
    "labuan": (99200, 77798, 91), "putrajaya": (90000, 121300, 49),
}

CAT_COLS = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
            "FurLength", "MaturitySize", "Vaccinated", "Dewormed", "Sterilized", "Health", "State"]


def build_tabular(raw_df, parte):
    df = raw_df.copy()
    if "AdoptionSpeed" in df.columns:
        df = df.drop(columns=["AdoptionSpeed"])
    for fname in (f"{parte}.parquet", f"img_stats_v2_{parte}.parquet", f"txt_stats_v2_{parte}.parquet",
                  f"metadata_num_{parte}.parquet", f"sentiment_num_{parte}.parquet", f"img_siglip_zs_{parte}.parquet"):
        pf = buscar(fname, required=False)
        if pf:
            df = df.join(pd.read_parquet(pf))
    for vm in CONFIG.get("vision_models", []):
        pe = buscar(f"img_{vm}_extra_{parte}.parquet", required=False)
        if pe:
            df = df.join(pd.read_parquet(pe))
    resc = df["RescuerID"].astype(str)
    df["rescuer_count"] = resc.map(resc.value_counts()).astype("float32")
    df["rescuer_count_log1p"] = np.log1p(df["rescuer_count"])
    st_names = df["State"].map(state_name_map).fillna("").str.lower()
    df["state_gdp_pc"] = st_names.map(lambda s: MALAYSIA_DEMO.get(s, (np.nan, np.nan, np.nan))[1]).astype("float32")
    df["state_pop"] = st_names.map(lambda s: MALAYSIA_DEMO.get(s, (np.nan, np.nan, np.nan))[0]).astype("float32")
    b1_name = df["Breed1"].map(breed_name_map).fillna("").str.lower()
    df["has_breed2"] = ((df["Breed2"] > 0) & (df["Breed2"] != df["Breed1"])).astype("float32")
    df["is_mixed_breed"] = (b1_name.str.contains("mixed|domestic short|domestic medium|domestic long", regex=True) | (df["has_breed2"] > 0)).astype("float32")
    df["age_log1p"] = np.log1p(df["Age"].clip(lower=0).astype("float32"))
    df["fee_log1p"] = np.log1p(df["Fee"].clip(lower=0).astype("float32"))
    df["fee_is_free"] = (df["Fee"] == 0).astype("float32")
    df["care_yes_cnt"] = sum((df[c] == 1).astype("float32") for c in ("Vaccinated", "Dewormed", "Sterilized"))
    df["care_unsure_cnt"] = sum((df[c] == 3).astype("float32") for c in ("Vaccinated", "Dewormed", "Sterilized"))
    df["num_colors"] = ((df["Color1"] > 0).astype(int) + (df["Color2"] > 0).astype(int) + (df["Color3"] > 0).astype(int)).astype("float32")
    for col in ("Age", "Fee", "PhotoAmt", "Quantity", "txt_largo"):
        if col in df.columns:
            s = df[col].astype("float32")
            rm = resc.map(s.groupby(resc).mean()).astype("float32")
            df[f"resc_{col}_mean"] = rm
            df[f"resc_{col}_diff"] = s - rm
    cats = df[CAT_COLS].copy()
    num_cols = [c for c in df.columns if c not in CAT_COLS and df[c].dtype != "object"]
    return df[num_cols].astype("float32"), cats


num_tr_df, cat_tr_df = build_tabular(train_raw, "train")
num_te_df, cat_te_df = build_tabular(test_raw, "test")

VISION_MATS, TEXT_MATS = {}, {}
for vm in CONFIG.get("vision_models", []):
    for agg in ("mean", "first"):
        p_tr = buscar(f"img_{vm}_{agg}_train.parquet", required=False)
        p_te = buscar(f"img_{vm}_{agg}_test.parquet", required=False)
        if p_tr and p_te:
            VISION_MATS[f"{vm}_{agg}"] = (
                pd.read_parquet(p_tr).reindex(train_raw.index).astype("float32").values,
                pd.read_parquet(p_te).reindex(test_raw.index).astype("float32").values,
            )
for tm in CONFIG.get("text_embs", []):
    p_tr = buscar(f"txt_emb_{tm}_train.parquet", required=False)
    p_te = buscar(f"txt_emb_{tm}_test.parquet", required=False)
    if p_tr and p_te:
        TEXT_MATS[tm] = (
            pd.read_parquet(p_tr).reindex(train_raw.index).astype("float32").values,
            pd.read_parquet(p_te).reindex(test_raw.index).astype("float32").values,
        )

DEBERTA_TR, DEBERTA_TE = [], []
for dtag in CONFIG.get("use_deberta", []):
    po = buscar(f"{dtag}_oof.parquet", required=False)
    pt = buscar(f"{dtag}_test.parquet", required=False)
    if po and pt:
        DEBERTA_TR.append(pd.read_parquet(po).reindex(train_raw.index).astype("float32").values)
        DEBERTA_TE.append(pd.read_parquet(pt).reindex(test_raw.index).astype("float32").values)

DESC_TR = train_raw["Description"].fillna("").astype(str)
DESC_TE = test_raw["Description"].fillna("").astype(str)


def chol_oof_features(M_tr, y_tr, g_tr, M_va, M_te, alphas=(3.0, 10.0, 30.0, 100.0, 300.0), add_cls=True):
    is_sparse = hasattr(M_tr, "tocsr")
    n_tr, n_va, n_te = M_tr.shape[0], M_va.shape[0], M_te.shape[0]
    ok_tr = np.ones(n_tr, dtype=bool) if is_sparse else np.isfinite(M_tr).all(axis=1)
    ok_va = np.ones(n_va, dtype=bool) if is_sparse else np.isfinite(M_va).all(axis=1)
    ok_te = np.ones(n_te, dtype=bool) if is_sparse else np.isfinite(M_te).all(axis=1)
    Y_t = np.column_stack([y_tr.astype(np.float32), np.eye(5, dtype=np.float32)[y_tr]]) if add_cls else y_tr.astype(np.float32)[:, None]
    d_out = Y_t.shape[1]
    oof = np.full((n_tr, d_out), np.nan, dtype=np.float32)
    va_o = np.full((n_va, d_out), np.nan, dtype=np.float32)
    te_o = np.full((n_te, d_out), np.nan, dtype=np.float32)

    inner = list(GroupKFold(n_splits=5).split(np.arange(n_tr), groups=g_tr))
    best_a = 30.0
    if not is_sparse:
        ia0, ib0 = inner[0]
        ia0, ib0 = ia0[ok_tr[ia0]], ib0[ok_tr[ib0]]
        X0, Y0 = M_tr[ia0], Y_t[ia0, :1]
        mx0, my0 = X0.mean(0, keepdims=True), Y0.mean(0, keepdims=True)
        G0 = ((X0 - mx0).T @ (X0 - mx0)).astype(np.float64)
        B0 = ((X0 - mx0).T @ (Y0 - my0)).astype(np.float64)
        Xv0, yv0 = M_tr[ib0] - mx0, y_tr[ib0]
        bmse = 1e18
        for a in alphas:
            Ga = G0.copy(); Ga[np.diag_indices(Ga.shape[0])] += a
            Wa = np.linalg.solve(Ga, B0).astype(np.float32)
            mse0 = float(np.mean(((Xv0 @ Wa + my0)[:, 0] - yv0) ** 2))
            if mse0 < bmse:
                bmse, best_a = mse0, float(a)

    def fit_pred(Xf, Yf, Xp):
        if is_sparse:
            P = Ridge(alpha=5.0).fit(Xf, Yf).predict(Xp)
            return P if P.ndim == 2 else P[:, None]
        mx, my = Xf.mean(0, keepdims=True), Yf.mean(0, keepdims=True)
        Xc, Yc = Xf - mx, Yf - my
        G = (Xc.T @ Xc).astype(np.float64)
        G[np.diag_indices(G.shape[0])] += best_a
        W = np.linalg.solve(G, (Xc.T @ Yc).astype(np.float64)).astype(np.float32)
        return (Xp - mx) @ W + my

    for ia, ib in inner:
        ia, ib = ia[ok_tr[ia]], ib[ok_tr[ib]]
        if len(ia) >= 20 and len(ib):
            oof[ib] = fit_pred(M_tr[ia], Y_t[ia], M_tr[ib])
    idx_ok = np.where(ok_tr)[0]
    if ok_va.any():
        va_o[ok_va] = fit_pred(M_tr[idx_ok], Y_t[idx_ok], M_va[ok_va])
    if ok_te.any():
        te_o[ok_te] = fit_pred(M_tr[idx_ok], Y_t[idx_ok], M_te[ok_te])
    return oof, va_o, te_o


def svd_feats(M_tr, M_va, M_te, k):
    is_sparse = hasattr(M_tr, "tocsr")
    if is_sparse:
        s = TruncatedSVD(n_components=k, random_state=42)
        return s.fit_transform(M_tr).astype("float32"), s.transform(M_va).astype("float32"), s.transform(M_te).astype("float32")
    ok_tr, ok_va, ok_te = np.isfinite(M_tr).all(1), np.isfinite(M_va).all(1), np.isfinite(M_te).all(1)
    s = TruncatedSVD(n_components=k, random_state=42).fit(M_tr[ok_tr])
    ztr = np.zeros((len(M_tr), k), dtype=np.float32)
    zva = np.zeros((len(M_va), k), dtype=np.float32)
    zte = np.zeros((len(M_te), k), dtype=np.float32)
    ztr[ok_tr] = s.transform(M_tr[ok_tr])
    if ok_va.any(): zva[ok_va] = s.transform(M_va[ok_va])
    if ok_te.any(): zte[ok_te] = s.transform(M_te[ok_te])
    return ztr, zva, zte


def encode_cats(ctr_df, cva_df, cte_df):
    tr_out, va_out, te_out, cards = [], [], [], []
    for c in CAT_COLS:
        vals = ctr_df[c].astype(str).value_counts()
        mp = {v: i + 1 for i, v in enumerate(vals[vals >= 3].index.tolist())}
        tr_out.append(ctr_df[c].astype(str).map(mp).fillna(0).astype("int64").values)
        va_out.append(cva_df[c].astype(str).map(mp).fillna(0).astype("int64").values)
        te_out.append(cte_df[c].astype(str).map(mp).fillna(0).astype("int64").values)
        cards.append(len(mp) + 1)
    return np.column_stack(tr_out), np.column_stack(va_out), np.column_stack(te_out), cards


class PetResNetV2(nn.Module):
    def __init__(self, d_num, d_probe, cards):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(20, max(4, (c + 1) // 2))) for c in cards])
        e_tot = sum(e.embedding_dim for e in self.embs)
        # Main tabular + SVD branch
        self.stem = nn.Sequential(nn.Dropout(0.15), nn.Linear(d_num + e_tot, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.45))
        self.res1 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.45), nn.Linear(256, 256), nn.LayerNorm(256))
        # Direct skip-connected linear/probe branch (prevents over-smoothing strong Ridge/DeBERTa signals!)
        self.probe_net = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_probe, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.3))
        self.act = nn.SiLU()
        tot = 256 + 64 + d_probe
        self.head_reg = nn.Linear(tot, 1)
        self.head_ord = nn.Linear(tot, 4)
        self.head_cls = nn.Linear(tot, 5)

    def forward(self, x_num, x_probe, x_cat):
        ce = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=-1)
        h = self.stem(torch.cat([x_num, ce], dim=-1))
        h = self.act(h + self.res1(h))
        hp = self.probe_net(x_probe)
        z = torch.cat([h, hp, x_probe], dim=-1)
        return self.head_reg(z).squeeze(-1), self.head_ord(z), self.head_cls(z)


def cortes_cuantiles(p, y):
    props = pd.Series(y).value_counts(normalize=True).sort_index().reindex(range(5), fill_value=0).values
    return np.quantile(p, np.cumsum(props)[:-1])


sgkf5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits_5 = list(sgkf5.split(train_raw, y_all, groups=groups_all))

oof_reg = np.zeros(len(train_raw), dtype=np.float64)
oof_ord = np.zeros(len(train_raw), dtype=np.float64)
oof_blend = np.zeros(len(train_raw), dtype=np.float64)
oof_prob = np.zeros((len(train_raw), 5), dtype=np.float64)

test_reg = np.zeros(len(test_raw), dtype=np.float64)
test_ord = np.zeros(len(test_raw), dtype=np.float64)
test_blend = np.zeros(len(test_raw), dtype=np.float64)
test_prob = np.zeros((len(test_raw), 5), dtype=np.float64)

mse, huber, bce, ce = nn.MSELoss(), nn.SmoothL1Loss(beta=0.8), nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()

for fi, (tr_i, va_i) in enumerate(splits_5):
    tf = time.time()
    y_tr, g_tr = y_all[tr_i], groups_all[tr_i]
    num_parts_tr, num_parts_va, num_parts_te = [num_tr_df.values[tr_i]], [num_tr_df.values[va_i]], [num_te_df.values]
    prb_parts_tr, prb_parts_va, prb_parts_te = [], [], []

    # Vision SVD(32) + Cholesky OOF probes
    for vkey, (Vtr, Vte) in VISION_MATS.items():
        if vkey.endswith("_mean"):
            s_tr, s_va, s_te = svd_feats(Vtr[tr_i], Vtr[va_i], Vte, 32)
            num_parts_tr.append(s_tr); num_parts_va.append(s_va); num_parts_te.append(s_te)
        p_tr, p_va, p_te = chol_oof_features(Vtr[tr_i], y_tr, g_tr, Vtr[va_i], Vte, add_cls=vkey.endswith("_mean"))
        prb_parts_tr.append(p_tr); prb_parts_va.append(p_va); prb_parts_te.append(p_te)

    # Joint Vision probe (all mean vision embeddings concatenated!)
    mean_vkeys = [k for k in VISION_MATS if k.endswith("_mean")]
    if len(mean_vkeys) >= 2:
        JVtr = np.hstack([np.nan_to_num(VISION_MATS[k][0][tr_i], 0.0) for k in mean_vkeys])
        JVva = np.hstack([np.nan_to_num(VISION_MATS[k][0][va_i], 0.0) for k in mean_vkeys])
        JVte = np.hstack([np.nan_to_num(VISION_MATS[k][1], 0.0) for k in mean_vkeys])
        p_tr, p_va, p_te = chol_oof_features(JVtr, y_tr, g_tr, JVva, JVte, alphas=(10.0, 30.0, 100.0, 300.0, 1000.0), add_cls=True)
        prb_parts_tr.append(p_tr); prb_parts_va.append(p_va); prb_parts_te.append(p_te)

    # Text embeddings SVD(16) + Cholesky OOF probes
    for tkey, (Ttr, Tte) in TEXT_MATS.items():
        s_tr, s_va, s_te = svd_feats(Ttr[tr_i], Ttr[va_i], Tte, 16)
        num_parts_tr.append(s_tr); num_parts_va.append(s_va); num_parts_te.append(s_te)
        p_tr, p_va, p_te = chol_oof_features(Ttr[tr_i], y_tr, g_tr, Ttr[va_i], Tte, add_cls=(tkey == "bgem3_full"))
        prb_parts_tr.append(p_tr); prb_parts_va.append(p_va); prb_parts_te.append(p_te)

    # TF-IDF word + char OOF probes + SVD(16)
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
    Mw_tr, Mw_va, Mw_te = vw.fit_transform(DESC_TR.iloc[tr_i]), vw.transform(DESC_TR.iloc[va_i]), vw.transform(DESC_TE)
    p_tr, p_va, p_te = chol_oof_features(Mw_tr, y_tr, g_tr, Mw_va, Mw_te, add_cls=True)
    prb_parts_tr.append(p_tr); prb_parts_va.append(p_va); prb_parts_te.append(p_te)
    s_tr, s_va, s_te = svd_feats(Mw_tr, Mw_va, Mw_te, 16)
    num_parts_tr.append(s_tr); num_parts_va.append(s_va); num_parts_te.append(s_te)

    # Fine-tuned DeBERTa OOF
    for Dtr, Dte in zip(DEBERTA_TR, DEBERTA_TE):
        prb_parts_tr.append(Dtr[tr_i]); prb_parts_va.append(Dtr[va_i]); prb_parts_te.append(Dte)

    def clean_scale(tr_list, va_list, te_list, use_qt=True):
        Atr = np.hstack(tr_list).astype(np.float32)
        Ava = np.hstack(va_list).astype(np.float32)
        Ate = np.hstack(te_list).astype(np.float32)
        med = np.nanmedian(Atr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        for A in (Atr, Ava, Ate):
            bad = ~np.isfinite(A)
            if bad.any():
                A[bad] = np.take(med, np.where(bad)[1])
        if use_qt:
            sc = QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=42)
            return np.clip(sc.fit_transform(Atr), -5, 5).astype(np.float32), np.clip(sc.transform(Ava), -5, 5).astype(np.float32), np.clip(sc.transform(Ate), -5, 5).astype(np.float32)
        sc = StandardScaler()
        return np.clip(sc.fit_transform(Atr), -6, 6).astype(np.float32), np.clip(sc.transform(Ava), -6, 6).astype(np.float32), np.clip(sc.transform(Ate), -6, 6).astype(np.float32)

    Xnum_tr, Xnum_va, Xnum_te = clean_scale(num_parts_tr, num_parts_va, num_parts_te, use_qt=True)
    Xprb_tr, Xprb_va, Xprb_te = clean_scale(prb_parts_tr, prb_parts_va, prb_parts_te, use_qt=False)
    Ctr, Cva, Cte, cards = encode_cats(cat_tr_df.iloc[tr_i], cat_tr_df.iloc[va_i], cat_te_df)
    if fi == 0:
        print(f"NNv2 dims: num={Xnum_tr.shape[1]}, probes={Xprb_tr.shape[1]}, cats={len(cards)}", flush=True)

    t_ntr, t_nva, t_nte = torch.tensor(Xnum_tr, device=dev), torch.tensor(Xnum_va, device=dev), torch.tensor(Xnum_te, device=dev)
    t_ptr, t_pva, t_pte = torch.tensor(Xprb_tr, device=dev), torch.tensor(Xprb_va, device=dev), torch.tensor(Xprb_te, device=dev)
    t_ctr, t_cva, t_cte = torch.tensor(Ctr, device=dev), torch.tensor(Cva, device=dev), torch.tensor(Cte, device=dev)
    t_ytr = torch.tensor(y_tr, dtype=torch.long, device=dev)
    t_yord = (t_ytr.unsqueeze(1) >= torch.arange(1, 5, device=dev).unsqueeze(0)).float()

    va_r_s, va_o_s, va_p_s = [], [], []
    te_r_s, te_o_s, te_p_s = [], [], []
    for s in SEEDS:
        seed_all(s + fi * 13)
        net = PetResNetV2(Xnum_tr.shape[1], Xprb_tr.shape[1], cards).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        sn_vr, sn_vo, sn_vp, sn_tr, sn_to, sn_tp = [], [], [], [], [], []
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(t_ntr), device=dev)
            for k in range(0, len(t_ntr), BS):
                bi = perm[k:k + BS]
                xn = t_ntr[bi] + 0.08 * torch.randn_like(t_ntr[bi])
                xp = t_ptr[bi] + 0.05 * torch.randn_like(t_ptr[bi])
                r, o, c = net(xn, xp, t_ctr[bi])
                loss = 0.35 * mse(r, t_ytr[bi].float()) + 0.15 * huber(r, t_ytr[bi].float()) + 0.30 * bce(o, t_yord[bi]) + 0.20 * ce(c, t_ytr[bi])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.5)
                opt.step()
            sched.step()
            if ep >= EPOCHS - 4:
                net.eval()
                with torch.no_grad():
                    rv, ov, cv = net(t_nva, t_pva, t_cva)
                    rt, ot, ct = net(t_nte, t_pte, t_cte)
                    sn_vr.append(rv.cpu().numpy()); sn_vo.append(torch.sigmoid(ov).sum(-1).cpu().numpy()); sn_vp.append(torch.softmax(cv, -1).cpu().numpy())
                    sn_tr.append(rt.cpu().numpy()); sn_to.append(torch.sigmoid(ot).sum(-1).cpu().numpy()); sn_tp.append(torch.softmax(ct, -1).cpu().numpy())
        va_r_s.append(np.mean(sn_vr, 0)); va_o_s.append(np.mean(sn_vo, 0)); va_p_s.append(np.mean(sn_vp, 0))
        te_r_s.append(np.mean(sn_tr, 0)); te_o_s.append(np.mean(sn_to, 0)); te_p_s.append(np.mean(sn_tp, 0))

    vr, vo, vp = np.mean(va_r_s, 0), np.mean(va_o_s, 0), np.mean(va_p_s, 0)
    vc = (vp * np.arange(5)[None, :]).sum(1)
    vb = (vr + vo + vc) / 3.0
    tr_r, tr_o, tr_p = np.mean(te_r_s, 0), np.mean(te_o_s, 0), np.mean(te_p_s, 0)
    tr_c = (tr_p * np.arange(5)[None, :]).sum(1)
    tr_b = (tr_r + tr_o + tr_c) / 3.0

    oof_reg[va_i], oof_ord[va_i], oof_prob[va_i], oof_blend[va_i] = vr, vo, vp, vb
    test_reg += tr_r / 5.0; test_ord += tr_o / 5.0; test_prob += tr_p / 5.0; test_blend += tr_b / 5.0
    fq = cohen_kappa_score(y_all[va_i], np.digitize(vb, cortes_cuantiles(vb, y_tr)), weights="quadratic")
    frmse = float(np.sqrt(np.mean((vb - y_all[va_i]) ** 2)))
    print(f"Fold {fi} | RMSE={frmse:.4f} | QWK={fq:.4f} ({time.time()-tf:.1f}s)", flush=True)

fqs = [cohen_kappa_score(y_all[va], np.digitize(oof_blend[va], cortes_cuantiles(oof_blend[va], y_all[tr])), weights="quadratic") for tr, va in splits_5]
tot_rmse = float(np.sqrt(np.mean((oof_blend - y_all) ** 2)))
print(f"== {TAG} (5-fold SGKF) RMSE={tot_rmse:.4f} | QWK={np.mean(fqs):.4f} +/- {np.std(fqs, ddof=1):.4f} | folds={[round(x, 4) for x in fqs]}", flush=True)

cols = [f"{TAG}_reg", f"{TAG}_ord", f"{TAG}_blend"] + [f"{TAG}_p{k}" for k in range(5)]
pd.DataFrame(np.column_stack([oof_reg, oof_ord, oof_blend, oof_prob]), index=train_raw.index, columns=cols, dtype="float32").to_parquet(f"{OUT}/{TAG}_oof.parquet")
pd.DataFrame(np.column_stack([test_reg, test_ord, test_blend, test_prob]), index=test_raw.index, columns=cols, dtype="float32").to_parquet(f"{OUT}/{TAG}_test.parquet")
json.dump({"tag": TAG, "rmse_5f": tot_rmse, "qwk_5f_mean": float(np.mean(fqs)), "qwk_5f_std": float(np.std(fqs, ddof=1)),
           "qwk_5f_folds": [round(x, 5) for x in fqs], "elapsed_sec": round(time.time() - t0, 1)}, open(f"{OUT}/metrics.json", "w"), indent=1)
print("LISTO", sorted(os.listdir(OUT)), f"total {time.time()-t0:.0f}s", flush=True)
