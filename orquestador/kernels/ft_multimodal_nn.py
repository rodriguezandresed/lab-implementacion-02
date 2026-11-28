# Kernel GPU: Multi-Modal Gated Residual Neural Network (PyTorch) directamente sobre
# embeddings completos (DINOv2-L 1024d + SigLIP-SO400M 1152d + BGE-M3 1024d + BGE-M3-Full 1024d + E5-L 1024d)
# + Embeddings categoricos aprendidos + Tabular SOTA (RankGauss).
# Inspirado en 1st/5th place solutions con Modality Dropout + Feature Noise + CosineAnnealing + Multi-Task (MSE + CORAL Ordinal + 5-class CE).
# ZERO leakage: scalers/QuantileTransformers/group stats ajustados estrictamente dentro de cada fold!
import gc, glob, json, os, random, re, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer

try:
    CONFIG
except NameError:
    CONFIG = {
        "tag": "mm_nn_v1",
        "vision_models": ["dinov2-l", "siglip-so400m"],
        "text_embs": ["bgem3", "bgem3_full", "e5l"],
        "seeds": [42, 100, 2024, 777, 1337],
        "epochs": 28,
        "batch_size": 256,
        "lr": 1.2e-3,
        "wd": 3e-2,
        "n_folds_extra": 10,
    }

TAG = CONFIG.get("tag", "mm_nn_v1")
SEEDS = CONFIG.get("seeds", [42, 100, 2024, 777, 1337])
EPOCHS = int(CONFIG.get("epochs", 28))
BS = int(CONFIG.get("batch_size", 256))
LR = float(CONFIG.get("lr", 1.2e-3))
WD = float(CONFIG.get("wd", 3e-2))
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
    nums = df[num_cols].astype("float32")
    return nums, cats


num_tr_df, cat_tr_df = build_tabular(train_raw, "train")
num_te_df, cat_te_df = build_tabular(test_raw, "test")
print(f"Tabular nums: {num_tr_df.shape}, cats: {cat_tr_df.shape}", flush=True)

# Load dense embedding modalities
MODS_TR, MODS_TE, MOD_NAMES = [], [], []
for vm in CONFIG.get("vision_models", []):
    for agg in ("mean", "first"):
        p_tr = buscar(f"img_{vm}_{agg}_train.parquet", required=False)
        p_te = buscar(f"img_{vm}_{agg}_test.parquet", required=False)
        if p_tr and p_te:
            Mtr = pd.read_parquet(p_tr).reindex(train_raw.index).fillna(0.0).values.astype("float32")
            Mte = pd.read_parquet(p_te).reindex(test_raw.index).fillna(0.0).values.astype("float32")
            MODS_TR.append(Mtr); MODS_TE.append(Mte); MOD_NAMES.append(f"v_{vm}_{agg}")
            print(f"Loaded modality v_{vm}_{agg}: {Mtr.shape}", flush=True)

for tm in CONFIG.get("text_embs", []):
    p_tr = buscar(f"txt_emb_{tm}_train.parquet", required=False)
    p_te = buscar(f"txt_emb_{tm}_test.parquet", required=False)
    if p_tr and p_te:
        Mtr = pd.read_parquet(p_tr).reindex(train_raw.index).fillna(0.0).values.astype("float32")
        Mte = pd.read_parquet(p_te).reindex(test_raw.index).fillna(0.0).values.astype("float32")
        MODS_TR.append(Mtr); MODS_TE.append(Mte); MOD_NAMES.append(f"t_{tm}")
        print(f"Loaded modality t_{tm}: {Mtr.shape}", flush=True)

# Encode categorical indices fit ONLY per fold (actually vocab mapping from fold train, 0 = UNK)
def encode_cats(ctr_df, cva_df, cte_df):
    tr_out, va_out, te_out, cards = [], [], [], []
    for c in CAT_COLS:
        vals = ctr_df[c].astype(str).value_counts()
        valid = vals[vals >= 3].index.tolist()
        mp = {v: i + 1 for i, v in enumerate(valid)}
        tr_out.append(ctr_df[c].astype(str).map(mp).fillna(0).astype("int64").values)
        va_out.append(cva_df[c].astype(str).map(mp).fillna(0).astype("int64").values)
        te_out.append(cte_df[c].astype(str).map(mp).fillna(0).astype("int64").values)
        cards.append(len(mp) + 1)
    return np.column_stack(tr_out), np.column_stack(va_out), np.column_stack(te_out), cards


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalPetNet(nn.Module):
    def __init__(self, num_dim, cards, mod_dims):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(24, max(4, (c + 1) // 2))) for c in cards])
        emb_tot = sum(e.embedding_dim for e in self.embs)
        self.tab_enc = nn.Sequential(
            nn.Linear(num_dim + emb_tot, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.35),
        )
        self.mod_encs = nn.ModuleList([ModalityEncoder(d, 96, drop=0.55) for d in mod_dims])
        # Gated fusion over tabular + each embedding modality
        tot_dim = 192 + 96 * len(mod_dims)
        self.gate = nn.Sequential(nn.Linear(tot_dim, tot_dim), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Linear(tot_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.4),
        )
        self.res = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
        )
        self.act = nn.SiLU()
        self.head_reg = nn.Linear(256, 1)
        self.head_ord = nn.Linear(256, 4)
        self.head_cls = nn.Linear(256, 5)

    def forward(self, x_num, x_cat, mods):
        ce = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=-1)
        ht = self.tab_enc(torch.cat([x_num, ce], dim=-1))
        hm_list = []
        for enc, m in zip(self.mod_encs, mods):
            h = enc(m)
            # Modality dropout during training (15% chance to drop an entire modality per sample!)
            if self.training:
                keep = (torch.rand(h.size(0), 1, device=h.device) > 0.15).float()
                h = h * keep / 0.85
            hm_list.append(h)
        h_all = torch.cat([ht] + hm_list, dim=-1)
        h_all = h_all * self.gate(h_all)
        z = self.fuse(h_all)
        z = self.act(z + self.res(z))
        return self.head_reg(z).squeeze(-1), self.head_ord(z), self.head_cls(z)


def cortes_cuantiles(p, y):
    props = pd.Series(y).value_counts(normalize=True).sort_index().reindex(range(5), fill_value=0).values
    return np.quantile(p, np.cumsum(props)[:-1])


def run_cv(n_splits, seed_split):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_split)
    splits = list(sgkf.split(train_raw, y_all, groups=groups_all))
    oof_blend = np.zeros(len(train_raw), dtype=np.float64)
    oof_reg = np.zeros(len(train_raw), dtype=np.float64)
    oof_ord = np.zeros(len(train_raw), dtype=np.float64)
    oof_prob = np.zeros((len(train_raw), 5), dtype=np.float64)

    test_blend = np.zeros(len(test_raw), dtype=np.float64)
    test_reg = np.zeros(len(test_raw), dtype=np.float64)
    test_ord = np.zeros(len(test_raw), dtype=np.float64)
    test_prob = np.zeros((len(test_raw), 5), dtype=np.float64)

    mod_dims = [m.shape[1] for m in MODS_TR]
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss(beta=0.8)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    for fi, (tr_i, va_i) in enumerate(splits):
        tf = time.time()
        # Fit RankGauss & median imputation strictly on fold train
        Atr = num_tr_df.values[tr_i].copy()
        Ava = num_tr_df.values[va_i].copy()
        Ate = num_te_df.values.copy()
        med = np.nanmedian(Atr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        for A in (Atr, Ava, Ate):
            bad = ~np.isfinite(A)
            if bad.any():
                A[bad] = np.take(med, np.where(bad)[1])
        qt = QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=42)
        Atr = np.clip(qt.fit_transform(Atr), -5, 5).astype("float32")
        Ava = np.clip(qt.transform(Ava), -5, 5).astype("float32")
        Ate = np.clip(qt.transform(Ate), -5, 5).astype("float32")

        Ctr, Cva, Cte, cards = encode_cats(cat_tr_df.iloc[tr_i], cat_tr_df.iloc[va_i], cat_te_df)

        t_Atr = torch.tensor(Atr, device=dev)
        t_Ava = torch.tensor(Ava, device=dev)
        t_Ate = torch.tensor(Ate, device=dev)
        t_Ctr = torch.tensor(Ctr, device=dev)
        t_Cva = torch.tensor(Cva, device=dev)
        t_Cte = torch.tensor(Cte, device=dev)
        t_ytr = torch.tensor(y_all[tr_i], dtype=torch.long, device=dev)
        t_yord = (t_ytr.unsqueeze(1) >= torch.arange(1, 5, device=dev).unsqueeze(0)).float()
        t_Mtr = [torch.tensor(m[tr_i], device=dev) for m in MODS_TR]
        t_Mva = [torch.tensor(m[va_i], device=dev) for m in MODS_TR]
        t_Mte = [torch.tensor(m, device=dev) for m in MODS_TE]

        va_r_s, va_o_s, va_p_s = [], [], []
        te_r_s, te_o_s, te_p_s = [], [], []
        for s in SEEDS:
            seed_all(s + fi * 17)
            net = MultiModalPetNet(Atr.shape[1], cards, mod_dims).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

            # Evaluate last 4 epochs average (SWA-like smoothing without peeking at valid!)
            snap_va_r, snap_va_o, snap_va_p = [], [], []
            snap_te_r, snap_te_o, snap_te_p = [], [], []
            for ep in range(EPOCHS):
                net.train()
                perm = torch.randperm(len(t_Atr), device=dev)
                for k in range(0, len(t_Atr), BS):
                    bi = perm[k:k + BS]
                    xn = t_Atr[bi] + 0.05 * torch.randn_like(t_Atr[bi])
                    xc = t_Ctr[bi]
                    xm = [m[bi] for m in t_Mtr]
                    r, o, c = net(xn, xc, xm)
                    loss = 0.35 * mse(r, t_ytr[bi].float()) + 0.15 * huber(r, t_ytr[bi].float()) + 0.30 * bce(o, t_yord[bi]) + 0.20 * ce(c, t_ytr[bi])
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                    opt.step()
                sched.step()
                if ep >= EPOCHS - 5:
                    net.eval()
                    with torch.no_grad():
                        rv, ov, cv = net(t_Ava, t_Cva, t_Mva)
                        rt, ot, ct = net(t_Ate, t_Cte, t_Mte)
                        snap_va_r.append(rv.cpu().numpy())
                        snap_va_o.append(torch.sigmoid(ov).sum(-1).cpu().numpy())
                        snap_va_p.append(torch.softmax(cv, dim=-1).cpu().numpy())
                        snap_te_r.append(rt.cpu().numpy())
                        snap_te_o.append(torch.sigmoid(ot).sum(-1).cpu().numpy())
                        snap_te_p.append(torch.softmax(ct, dim=-1).cpu().numpy())
            va_r_s.append(np.mean(snap_va_r, axis=0))
            va_o_s.append(np.mean(snap_va_o, axis=0))
            va_p_s.append(np.mean(snap_va_p, axis=0))
            te_r_s.append(np.mean(snap_te_r, axis=0))
            te_o_s.append(np.mean(snap_te_o, axis=0))
            te_p_s.append(np.mean(snap_te_p, axis=0))

        vr = np.mean(va_r_s, axis=0)
        vo = np.mean(va_o_s, axis=0)
        vp = np.mean(va_p_s, axis=0)
        vc = (vp * np.arange(5)[None, :]).sum(axis=1)
        vb = (vr + vo + vc) / 3.0

        tr_r = np.mean(te_r_s, axis=0)
        tr_o = np.mean(te_o_s, axis=0)
        tr_p = np.mean(te_p_s, axis=0)
        tr_c = (tr_p * np.arange(5)[None, :]).sum(axis=1)
        tr_b = (tr_r + tr_o + tr_c) / 3.0

        oof_reg[va_i], oof_ord[va_i], oof_prob[va_i], oof_blend[va_i] = vr, vo, vp, vb
        test_reg += tr_r / n_splits
        test_ord += tr_o / n_splits
        test_prob += tr_p / n_splits
        test_blend += tr_b / n_splits

        fq = cohen_kappa_score(y_all[va_i], np.digitize(vb, cortes_cuantiles(vb, y_all[tr_i])), weights="quadratic")
        frmse = float(np.sqrt(np.mean((vb - y_all[va_i]) ** 2)))
        print(f"[{n_splits}-fold] Fold {fi} | RMSE={frmse:.4f} | QWK={fq:.4f} ({time.time()-tf:.1f}s)", flush=True)

    fqs = [cohen_kappa_score(y_all[va_i], np.digitize(oof_blend[va_i], cortes_cuantiles(oof_blend[va_i], y_all[tr_i])), weights="quadratic") for tr_i, va_i in splits]
    tot_rmse = float(np.sqrt(np.mean((oof_blend - y_all) ** 2)))
    print(f"== {TAG} ({n_splits}-fold) RMSE={tot_rmse:.4f} | QWK={np.mean(fqs):.4f} +/- {np.std(fqs, ddof=1):.4f} | folds={[round(x, 4) for x in fqs]}", flush=True)
    return oof_reg, oof_ord, oof_blend, oof_prob, test_reg, test_ord, test_blend, test_prob, fqs, tot_rmse


r5, o5, b5, p5, tr5, to5, tb5, tp5, fqs5, rmse5 = run_cv(5, 42)

cols = [f"{TAG}_reg", f"{TAG}_ord", f"{TAG}_blend"] + [f"{TAG}_p{k}" for k in range(5)]
pd.DataFrame(np.column_stack([r5, o5, b5, p5]), index=train_raw.index, columns=cols, dtype="float32").to_parquet(f"{OUT}/{TAG}_oof.parquet")
pd.DataFrame(np.column_stack([tr5, to5, tb5, tp5]), index=test_raw.index, columns=cols, dtype="float32").to_parquet(f"{OUT}/{TAG}_test.parquet")

# Also run 10-fold for extra-smooth test predictions!
n_ex = int(CONFIG.get("n_folds_extra", 10))
if n_ex > 5:
    r10, o10, b10, p10, tr10, to10, tb10, tp10, fqs10, rmse10 = run_cv(n_ex, 2026)
    pd.DataFrame({"blend_5f": tb5, "blend_10f": tb10, "blend_all": 0.4 * tb5 + 0.6 * tb10}, index=test_raw.index, dtype="float32").to_parquet(f"{OUT}/{TAG}_test_10f.parquet")
else:
    fqs10, rmse10 = None, None

json.dump({
    "tag": TAG, "rmse_5f": rmse5, "qwk_5f_mean": float(np.mean(fqs5)), "qwk_5f_std": float(np.std(fqs5, ddof=1)),
    "qwk_5f_folds": [round(x, 5) for x in fqs5], "rmse_10f": rmse10,
    "qwk_10f_mean": float(np.mean(fqs10)) if fqs10 else None, "elapsed_sec": round(time.time() - t0, 1)
}, open(f"{OUT}/metrics.json", "w"), indent=1)
print("LISTO", sorted(os.listdir(OUT)), f"total {time.time()-t0:.0f}s", flush=True)
