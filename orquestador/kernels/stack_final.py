# Level-2 Context-Aware Meta-Stacker & Post-Processing Engine (Zero-Leakage)
# Combines OOF & Test predictions from ALL diverse Level-1 families aligned on exact 5-fold SGKF(seed=42):
#   - GBDT families (LGBM RMSE/Deep/Huber/Cls/Ordinal, CatBoost RMSE/Cls, XGBoost RMSE/Cls)
#   - Deep Multi-Modal Gated Residual NN (over raw DINOv2-L + SigLIP-SO400M + ConvNeXt-L + BGE-M3 + E5-L)
#   - Fine-tuned End-to-End Transformers (DeBERTa-v3-base, mDeBERTa-v3-base)
# Evaluates: Linear Ridge Stack, Non-Negative Convex Blend, 1st-Place Rank Averaging,
# Context-Aware Gated Stacker (knows has_photo / txt_vacia / PhotoAmt / Type / rescuer_count!),
# and compares Quantile Matching vs OOF Nelder-Mead vs Shrinkage-Blended Thresholds!
import glob, json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMRegressor

try:
    CONFIG
except NameError:
    CONFIG = {"exp": "EXP-STACK", "oof_sources": [], "thresh_mode": "auto"}

OUT = "/kaggle/working"
t0 = time.time()


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
y_all = train_raw["AdoptionSpeed"].astype(int).values
groups_all = train_raw["RescuerID"].astype(str).values

sgkf5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits_5 = list(sgkf5.split(train_raw, y_all, groups=groups_all))


def cortes_cuantiles(p, y):
    props = pd.Series(y).value_counts(normalize=True).sort_index().reindex(range(5), fill_value=0).values
    return np.quantile(p, np.cumsum(props)[:-1])


def opt_rounder_fit(p_tr, y_tr):
    init = cortes_cuantiles(p_tr, y_tr)
    def loss(th):
        return -cohen_kappa_score(y_tr, np.digitize(p_tr, np.sort(th)), weights="quadratic")
    res = minimize(loss, init, method="Nelder-Mead", options={"maxiter": 500})
    return np.sort(res.x)


def score_oof(p, splits):
    rmse = float(np.sqrt(np.mean((p - y_all) ** 2)))
    fq_q = [cohen_kappa_score(y_all[va], np.digitize(p[va], cortes_cuantiles(p[va], y_all[tr])), weights="quadratic") for tr, va in splits]
    fq_nm = []
    fq_sh = []
    for tr, va in splits:
        th_nm = opt_rounder_fit(p[tr], y_all[tr])
        th_q = cortes_cuantiles(p[va], y_all[tr])
        fq_nm.append(cohen_kappa_score(y_all[va], np.digitize(p[va], th_nm), weights="quadratic"))
        fq_sh.append(cohen_kappa_score(y_all[va], np.digitize(p[va], 0.65 * th_q + 0.35 * th_nm), weights="quadratic"))
    return {
        "rmse": rmse,
        "qwk_q": float(np.mean(fq_q)), "std_q": float(np.std(fq_q, ddof=1)), "folds_q": [round(x, 5) for x in fq_q],
        "qwk_nm": float(np.mean(fq_nm)), "std_nm": float(np.std(fq_nm, ddof=1)),
        "qwk_sh": float(np.mean(fq_sh)), "std_sh": float(np.std(fq_sh, ddof=1)),
    }


# Discover ALL attached OOF + Test parquet pairs automatically across /kaggle/input/ (depths 1..4)
oof_cols, test_cols = {}, {}
prob_oof_cols, prob_test_cols = {}, {}

input_dirs = []
for prof in range(1, 5):
    for d in sorted(glob.glob("/kaggle/input/" + "*/" * prof)):
        d = d.rstrip("/")
        if os.path.isdir(d) and (os.path.exists(os.path.join(d, "oof_preds.parquet")) or glob.glob(os.path.join(d, "*_oof.parquet"))):
            input_dirs.append(d)
input_dirs = list(dict.fromkeys(input_dirs))
print("Found OOF input dirs:", input_dirs, flush=True)

for d in input_dirs:
    slug = os.path.basename(d)
    # 1) sota_engine outputs (oof_preds.parquet / test_preds.parquet)
    po = os.path.join(d, "oof_preds.parquet")
    pt = os.path.join(d, "test_preds.parquet")
    if os.path.exists(po) and os.path.exists(pt):
        do = pd.read_parquet(po).reindex(train_raw.index)
        dt = pd.read_parquet(pt).reindex(test_raw.index)
        for c in do.columns:
            if c == "best_ens":
                continue
            key = f"{slug}__{c}"
            oof_cols[key] = do[c].values.astype(np.float64)
            test_cols[key] = dt[c].values.astype(np.float64)
    # 2) ft_multimodal_nn or ft_deberta outputs (*_oof.parquet / *_test.parquet)
    for po in sorted(glob.glob(os.path.join(d, "*_oof.parquet"))):
        base = os.path.basename(po)[:-len("_oof.parquet")]
        pt = os.path.join(d, f"{base}_test.parquet")
        if not os.path.exists(pt):
            continue
        do = pd.read_parquet(po).reindex(train_raw.index)
        dt = pd.read_parquet(pt).reindex(test_raw.index)
        # Prefer 10-fold smoothed test blend if available!
        pt10 = os.path.join(d, f"{base}_test_10f.parquet")
        dt10 = pd.read_parquet(pt10).reindex(test_raw.index) if os.path.exists(pt10) else None
        for c in do.columns:
            key = f"{slug}__{c}"
            if c.endswith("_blend") or c.endswith("_reg") or c.endswith("_ord") or c.endswith("_cls"):
                oof_cols[key] = do[c].values.astype(np.float64)
                if dt10 is not None and c.endswith("_blend") and "blend_all" in dt10.columns:
                    test_cols[key] = dt10["blend_all"].values.astype(np.float64)
                else:
                    test_cols[key] = dt[c].values.astype(np.float64)
            else:
                prob_oof_cols[key] = do[c].values.astype(np.float64)
                prob_test_cols[key] = dt[c].values.astype(np.float64)

print(f"Discovered {len(oof_cols)} Level-1 continuous models and {len(prob_oof_cols)} auxiliary prob cols:", flush=True)
indiv = {}
for k in sorted(oof_cols):
    st = score_oof(oof_cols[k], splits_5)
    indiv[k] = st
    print(f"  {k:45s} | RMSE={st['rmse']:.4f} | QWK_q={st['qwk_q']:.4f} | QWK_nm={st['qwk_nm']:.4f}", flush=True)

# Filter out redundant sub-heads if blend is strictly better? Keep all diverse models in matrix!
keys = sorted(oof_cols.keys())
M_oof = np.column_stack([oof_cols[k] for k in keys])
M_test = np.column_stack([test_cols[k] for k in keys])

# Rank-transformed versions (1st place Wodori trick!)
R_oof = np.column_stack([rankdata(oof_cols[k]) / len(y_all) for k in keys])
R_test = np.column_stack([rankdata(test_cols[k]) / len(test_raw) for k in keys])

# Context features for Context-Aware Gated Meta-Stacker (zero leakage!)
def ctx_feats(df):
    resc = df["RescuerID"].astype(str)
    rc = resc.map(resc.value_counts()).astype(np.float32)
    desc_len = df["Description"].fillna("").astype(str).str.len().astype(np.float32)
    return np.column_stack([
        (df["PhotoAmt"] > 0).astype(np.float32).values,
        np.log1p(df["PhotoAmt"].clip(lower=0).astype(np.float32).values),
        (desc_len == 0).astype(np.float32).values,
        np.log1p(desc_len.values),
        (df["Type"] == 1).astype(np.float32).values,
        np.log1p(df["Age"].clip(lower=0).astype(np.float32).values),
        np.log1p(rc.values),
    ])

C_tr = ctx_feats(train_raw)
C_te = ctx_feats(test_raw)

# Include auxiliary class probs if available
P_oof = np.column_stack([prob_oof_cols[k] for k in sorted(prob_oof_cols)]) if prob_oof_cols else None
P_test = np.column_stack([prob_test_cols[k] for k in sorted(prob_test_cols)]) if prob_test_cols else None

# ---- Meta-Model 1: Nested Non-Negative Convex Blend (RMSE) ----
oof_nnls = np.zeros(len(y_all))
w_folds = []
for tr, va in splits_5:
    def obj(w):
        w = np.maximum(w, 0)
        w = w / (w.sum() + 1e-12)
        return np.mean((M_oof[tr] @ w - y_all[tr]) ** 2) + 1e-4 * np.sum(w ** 2)
    res = minimize(obj, np.ones(len(keys)) / len(keys), method="SLSQP",
                   bounds=[(0, 1)] * len(keys), constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0})
    wf = np.maximum(res.x, 0)
    wf /= wf.sum()
    w_folds.append(wf)
    oof_nnls[va] = M_oof[va] @ wf
w_mean = np.mean(w_folds, axis=0)
w_mean /= w_mean.sum()
test_nnls = M_test @ w_mean

# ---- Meta-Model 2: Nested Rank Average with Non-Negative Weights (1st place!) ----
oof_rank = np.zeros(len(y_all))
for (tr, va), wf in zip(splits_5, w_folds):
    oof_rank[va] = R_oof[va] @ wf
test_rank = R_test @ w_mean

# ---- Meta-Model 3: Nested RidgeCV / BayesianRidge on Raw OOF + Auxiliary Probs ----
Xmeta_tr = np.hstack([M_oof, P_oof]) if P_oof is not None else M_oof
Xmeta_te = np.hstack([M_test, P_test]) if P_test is not None else M_test
oof_ridge = np.zeros(len(y_all))
test_ridge = np.zeros(len(test_raw))
for tr, va in splits_5:
    br = RidgeCV(alphas=np.logspace(0, 4, 25)).fit(Xmeta_tr[tr], y_all[tr])
    oof_ridge[va] = br.predict(Xmeta_tr[va])
    test_ridge += br.predict(Xmeta_te) / 5.0

# ---- Meta-Model 4: Context-Aware Gated Shallow LGBM Meta-Stacker ----
Xctx_tr = np.hstack([M_oof, R_oof, C_tr]) if P_oof is None else np.hstack([M_oof, R_oof, P_oof, C_tr])
Xctx_te = np.hstack([M_test, R_test, C_te]) if P_test is None else np.hstack([M_test, R_test, P_test, C_te])
oof_ctx = np.zeros(len(y_all))
test_ctx = np.zeros(len(test_raw))
for tr, va in splits_5:
    pv_s, pt_s = [], []
    for s in (42, 100, 2024):
        m = LGBMRegressor(
            n_estimators=180, learning_rate=0.02, num_leaves=6, max_depth=3, min_child_samples=80,
            subsample=0.75, subsample_freq=1, colsample_bytree=0.6, reg_alpha=5.0, reg_lambda=25.0,
            random_state=s, verbose=-1
        )
        m.fit(Xctx_tr[tr], y_all[tr])
        pv_s.append(m.predict(Xctx_tr[va]))
        pt_s.append(m.predict(Xctx_te))
    oof_ctx[va] = np.mean(pv_s, axis=0)
    test_ctx += np.mean(pt_s, axis=0) / 5.0

# ---- Meta-Model 5: Super-Blend (Rank of Ridge + Rank of NNLS + Rank of Context-LGBM) ----
oof_super = 0.40 * (rankdata(oof_ridge) / len(y_all)) + 0.35 * oof_rank + 0.25 * (rankdata(oof_ctx) / len(y_all))
test_super = 0.40 * (rankdata(test_ridge) / len(test_raw)) + 0.35 * test_rank + 0.25 * (rankdata(test_ctx) / len(test_raw))

# ---- Meta-Model 6: Pure Linear Rank Blend (50% Ridge-rank + 50% NNLS-rank — ultra low overfitting!) ----
oof_linrank = 0.50 * (rankdata(oof_ridge) / len(y_all)) + 0.50 * oof_rank
test_linrank = 0.50 * (rankdata(test_ridge) / len(test_raw)) + 0.50 * test_rank

cands = {
    "meta_nnls": (oof_nnls, test_nnls),
    "meta_rank": (oof_rank, test_rank),
    "meta_ridge": (oof_ridge, test_ridge),
    "meta_ctx_lgbm": (oof_ctx, test_ctx),
    "meta_linrank": (oof_linrank, test_linrank),
    "meta_super": (oof_super, test_super),
}

print("\n=== Level-2 Meta-Stacker Comparison (5-fold StratifiedGroupKFold) ===", flush=True)
meta_res = {}
best_name, best_score, best_oof, best_test = None, -1.0, None, None
for cname, (coof, ctest) in cands.items():
    st = score_oof(coof, splits_5)
    meta_res[cname] = st
    print(f"  [{cname:15s}] QWK_q={st['qwk_q']:.4f} +/- {st['std_q']:.4f} | QWK_nm={st['qwk_nm']:.4f} | QWK_sh={st['qwk_sh']:.4f} | folds_q={st['folds_q']}", flush=True)
    if st["qwk_q"] > best_score:
        best_score, best_name, best_oof, best_test = st["qwk_q"], cname, coof, ctest

weights_dict = {k: round(float(w), 4) for k, w in zip(keys, w_mean) if w > 0.005}
print("\nTop NNLS weights:", weights_dict, flush=True)
print(f"Selected best meta-model: {best_name} (QWK_q={best_score:.4f})", flush=True)

# Generate final submission (quantile matching — proven zero-overfit on Private LB!)
final_preds = pd.Series(
    np.digitize(best_test, cortes_cuantiles(best_test, y_all)),
    index=test_raw.index, name="AdoptionSpeed"
).astype(int)
final_preds.to_csv(f"{OUT}/submission.csv")

# Also save shrinkage-threshold submission for comparison if desired
th_nm_full = opt_rounder_fit(best_oof, y_all)
th_q_full = cortes_cuantiles(best_test, y_all)
final_preds_sh = pd.Series(
    np.digitize(best_test, 0.65 * th_q_full + 0.35 * th_nm_full),
    index=test_raw.index, name="AdoptionSpeed"
).astype(int)
final_preds_sh.to_csv(f"{OUT}/submission_shrink.csv")

res = {
    "exp": CONFIG["exp"], "best_meta": best_name, "best_qwk_q": best_score,
    "best_std_q": meta_res[best_name]["std_q"], "best_folds_q": meta_res[best_name]["folds_q"],
    "best_qwk_nm": meta_res[best_name]["qwk_nm"], "best_qwk_sh": meta_res[best_name]["qwk_sh"],
    "weights": weights_dict, "meta_res": meta_res, "indiv": indiv,
    "class_dist": final_preds.value_counts().sort_index().to_dict(),
    "elapsed_sec": round(time.time() - t0, 1),
}
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
open(f"{OUT}/resultado.txt", "w").write(
    f"{CONFIG['exp']} | best={best_name} | QWK_q={best_score:.4f} +/- {meta_res[best_name]['std_q']:.4f} | "
    f"QWK_sh={meta_res[best_name]['qwk_sh']:.4f} | QWK_nm={meta_res[best_name]['qwk_nm']:.4f}\n"
    f"folds_q: {meta_res[best_name]['folds_q']}\nweights: {weights_dict}\n"
)
print("LISTO", sorted(os.listdir(OUT)), flush=True)
