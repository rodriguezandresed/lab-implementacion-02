# Kernel del modelo (lo sube orquestador.py; CONFIG se inyecta arriba de este archivo).
#
# Es EXP-026 (README: mejor modelo, private 0.42045) con UNA cosa parametrizada: de donde salen las columnas
# img_feat_*. Corre dos veces el mismo pipeline con los folds agrupados por RescuerID:
#   baseline = EXP-026 con CONFIG["img_base"] (EfficientNetB0 foto 1)
#   variante = EXP-026 con CONFIG["img_exp"]  (el embedding nuevo)
# y despues entrena la variante con KFold(5) para el submission (igual que EXP-025/026).
# Deja en /kaggle/working: submission.csv, results.json y resultado.txt.
import glob, json, os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold, StratifiedGroupKFold
from lightgbm import LGBMRegressor

try:
    CONFIG
except NameError:
    CONFIG = {"exp": "EXP-031", "n_svd": 20,
              "base": {"img": "effb0_first", "emo": "catedra", "drop": []},
              "variante": {"img": "dinov2-s_first", "emo": "catedra", "drop": []}}
# compatibilidad con la config vieja (EXP-030 a 032): solo cambiaba la imagen
CONFIG.setdefault("base", {"img": CONFIG.get("img_base"), "emo": "catedra", "drop": []})
CONFIG.setdefault("variante", {"img": CONFIG.get("img_exp"), "emo": "catedra", "drop": []})
OUT = "/kaggle/working"
N_SVD = CONFIG.get("n_svd", 20)
BEST_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, colsample_bytree=0.6, random_state=42, verbose=-1)


def buscar(archivo):
    for prof in range(0, 5):
        for d in glob.glob("/kaggle/input/" + "*/" * prof):
            if os.path.exists(os.path.join(d, archivo)):
                return os.path.join(d, archivo)
    raise FileNotFoundError(f"{archivo} no esta en /kaggle/input: {sorted(glob.glob('/kaggle/input/*/*'))[:60]}")


# ---------------------------------------------------------------- datos (celdas 2 a 5 del baseline)
train = pd.read_csv(buscar("train/train.csv")).set_index("PetID")
test = pd.read_csv(buscar("test/test.csv")).set_index("PetID")
folds = [train.index[v] for _, v in KFold(5).split(train)]  # los del baseline, para el submission
for c in ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'FurLength', 'MaturitySize',
          'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'RescuerID']:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")

def emociones(fuente, parte):
    """catedra = 7 emociones de DistilRoBERTa (nickel, train.parquet); otra = emo_<fuente>_<parte>.parquet de lab2-prep-emo-*."""
    return pd.read_parquet(buscar(f"{parte}.parquet" if fuente == "catedra" else f"emo_{fuente}_{parte}.parquet"))


def armar(df, parte, spec):
    """Mismo orden de columnas que EXP-026: tabulares, emociones, img_feat_*, txt_stats, img_stats, rescuer_count.
    spec["drop"]: columnas que no entran al modelo (se sacan al final: RescuerID sigue sirviendo para rescuer_count)."""
    df = df.join(emociones(spec.get("emo", "catedra"), parte))
    df = df.join(pd.read_parquet(buscar(f"img_{spec['img']}_{parte}.parquet"))).select_dtypes(exclude="O")
    df = df.join(pd.read_parquet(buscar(f"txt_stats_{parte}.parquet"))).join(pd.read_parquet(buscar(f"img_stats_{parte}.parquet")))
    rescatista = df["RescuerID"].astype(str)
    df["rescuer_count"] = rescatista.map(rescatista.value_counts()).astype(int)
    return df.drop(columns=spec.get("drop", []))


# ---------------------------------------------------------------- txt_score (EXP-022), opcional por spec["txt_score"]
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
GRUPOS = train["RescuerID"].astype(str)  # antes de cualquier drop: sirve aunque la variante no use RescuerID
_EMB = {}


def embedding(nombre):
    if nombre not in _EMB:
        _EMB[nombre] = pd.concat([pd.read_parquet(buscar(f"txt_emb_{nombre}_{p}.parquet")) for p in ("train", "test")]).dropna()
    return _EMB[nombre]


def _ridge(EMB, idx_fit, y, idx_pred):
    idx_fit = idx_fit[idx_fit.isin(EMB.index)]
    m = Ridge(alpha=10.0).fit(EMB.loc[idx_fit].values, y.loc[idx_fit].values)
    out = pd.Series(np.nan, index=idx_pred)
    ok = idx_pred[idx_pred.isin(EMB.index)]
    if len(ok):
        out.loc[ok] = m.predict(EMB.loc[ok].values)
    return out


def txt_score(nombre, X_train, y_train, X_valid, X_test):
    """En el train del fold el score es out-of-fold (GroupKFold(5) por RescuerID adentro del train);
    en valid y test, el ridge ajustado con todo el train del fold. No mira valid ni test para ajustar."""
    EMB = embedding(nombre)
    oof = pd.Series(np.nan, index=X_train.index)
    for a, b in GroupKFold(n_splits=5).split(X_train, groups=GRUPOS.loc[X_train.index].values):
        oof.iloc[b] = _ridge(EMB, X_train.index[a], y_train, X_train.index[b]).values
    return (X_train.assign(txt_score=oof.values),
            X_valid.assign(txt_score=_ridge(EMB, X_train.index, y_train, X_valid.index).values),
            X_test.assign(txt_score=_ridge(EMB, X_train.index, y_train, X_test.index).values))


# ---------------------------------------------------------------- pipeline de EXP-026 (celda 7)
def reducir(X, svd, IMG):
    Z = pd.DataFrame(np.nan, index=X.index, columns=[f"img_svd_{i}" for i in range(N_SVD)])
    tiene = X[IMG].notna().all(axis=1)
    if tiene.any():
        Z.loc[tiene] = svd.transform(X.loc[tiene, IMG].values)
    return X.drop(columns=IMG).join(Z)


def cortes_cuantiles(p, y):
    props = y.value_counts(normalize=True).sort_index().reindex(range(5), fill_value=0).values
    return np.quantile(p, np.cumsum(props)[:-1])


def correr(train, test, folds, spec):
    IMG = [c for c in train.columns if c.startswith("img_feat_")]
    fold_preds, test_out = [], []
    for fold in folds:
        X_train = train.drop(fold).drop("AdoptionSpeed", axis=1).select_dtypes(exclude="O")
        y_train = train.drop(fold)["AdoptionSpeed"]
        X_valid = train.loc[fold].drop("AdoptionSpeed", axis=1).select_dtypes(exclude="O")
        X_test = test[X_train.columns]
        svd = TruncatedSVD(n_components=N_SVD, random_state=42)  # fit solo con el train del fold
        svd.fit(X_train.loc[X_train[IMG].notna().all(axis=1), IMG].values)
        X_train, X_valid, X_test = reducir(X_train, svd, IMG), reducir(X_valid, svd, IMG), reducir(X_test, svd, IMG)
        if spec.get("txt_score"):
            X_train, X_valid, X_test = txt_score(spec["txt_score"], X_train, y_train, X_valid, X_test)
        learner = LGBMRegressor(**BEST_PARAMS)
        learner.fit(X_train, y_train)
        p_valid = learner.predict(X_valid)
        fold_preds.append(pd.Series(np.digitize(p_valid, cortes_cuantiles(p_valid, y_train)), index=fold))
        test_out.append(pd.Series(learner.predict(X_test), index=test.index))
    p_test = sum(test_out) / len(test_out)
    t = pd.Series(np.digitize(p_test, cortes_cuantiles(p_test, train["AdoptionSpeed"])), index=test.index)
    return pd.concat(fold_preds), t.rename("AdoptionSpeed"), len(IMG)


def qwk_por_fold(pred, folds):
    return [cohen_kappa_score(train.loc[f, "AdoptionSpeed"], pred.loc[f], weights="quadratic") for f in folds]


# ---------------------------------------------------------------- corrida
folds_g = [train.index[v] for _, v in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
           .split(train, train["AdoptionSpeed"], groups=train["RescuerID"])]

B, V = CONFIG["base"], CONFIG["variante"]
train_b, test_b = armar(train, "train", B), armar(test, "test", B)
train_v, test_v = armar(train, "train", V), armar(test, "test", V)
print("columnas base", train_b.shape[1] - 1, "| variante", train_v.shape[1] - 1,
      "| solo en base:", sorted(set(train_b.columns) - set(train_v.columns))[:20],
      "| solo en variante:", sorted(set(train_v.columns) - set(train_b.columns))[:20], flush=True)
assert len(train_b) == len(train_v) == len(train) and len(test_b) == len(test_v) == len(test)
sin_img = {n: int(d.filter(like="img_feat_").isna().all(axis=1).sum()) for n, d in
           [("train_base", train_b), ("test_base", test_b), ("train_exp", train_v), ("test_exp", test_v)]}
print("mascotas sin embedding:", sin_img, flush=True)

pred_base, _, d_base = correr(train_b, test_b, folds_g, B)
pred_exp, _, d_exp = correr(train_v, test_v, folds_g, V)
pred_k, test_preds, _ = correr(train_v, test_v, folds, V)  # submission: variante con KFold(5), como EXP-026

base, exp = qwk_por_fold(pred_base, folds_g), qwk_por_fold(pred_exp, folds_g)
delta = [e - b for b, e in zip(base, exp)]
res = {
    "exp": CONFIG["exp"], "config": CONFIG,
    "cv": "StratifiedGroupKFold(5, shuffle=True, random_state=42) groups=RescuerID",
    "img_dims": {"base": d_base, "exp": d_exp}, "sin_img": sin_img,
    "qwk_folds_base": base, "qwk_folds_exp": exp, "delta_folds": delta,
    "qwk_mean_base": float(np.mean(base)), "qwk_std_base": float(np.std(base, ddof=1)),
    "qwk_mean_exp": float(np.mean(exp)), "qwk_std_exp": float(np.std(exp, ddof=1)),
    "delta_mean": float(np.mean(delta)), "folds_mejoran": int(sum(d > 0 for d in delta)),
    "qwk_kfold_submission": float(np.mean(qwk_por_fold(pred_k, folds))),
    "params": BEST_PARAMS, "n_svd": N_SVD,
}
tabla = pd.DataFrame({"baseline": base, "exp": exp, "delta": delta}, index=[f"fold_{i}" for i in range(5)])
txt = "\n".join([
    f"{CONFIG['exp']}: variante {V} | baseline {B}",
    "folds agrupados por RescuerID",
    tabla.round(4).to_string(),
    f"media baseline {res['qwk_mean_base']:.4f} desvio {res['qwk_std_base']:.4f}",
    f"media exp      {res['qwk_mean_exp']:.4f} desvio {res['qwk_std_exp']:.4f}",
    f"delta media    {res['delta_mean']:+.4f}",
    f"folds que mejoran: {res['folds_mejoran']} de 5",
    f"CV KFold(5) del modelo de la submission: {res['qwk_kfold_submission']:.4f} (EXP-026: 0.4693)",
])
print("== resultado ==\n" + txt, flush=True)
open(f"{OUT}/resultado.txt", "w").write(txt + "\n")
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
test_preds.to_csv(f"{OUT}/submission.csv")
print(test_preds.value_counts().sort_index().to_dict())
