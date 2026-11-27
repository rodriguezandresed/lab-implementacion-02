CONFIG = {'exp': 'EXP-039', 'n_svd': 20, 'specs': {'033': {'img': 'dinov2-s_first', 'emo': 'catedra', 'drop': ['RescuerID'], 'txt_score': ''}, '039': {'img': 'dinov2-b_first', 'emo': 'catedra', 'drop': ['RescuerID'], 'txt_score': ''}}, 'pares': [('039', '033')], 'seeds': [42, 1, 2, 3, 4], 'variante': '039'}

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



# ---------------------------------------------------------------- cola: params y promedio de seeds por spec
_correr_original, _BP, _LGBM = correr, dict(BEST_PARAMS), LGBMRegressor
class _PromedioSeeds:
    """n LGBMRegressor con random_state 42..42+n-1; predice el promedio."""
    def __init__(self, n, **p):
        self.ms = [_LGBM(**{**p, "random_state": p.get("random_state", 42) + i}) for i in range(n)]
    def fit(self, X, y):
        for m in self.ms:
            m.fit(X, y)
        return self
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ms], axis=0)

def correr(train, test, folds, spec):
    global BEST_PARAMS, LGBMRegressor
    BEST_PARAMS = dict(spec.get("params") or _BP)
    n = spec.get("lgbm_seeds", 1)
    LGBMRegressor = (lambda **p: _PromedioSeeds(n, **p)) if n > 1 else _LGBM
    try:
        return _correr_original(train, test, folds, spec)
    finally:
        BEST_PARAMS, LGBMRegressor = _BP, _LGBM

def _folds_seed(s):
    return [train.index[v] for _, v in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
            .split(train, train["AdoptionSpeed"], groups=train["RescuerID"])]

# ---------------------------------------------------------------- corrida EXP-037 (desempate con 5 particiones)
SPECS, PARES, SEEDS = CONFIG["specs"], CONFIG["pares"], CONFIG["seeds"]
datos = {n: (armar(train, "train", s), armar(test, "test", s)) for n, s in SPECS.items()}
for n, (tr, te) in datos.items():
    print(n, "columnas", tr.shape[1] - 1, "sin embedding:", int(tr.filter(like="img_feat_").isna().all(axis=1).sum()), flush=True)
qwk = {}  # (spec, seed) -> 5 qwk
for s in SEEDS:
    fg = [train.index[v] for _, v in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
          .split(train, train["AdoptionSpeed"], groups=train["RescuerID"])]
    for n, spec in SPECS.items():
        pred, _, _ = correr(datos[n][0], datos[n][1], fg, spec)
        qwk[(n, s)] = qwk_por_fold(pred, fg)
        print(f"seed {s} spec {n}: media {np.mean(qwk[(n, s)]):.4f}", flush=True)

lineas, res = [], {"seeds": SEEDS, "specs": SPECS, "qwk": {f"{n}_{s}": v for (n, s), v in qwk.items()}, "pares": {}}
medias = pd.DataFrame({n: [np.mean(qwk[(n, s)]) for s in SEEDS] for n in SPECS}, index=[f"seed_{s}" for s in SEEDS])
lineas += ["QWK medio por seed (folds agrupados por RescuerID)", medias.round(4).to_string(),
           "promedio " + "  ".join(f"{n} {medias[n].mean():.4f}" for n in SPECS), ""]
for v, b in PARES:
    d_fold = np.array([np.array(qwk[(v, s)]) - np.array(qwk[(b, s)]) for s in SEEDS])  # seeds x folds
    d_seed = d_fold.mean(axis=1)
    r = {"delta_media": float(d_fold.mean()), "delta_min_seed": float(d_seed.min()), "delta_max_seed": float(d_seed.max()),
         "seeds_mejoran": int((d_seed > 0).sum()), "seeds_empeoran": int((d_seed < 0).sum()),
         "folds_mejoran": int((d_fold > 0).sum()), "delta_por_seed": d_seed.tolist()}
    if r["delta_media"] >= 0.004 and r["seeds_mejoran"] >= 4:
        r["decision"] = "gana la variante"
    elif r["delta_media"] <= -0.004 or r["seeds_empeoran"] >= 4:
        r["decision"] = "pierde la variante"
    else:
        r["decision"] = "empate"
    res["pares"][f"{v}_vs_{b}"] = r
    lineas.append(f"{v} vs {b}: delta {r['delta_media']:+.4f} (por seed {r['delta_min_seed']:+.4f} a {r['delta_max_seed']:+.4f}), "
                  f"mejora en {r['seeds_mejoran']}/5 seeds y {r['folds_mejoran']}/25 folds -> {r['decision']}")
    lineas.append("   por seed: " + "  ".join(f"{s}:{d:+.4f}" for s, d in zip(SEEDS, d_seed)))
txt = "\n".join(lineas)
print("== resultado ==\n" + txt, flush=True)
open(f"{OUT}/resultado.txt", "w").write(txt + "\n")
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)

# ---------------------------------------------------------------- submission: la variante con KFold(5), como el orquestador
VAR = CONFIG["variante"]
tr, te = datos[VAR]
pred_k, test_preds, _ = correr(tr, te, folds, SPECS[VAR])
kf = float(np.mean(qwk_por_fold(pred_k, folds)))
linea = f"CV KFold(5) del modelo de la submission: {kf:.4f} (EXP-033: 0.4623)"
print(linea, flush=True)
open(f"{OUT}/resultado.txt", "a").write(linea + "\n")
r = json.load(open(f"{OUT}/results.json")); r["qwk_kfold_submission"] = kf; r["specs_finales"] = SPECS
json.dump(r, open(f"{OUT}/results.json", "w"), indent=1)
test_preds.to_csv(f"{OUT}/submission.csv")
print(test_preds.value_counts().sort_index().to_dict())
