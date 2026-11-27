"""Cola de experimentos (EXP-038 a 043 sobre EXP-033, EXP-044 a 047 sobre EXP-040), medidos como EXP-037 (5 particiones agrupadas) + submission.

Uso (desde la raiz del lab):
    python experiments/cola.py                    # corre toda la cola: preps, modelos, submits, privates
    python experiments/cola.py --armar            # solo arma experiments/EXP-###/exp-###.py
    python experiments/cola.py EXP-041 EXP-042    # solo esos

Cada kernel = orquestador/kernels/modelo.py hasta "corrida" (mismo pipeline) + ENVOLTURA (params y promedio de seeds
por spec) + el codigo propio del experimento + la corrida de 5 seeds de EXP-037 + la submission de la variante.
Antes de la cola sube el submit de EXP-036 (si su kernel ya termino).
"""
import importlib.util, json, os, re, sys, time

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)
_s = importlib.util.spec_from_file_location("e37", os.path.join(AQUI, "EXP-037", "correr.py"))
e37 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(e37)
USER, COMP = e37.USER, e37.COMP

BASE = {"img": "dinov2-s_first", "emo": "catedra", "drop": ["RescuerID"], "txt_score": ""}
def v(**k):
    d = json.loads(json.dumps(BASE)); d.update(k); return d

ENVOLTURA = '''
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
'''

TUNE = '''
# ---------------------------------------------------------------- EXP-042: grid chico con la particion seed 42
from itertools import product
GRID = CONFIG.get("grid") or dict(num_leaves=[15, 31, 63], learning_rate=[0.02, 0.03, 0.05],
                                  n_estimators=[300, 600], min_child_samples=[20, 50])
_tr, _te = armar(train, "train", CONFIG["specs"]["033"]), armar(test, "test", CONFIG["specs"]["033"])
_fg = _folds_seed(42)
filas = []
for vals in product(*GRID.values()):
    p = {**_BP, **dict(zip(GRID, vals))}
    pred, _, _ = correr(_tr, _te, _fg, {**CONFIG["specs"]["033"], "params": p})
    filas.append({**dict(zip(GRID, vals)), "qwk": float(np.mean(qwk_por_fold(pred, _fg)))})
    print("grid", filas[-1], flush=True)
grid = pd.DataFrame(filas).sort_values("qwk", ascending=False)
print("== grid (seed 42) ==\\n" + grid.head(10).round(4).to_string(), flush=True)
mejor = {k: (int(v) if k != "learning_rate" else float(v)) for k, v in grid.iloc[0][list(GRID)].items()}
CONFIG["specs"]["042"]["params"] = {**_BP, **mejor}
print("params elegidos:", CONFIG["specs"]["042"]["params"], flush=True)
grid.to_csv(f"{OUT}/grid.csv", index=False)
'''

GA = '''
# ---------------------------------------------------------------- EXP-043: seleccion genetica de columnas (particion seed 42)
GA_CFG = {"pob": 20, "gen": 12, "elite": 2, "torneo": 3, "p_ini": 0.8, **CONFIG.get("ga", {})}
_spec = CONFIG["specs"]["033"]
_tr = armar(train, "train", _spec)
_fg = _folds_seed(42)
IMGC = [c for c in _tr.columns if c.startswith("img_feat_")]
GENES = [c for c in _tr.drop(columns="AdoptionSpeed").select_dtypes(exclude="O").columns if c not in IMGC]
print("genes:", len(GENES), GENES, flush=True)
# por fold: matrices ya reducidas (SVD ajustado en el train del fold); la imagen entra siempre
_cache_f = []
for fold in _fg:
    Xt = _tr.drop(fold).drop("AdoptionSpeed", axis=1).select_dtypes(exclude="O")
    Xv = _tr.loc[fold].drop("AdoptionSpeed", axis=1).select_dtypes(exclude="O")
    svd = TruncatedSVD(n_components=N_SVD, random_state=42).fit(Xt.loc[Xt[IMGC].notna().all(axis=1), IMGC].values)
    _cache_f.append((reducir(Xt, svd, IMGC), _tr.drop(fold)["AdoptionSpeed"], reducir(Xv, svd, IMGC), fold))
SVDC = [f"img_svd_{i}" for i in range(N_SVD)]
_memo = {}
def aptitud(mask):
    k = tuple(int(b) for b in mask)
    if k not in _memo:
        cols = [g for g, b in zip(GENES, k) if b] + SVDC
        q = []
        for Xt, yt, Xv, fold in _cache_f:
            m = _LGBM(**_BP).fit(Xt[cols], yt)
            p = m.predict(Xv[cols])
            q.append(cohen_kappa_score(train.loc[fold, "AdoptionSpeed"], np.digitize(p, cortes_cuantiles(p, yt)), weights="quadratic"))
        _memo[k] = float(np.mean(q))
    return _memo[k]
rng = np.random.default_rng(42)
n = len(GENES)
pob = [np.ones(n, dtype=int)] + [(rng.random(n) < GA_CFG["p_ini"]).astype(int) for _ in range(GA_CFG["pob"] - 1)]
for g in range(GA_CFG["gen"]):
    fit = [aptitud(m) for m in pob]
    orden = np.argsort(fit)[::-1]
    print(f"gen {g}: mejor {fit[orden[0]]:.4f} ({int(pob[orden[0]].sum())} genes) | todas las columnas {aptitud(np.ones(n, dtype=int)):.4f}", flush=True)
    nueva = [pob[i].copy() for i in orden[:GA_CFG["elite"]]]
    def torneo():
        idx = rng.choice(len(pob), GA_CFG["torneo"], replace=False)
        return pob[max(idx, key=lambda i: fit[i])]
    while len(nueva) < GA_CFG["pob"]:
        a, b = torneo(), torneo()
        hijo = np.where(rng.random(n) < 0.5, a, b)
        hijo = np.where(rng.random(n) < 1.0 / n, 1 - hijo, hijo)
        nueva.append(hijo)
    pob = nueva
fit = [aptitud(m) for m in pob]
mejor = pob[int(np.argmax(fit))]
sacadas = [gn for gn, b in zip(GENES, mejor) if not b]
print(f"GA: seed 42 {max(fit):.4f} contra {aptitud(np.ones(n, dtype=int)):.4f} con todas; saca {len(sacadas)}: {sacadas}", flush=True)
CONFIG["specs"]["043"]["drop"] = CONFIG["specs"]["043"]["drop"] + sacadas
json.dump({"genes": GENES, "sacadas": sacadas, "evaluaciones": len(_memo)}, open(f"{OUT}/ga.json", "w"), indent=1)
'''

SUBMISSION = '''
# ---------------------------------------------------------------- submission: la variante con KFold(5), como el orquestador
VAR = CONFIG["variante"]
tr, te = datos[VAR]
pred_k, test_preds, _ = correr(tr, te, folds, SPECS[VAR])
kf = float(np.mean(qwk_por_fold(pred_k, folds)))
linea = f"CV KFold(5) del modelo de la submission: {kf:.4f} ({CONFIG.get('ref_kfold', 'EXP-033: 0.4623')})"
print(linea, flush=True)
open(f"{OUT}/resultado.txt", "a").write(linea + "\\n")
r = json.load(open(f"{OUT}/results.json")); r["qwk_kfold_submission"] = kf; r["specs_finales"] = SPECS
json.dump(r, open(f"{OUT}/results.json", "w"), indent=1)
test_preds.to_csv(f"{OUT}/submission.csv")
print(test_preds.value_counts().sort_index().to_dict())
'''

EXPS = {
    "EXP-038": dict(desc="foto promedio + txt_score e5", var=v(img="dinov2-s_mean", txt_score="e5"),
                    preps=["stats", "dinov2-s", "txt-e5"], extra=""),
    "EXP-039": dict(desc="DINOv2-B foto 1", var=v(img="dinov2-b_first"), preps=["stats", "dinov2-s", "dinov2-b"], extra=""),
    "EXP-040": dict(desc="CLIP-L14 promedio de fotos", var=v(img="clip-l14_mean"), preps=["stats", "dinov2-s", "clip-l14"], extra=""),
    "EXP-041": dict(desc="promedio de 5 seeds de LightGBM", var=v(lgbm_seeds=5), preps=["stats", "dinov2-s"], extra=""),
    "EXP-042": dict(desc="hiperparametros re-tuneados", var=v(params=None), preps=["stats", "dinov2-s"], extra=TUNE),
    "EXP-043": dict(desc="seleccion genetica de columnas", var=v(), preps=["stats", "dinov2-s"], extra=GA),
}


# EXP-044 a 047: sobre la base nueva EXP-040 (CLIP-L/14 promedio de fotos)
BASE40 = v(img="clip-l14_mean")
PARAMS42 = {"n_estimators": 300, "learning_rate": 0.02, "num_leaves": 31, "colsample_bytree": 0.6,
            "random_state": 42, "verbose": -1, "min_child_samples": 50}  # los que eligio EXP-042
def v40(**k):
    d = json.loads(json.dumps(BASE40)); d.update(k); return d
P40 = ["stats", "clip-l14"]
EXPS.update({
    "EXP-044": dict(desc="txt_score e5", var=v40(txt_score="e5"), preps=P40 + ["txt-e5"], extra="", base="040"),
    "EXP-045": dict(desc="promedio de 5 seeds de LightGBM", var=v40(lgbm_seeds=5), preps=P40, extra="", base="040"),
    "EXP-046": dict(desc="hiperparametros de EXP-042", var=v40(params=PARAMS42), preps=P40, extra="", base="040"),
    "EXP-047": dict(desc="txt_score + 5 seeds + params de EXP-042 (todo junto)",
                    var=v40(txt_score="e5", lgbm_seeds=5, params=PARAMS42), preps=P40 + ["txt-e5"], extra="", base="040"),
})
# EXP-048: ablacion, EXP-040 sin el procesamiento de texto (7 emociones de la catedra + 8 txt_stats)
EMO7 = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
TXT8 = ["txt_largo", "txt_palabras", "txt_frac_chino", "txt_frac_malayo", "txt_frac_mayus", "txt_exclamaciones", "txt_telefono", "txt_precio"]
EXPS["EXP-048"] = dict(desc="sin texto (ablacion: sin las 7 emociones ni las 8 txt_stats)",
                       var=v40(drop=["RescuerID"] + EMO7 + TXT8), preps=P40, extra="", base="040")
BASES = {"033": {"spec": BASE, "kfold": "EXP-033: 0.4623", "private": "0.42944"},
         "040": {"spec": BASE40, "kfold": "EXP-040: 0.4683", "private": "0.44183"}}
def base_de(exp):
    bk = EXPS[exp].get("base", "033")
    return bk, BASES[bk]["spec"]


def armar(exp):
    e, n = EXPS[exp], exp[-3:]
    src = open(os.path.join(e37.K, "modelo.py"), encoding="utf-8").read()
    corte = src.index("# ---------------------------------------------------------------- corrida")
    bk, bspec = base_de(exp)
    config = {"exp": exp, "n_svd": 20, "specs": {bk: bspec, n: e["var"]}, "pares": [(n, bk)],
              "ref_kfold": BASES[bk]["kfold"],
              "seeds": e37.SEEDS, "variante": n}
    codigo = f"CONFIG = {config!r}\n\n" + src[:corte] + ENVOLTURA + e["extra"] + e37.CORRIDA + SUBMISSION
    compile(codigo, f"exp-{n}.py", "exec")
    d = os.path.join(AQUI, exp)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, f"exp-{n}.py"), "w", encoding="utf-8").write(codigo)
    return codigo


def push(slug, archivo, codigo, gpu, internet, fuentes):
    import tempfile
    d = tempfile.mkdtemp(prefix=f"push-{slug}-")
    open(os.path.join(d, archivo), "w", encoding="utf-8").write(codigo)
    json.dump({"id": f"{USER}/{slug}", "title": slug, "code_file": archivo, "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": gpu, "enable_internet": internet, "dataset_sources": [],
               "competition_sources": [COMP], "kernel_sources": fuentes}, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    for intento in range(20):
        out = e37.kaggle("kernels", "push", "-p", d)
        m = re.search(r"[Vv]ersion (\d+) successfully pushed", out)
        if m:
            print(slug, "-> version", m.group(1), flush=True)
            return int(m.group(1))
        print(slug, "push fallo, reintento en 2 min:", out.strip()[-200:], flush=True)
        time.sleep(120)
    raise SystemExit(f"{slug}: no se pudo subir")


def submit(slug, version, msg, salida):
    out = e37.kaggle("competitions", "submit", COMP, "-k", f"{USER}/{slug}", "-v", str(version), "-f", "submission.csv", "-m", msg)
    print("submit", slug, "v", version, "->", out.strip()[-200:], flush=True)
    open(os.path.join(salida, "resultado.txt"), "a", encoding="utf-8").write(f"submit: {msg} | {out.strip()[-120:]}\n")


def privates(mensajes, limite_min=60):
    """Espera el private de cada submit (por descripcion) y lo devuelve."""
    t0, res = time.time(), {}
    while time.time() - t0 < limite_min * 60:
        out = e37.kaggle("competitions", "submissions", COMP, "--csv")
        lineas = out.splitlines()
        for msg in mensajes:
            for l in lineas:
                if msg in l and msg not in res:
                    campos = l.split(",")
                    nums = [c for c in campos if re.fullmatch(r"0\.\d+", c.strip())]
                    if len(nums) >= 2:
                        res[msg] = nums[-1]
        if len(res) == len(mensajes):
            break
        time.sleep(60)
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cola = args or list(EXPS)
    codigos = {exp: armar(exp) for exp in cola}
    if "--armar" in sys.argv:
        print("armados:", cola)
        return
    enviados = []
    # 0) submit de EXP-036 cuando su kernel termine
    if not args:
        while e37.estado("lab2-exp-036") in ("running", "queued"):
            time.sleep(30)
        s36 = os.path.join(AQUI, "EXP-036", "salida")
        if e37.estado("lab2-exp-036") == "complete" and os.path.exists(os.path.join(s36, "results.json")):
            r = json.load(open(os.path.join(s36, "results.json")))
            if r["pares"]["036_vs_033"]["delta_media"] <= 0.02:
                msg = "EXP-036 emociones traducidas (malayo/chino) sobre EXP-033"
                submit("lab2-exp-036", 1, msg, s36)
                enviados.append(("EXP-036", msg))
    # 1) preprocesamientos que falten (GPU para los modelos de imagen)
    prep_src = open(os.path.join(e37.K, "prep.py"), encoding="utf-8").read()
    faltan = sorted({p for exp in cola for p in EXPS[exp]["preps"] if e37.estado(f"lab2-prep-{p}") != "complete"})
    for p in faltan:
        if e37.estado(f"lab2-prep-{p}") not in ("running", "queued"):
            push(f"lab2-prep-{p}", "prep.py", f"CONFIG = {{'modelo': {p!r}, 'solo_primera': False}}\n\n" + prep_src,
                 p != "stats", p != "stats", [])
    # 2) modelos: se suben apenas sus preps estan; se espera a todos
    pendientes, versiones = list(cola), {}
    listo = lambda x: os.path.exists(os.path.join(AQUI, x, "salida", ".listo"))
    # se espera hasta bajar la salida de cada kernel subido: un estado raro (error de la API) no corta la espera
    while pendientes or not all(listo(x) for x in versiones):
        for exp in list(pendientes):
            preps = EXPS[exp]["preps"]
            st = [e37.estado(f"lab2-prep-{p}") for p in preps]
            if any(s in ("error", "cancel") for s in st):
                print(exp, "se saltea: fallo un preprocesamiento", dict(zip(preps, st)), flush=True)
                pendientes.remove(exp)
            elif all(s == "complete" for s in st):
                fuentes = [e37.TEXTO] + [f"{USER}/lab2-prep-{p}" for p in preps]
                versiones[exp] = push(f"lab2-{exp.lower()}", f"exp-{exp[-3:]}.py", codigos[exp], False, False, fuentes)
                pendientes.remove(exp)
        for exp in list(versiones):
            slug = f"lab2-{exp.lower()}"
            st = e37.estado(slug)
            if st in ("complete", "error", "cancel") and exp not in dict(enviados) and not os.path.exists(os.path.join(AQUI, exp, "salida", ".listo")):
                salida = os.path.join(AQUI, exp, "salida")
                os.makedirs(salida, exist_ok=True)
                print(exp, st, flush=True)
                e37.kaggle("kernels", "output", f"{USER}/{slug}", "-p", salida)
                open(os.path.join(salida, ".listo"), "w").write(st)
                if st != "complete" or not os.path.exists(os.path.join(salida, "results.json")):
                    print(exp, "termino sin resultado: ver", f"https://www.kaggle.com/code/{USER}/{slug}", flush=True)
                    continue
                print(open(os.path.join(salida, "resultado.txt"), encoding="utf-8").read(), flush=True)
                r = json.load(open(os.path.join(salida, "results.json")))
                if r["pares"][f"{exp[-3:]}_vs_{base_de(exp)[0]}"]["delta_media"] > 0.02:
                    print(exp, "delta > +0.02: regla de sospecha, NO se sube", flush=True)
                    continue
                msg = f"{exp} {EXPS[exp]['desc']} sobre EXP-{base_de(exp)[0]}"
                submit(slug, versiones[exp], msg, salida)
                enviados.append((exp, msg))
        time.sleep(30)
    # 3) privates
    pr = privates([m for _, m in enviados])
    for exp, msg in enviados:
        bk = base_de(exp)[0] if exp in EXPS else "033"
        linea = f"Kaggle: private {pr.get(msg, 'sin score todavia')} (EXP-{bk}: {BASES[bk]['private']}, mejor EXP-032: 0.44273)"
        print(exp, linea, flush=True)
        open(os.path.join(AQUI, exp, "salida", "resultado.txt"), "a", encoding="utf-8").write(linea + "\n")
    print("cola terminada", flush=True)


if __name__ == "__main__":
    main()
