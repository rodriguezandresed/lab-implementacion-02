"""EXP-037: arma y sube los kernels a la cuenta de la CLI de Kaggle (kaggle.json), espera y baja la salida.

Uso (desde la raiz del lab):
    python experiments/EXP-037/correr.py            # prep (si falta) + kernel de desempate + bajar salida
    python experiments/EXP-037/correr.py --armar    # solo arma exp-037.py (para probarlo)

exp-037.py = orquestador/kernels/modelo.py hasta "# ---- corrida" (el mismo pipeline) + la corrida de abajo.
"""
import json, os, subprocess, sys, tempfile, time

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(AQUI))
K = os.path.join(REPO, "orquestador", "kernels")
USER = "rodriguezandresed"
COMP = "petfinder-adoption-prediction"
TEXTO = "nickel/examen-lab-ii-preprocesado-texto"
PREP = ["stats", "dinov2-s", "txt-e5"]

SPECS = {
    "031": {"img": "dinov2-s_first", "emo": "catedra", "drop": [], "txt_score": ""},
    "032": {"img": "dinov2-s_mean", "emo": "catedra", "drop": [], "txt_score": ""},
    "033": {"img": "dinov2-s_first", "emo": "catedra", "drop": ["RescuerID"], "txt_score": ""},
    "035": {"img": "dinov2-s_first", "emo": "catedra", "drop": ["RescuerID"], "txt_score": "e5"},
}
PARES = [("032", "031"), ("033", "031"), ("035", "033")]
SEEDS = [42, 1, 2, 3, 4]

CORRIDA = '''
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
txt = "\\n".join(lineas)
print("== resultado ==\\n" + txt, flush=True)
open(f"{OUT}/resultado.txt", "w").write(txt + "\\n")
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
'''


def armar():
    src = open(os.path.join(K, "modelo.py"), encoding="utf-8").read()
    corte = src.index("# ---------------------------------------------------------------- corrida")
    config = {"exp": "EXP-037", "n_svd": 20, "specs": SPECS, "pares": PARES, "seeds": SEEDS}
    codigo = f"CONFIG = {config!r}\n\n" + src[:corte] + CORRIDA
    compile(codigo, "exp-037.py", "exec")
    open(os.path.join(AQUI, "exp-037.py"), "w", encoding="utf-8").write(codigo)
    return codigo


def kaggle(*args):
    env = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    return (p.stdout or "") + (p.stderr or "")


def push(slug, archivo, codigo, gpu, internet, fuentes):
    d = tempfile.mkdtemp(prefix=f"push-{slug}-")
    open(os.path.join(d, archivo), "w", encoding="utf-8").write(codigo)
    json.dump({"id": f"{USER}/{slug}", "title": slug, "code_file": archivo, "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": gpu, "enable_internet": internet, "dataset_sources": [],
               "competition_sources": [COMP], "kernel_sources": fuentes}, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    out = kaggle("kernels", "push", "-p", d)
    print(slug, "->", out.strip().splitlines()[-1] if out.strip() else out, flush=True)
    if "error" in out.lower() and "successfully" not in out.lower():
        raise SystemExit(out)


def estado(slug):
    out = kaggle("kernels", "status", f"{USER}/{slug}").lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in out:
            return s
    return "no existe" if "404" in out or "not found" in out else out.strip()[-80:]


def esperar(slugs, limite_min=240):
    t0, pend = time.time(), set(slugs)
    while pend:
        time.sleep(30)
        for s in sorted(pend):
            st = estado(s)
            if st in ("complete", "error", "cancel"):
                print(s, st, flush=True)
                if st != "complete":
                    raise SystemExit(f"{s} termino en {st}: https://www.kaggle.com/code/{USER}/{s}")
                pend.discard(s)
        if time.time() - t0 > limite_min * 60:
            raise SystemExit(f"limite de {limite_min} min esperando {pend}")


def main():
    codigo = armar()
    if "--armar" in sys.argv:
        print("exp-037.py armado")
        return
    prep_src = open(os.path.join(K, "prep.py"), encoding="utf-8").read()
    lanzados = []
    for m in PREP:
        slug = f"lab2-prep-{m}"
        st = estado(slug)
        if st == "complete":
            print(slug, "ya esta: se reusa", flush=True)
            continue
        if st not in ("running", "queued"):
            gpu = m != "stats"
            push(slug, "prep.py", f"CONFIG = {{'modelo': {m!r}, 'solo_primera': False}}\n\n" + prep_src, gpu, gpu, [])
        lanzados.append(slug)
    if lanzados:
        esperar(lanzados)
    push("lab2-exp-037", "exp-037.py", codigo, False, False, [TEXTO] + [f"{USER}/lab2-prep-{m}" for m in PREP])
    esperar(["lab2-exp-037"], limite_min=180)
    salida = os.path.join(AQUI, "salida")
    os.makedirs(salida, exist_ok=True)
    print(kaggle("kernels", "output", f"{USER}/lab2-exp-037", "-p", salida)[-300:])
    print(open(os.path.join(salida, "resultado.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
