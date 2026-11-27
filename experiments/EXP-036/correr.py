"""EXP-036: EXP-033 con emociones traducidas, medido como EXP-037 (5 particiones) + submission.

Uso (desde la raiz del lab):
    python experiments/EXP-036/correr.py            # sube el kernel, espera, baja la salida
    python experiments/EXP-036/correr.py --armar    # solo arma exp-036.py

Reusa experiments/EXP-037/correr.py: mismo pipeline (modelo.py hasta "corrida") y misma corrida de 5 seeds.
"""
import importlib.util, os, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(AQUI))
spec = importlib.util.spec_from_file_location("e37", os.path.join(REPO, "experiments", "EXP-037", "correr.py"))
e37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e37)

TRAD = "examen-lab-ii-preprocesado-texto-trad"
SPECS = {
    "033": {"img": "dinov2-s_first", "emo": "catedra", "drop": ["RescuerID"], "txt_score": ""},
    "036": {"img": "dinov2-s_first", "emo": "trad", "drop": ["RescuerID"], "txt_score": ""},
}

# emo "trad": las de la catedra, con las mascotas traducidas reemplazadas (emo_trad_*.parquet de texto-trad)
EMO_TRAD = '''
# ---------------------------------------------------------------- EXP-036: emociones traducidas
_emociones_catedra = emociones
def emociones(fuente, parte):
    if fuente != "trad":
        return _emociones_catedra(fuente, parte)
    e = _emociones_catedra("catedra", parte).copy()
    t = pd.read_parquet(buscar(f"emo_trad_{parte}.parquet"))
    assert set(t.columns) == set(e.columns) and t.index.isin(e.index).all(), (list(t.columns), list(e.columns))
    e.loc[t.index, t.columns] = t.values
    print("emociones traducidas:", parte, len(t), "filas reemplazadas", flush=True)
    return e

'''

SUBMISSION = '''
# ---------------------------------------------------------------- submission: la variante con KFold(5), como el orquestador
tr, te = datos["036"]
pred_k, test_preds, _ = correr(tr, te, folds, SPECS["036"])
kf = float(np.mean(qwk_por_fold(pred_k, folds)))
print(f"CV KFold(5) del modelo de la submission: {kf:.4f} (EXP-033: 0.4623)", flush=True)
open(f"{OUT}/resultado.txt", "a").write(f"CV KFold(5) del modelo de la submission: {kf:.4f} (EXP-033: 0.4623)\\n")
test_preds.to_csv(f"{OUT}/submission.csv")
print(test_preds.value_counts().sort_index().to_dict())
'''


def armar():
    src = open(os.path.join(e37.K, "modelo.py"), encoding="utf-8").read()
    corte = src.index("# ---------------------------------------------------------------- corrida")
    config = {"exp": "EXP-036", "n_svd": 20, "specs": SPECS, "pares": [("036", "033")], "seeds": e37.SEEDS}
    codigo = f"CONFIG = {config!r}\n\n" + src[:corte] + EMO_TRAD + e37.CORRIDA + SUBMISSION
    compile(codigo, "exp-036.py", "exec")
    open(os.path.join(AQUI, "exp-036.py"), "w", encoding="utf-8").write(codigo)
    return codigo


def main():
    codigo = armar()
    if "--armar" in sys.argv:
        print("exp-036.py armado")
        return
    fuentes = [e37.TEXTO, f"{e37.USER}/{TRAD}"] + [f"{e37.USER}/lab2-prep-{m}" for m in ("stats", "dinov2-s")]
    e37.push("lab2-exp-036", "exp-036.py", codigo, False, False, fuentes)
    e37.esperar(["lab2-exp-036"], limite_min=120)
    salida = os.path.join(AQUI, "salida")
    os.makedirs(salida, exist_ok=True)
    print(e37.kaggle("kernels", "output", f"{e37.USER}/lab2-exp-036", "-p", salida)[-300:])
    print(open(os.path.join(salida, "resultado.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
