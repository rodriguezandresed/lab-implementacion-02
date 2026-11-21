"""Corre experimentos en Kaggle respetando el límite de 2 sesiones GPU, baja la salida y anota log.csv.

Uso (desde la raíz del lab):
    python experiments/correr.py EXP-004 EXP-006       # esos, en ese orden
    python experiments/correr.py pendientes            # todos los que tienen notebook y no están en log.csv
    python experiments/correr.py --solo-bajar EXP-001  # no corre: baja la salida de la última versión y anota

Por experimento deja en experiments/EXP-###/salida/ el log de Kaggle, submission.csv y resultado.txt
(la tabla que imprime el notebook). Agrega una fila a experiments/log.csv con la decisión según el
criterio del template: aceptar si delta ≥ +0.004 y mejora en ≥ 4 de 5 folds; sospechar si delta > 0.02;
descartar si no. EXP-002 y EXP-016 cambian la medición: se anotan sin decisión.
"""
import csv, datetime, glob, json, os, re, subprocess, sys, time

AQUI = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(AQUI, "log.csv")
MAX_GPU = 2
ESPERA = 60


def kaggle(*args):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout + r.stderr).strip()


def meta(exp):
    return json.load(open(os.path.join(AQUI, exp, "kernel-metadata.json"), encoding="utf-8"))


def estado(exp):
    txt = kaggle("kernels", "status", meta(exp)["id"])
    m = re.search(r"KernelWorkerStatus\.(\w+)", txt)
    return m.group(1) if m else txt


def ya_anotados():
    if not os.path.exists(LOG):
        return set()
    return {r["exp_id"] for r in csv.DictReader(open(LOG, encoding="utf-8"))}


def pendientes():
    exps = sorted(os.path.basename(d) for d in glob.glob(os.path.join(AQUI, "EXP-*"))
                  if os.path.exists(os.path.join(d, "kernel-metadata.json")))
    hechos = ya_anotados()
    return [e for e in exps if e not in hechos]


def numero(patron, txt):
    m = re.search(patron, txt)
    return float(m.group(1)) if m else None


def recortar(texto):
    """Devuelve solo la tabla de resultados: desde el encabezado de la tabla hasta antes del head del submission."""
    lineas = texto.splitlines()
    marca = next((k for k, l in enumerate(lineas) if l.startswith("== resultado ==")), None)
    if marca is not None:
        fin = next((k for k in range(marca, len(lineas)) if lineas[k].startswith(("PetID,", "[NbConvertApp]", "/usr/"))), len(lineas))
        return "\n".join(lineas[marca + 1:fin]).strip() + "\n"
    ini = next((k for k, l in enumerate(lineas) if l.startswith("fold_0")), None)
    if ini is None:
        return texto[-3000:]
    fin = next((k for k in range(ini, len(lineas)) if lineas[k].startswith(("PetID,", "[NbConvertApp]", "/usr/"))), len(lineas))
    return "\n".join(lineas[max(0, ini - 1):fin]).strip() + "\n"


def bajar_y_anotar(exp):
    salida = os.path.join(AQUI, exp, "salida")
    os.makedirs(salida, exist_ok=True)
    print(f"[{exp}] bajando salida", flush=True)
    print(kaggle("kernels", "output", meta(exp)["id"], "-p", salida, "-o"), flush=True)
    logs = glob.glob(os.path.join(salida, "*.log"))
    texto = ""
    if logs:
        try:
            texto = "".join(e.get("data", "") for e in json.load(open(logs[0], encoding="utf-8")))
        except json.JSONDecodeError:
            texto = open(logs[0], encoding="utf-8", errors="replace").read()

    if "Exception encountered" in texto or ("Traceback" in texto and "media" not in texto):
        i = texto.find("Exception encountered")
        print(f"[{exp}] ERROR en el notebook:\n{texto[i:i + 1500]}", flush=True)
        return "error"

    bloque = recortar(texto)
    open(os.path.join(salida, "resultado.txt"), "w", encoding="utf-8").write(bloque)
    print(f"[{exp}] resultado:\n{bloque}", flush=True)

    n = exp.split("-")[1]
    if exp == "EXP-002":
        media = numero(r"media\s+([-\d.]+)\s+desvio", bloque)
        desvio = numero(r"media\s+[-\d.]+\s+desvio\s+([-\d.]+)", bloque)
        decision = "sin decision (cambia la medicion, no el modelo)"
    elif exp == "EXP-016":
        media = desvio = None
        decision = "sin decision (mide el ruido de la particion): ver salida/resultado.txt"
    else:
        media = numero(rf"media exp_{n}\s+([-\d.]+)", bloque)
        desvio = numero(rf"media exp_{n}\s+[-\d.]+\s+desvio\s+([-\d.]+)", bloque)
        desvio_base = numero(r"media baseline\s+[-\d.]+\s+desvio\s+([-\d.]+)", bloque)
        delta = numero(r"delta media\s+([-\d.]+)", bloque)
        mejoran = numero(r"folds que mejoran:\s+(\d+)", bloque)
        if None in (media, delta, mejoran):
            decision = "no se pudo leer el resultado: revisar salida/resultado.txt"
        elif delta > 0.02:
            decision = f"sospechar y frenar (delta {delta:+.4f} > 0.02)"
        elif delta >= 0.004 and mejoran >= 4 and not (desvio_base is not None and desvio > desvio_base):
            decision = f"aceptar (delta {delta:+.4f}, mejora en {int(mejoran)}/5)"
        else:
            decision = f"descartar (delta {delta:+.4f}, mejora en {int(mejoran)}/5" + (
                ", sube el desvio)" if desvio_base is not None and desvio > desvio_base else ")")

    nuevo = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["exp_id", "fecha", "descripcion", "qwk_cv_mean", "qwk_cv_std", "decision"])
        w.writerow([exp, datetime.date.today().isoformat(), meta(exp)["title"], media, desvio, decision])
    print(f"[{exp}] log.csv: {decision}", flush=True)
    return "ok"


def correr(cola):
    # los que ya están corriendo en Kaggle no se relanzan: se esperan y se bajan
    corriendo = [e for e in cola if estado(e) in ("RUNNING", "QUEUED")]
    cola = [e for e in cola if e not in corriendo]
    if corriendo:
        print("ya corriendo en Kaggle, se esperan:", corriendo, flush=True)
    while cola or corriendo:
        for exp in list(corriendo):
            st = estado(exp)
            if st not in ("RUNNING", "QUEUED"):
                print(f"[{exp}] terminó: {st}", flush=True)
                corriendo.remove(exp)
                bajar_y_anotar(exp)
        while cola and len(corriendo) < MAX_GPU:
            exp = cola[0]
            out = kaggle("kernels", "push", "-p", os.path.join(AQUI, exp))
            if "successfully" in out:
                print(f"[{exp}] lanzado: {out.splitlines()[-1]}", flush=True)
                corriendo.append(cola.pop(0))
                time.sleep(5)
            elif "Maximum batch GPU session" in out:
                break
            else:
                print(f"[{exp}] no se pudo lanzar: {out}", flush=True)
                cola.pop(0)
        if cola or corriendo:
            time.sleep(ESPERA)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    if args[0] == "--solo-bajar":
        for exp in args[1:]:
            bajar_y_anotar(exp)
        sys.exit(0)
    cola = pendientes() if args == ["pendientes"] else args
    print("cola:", cola, flush=True)
    correr(cola)
