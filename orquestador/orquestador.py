"""Orquestador: preprocesa en Kaggle, corre el modelo, captura el kappa, hace el submit y anota log.csv.

Uso (desde cualquier carpeta):
    python orquestador/orquestador.py EXP-030 --img clip-l14
    python orquestador/orquestador.py EXP-031 --img dinov2-b --agg mean
    python orquestador/orquestador.py EXP-030 --img clip-l14 --solo-scores     # solo vuelve a leer el private

Pasos:
  1. Conexion: lee el token del MCP de Kaggle desde .mcp.json (lo busca hacia arriba desde esta carpeta)
     y abre una sesion MCP contra https://www.kaggle.com/mcp.
  2. Preprocesamiento: un kernel por modelo, ucatalano/lab2-prep-<modelo> (stats, effb0 y el --img pedido).
     Si ya existe y su ultima corrida termino bien, se reusa (--rehacer para forzar). Cada uno deja parquets.
  3. Modelo: kernel ucatalano/lab2-<exp> con kernels/modelo.py (EXP-026 con la imagen nueva), sin internet,
     que lee esos parquets. Se baja results.json, resultado.txt y submission.csv a experiments/<EXP>/salida/.
  4. Regla de AGENTS.md: si el delta de CV es > +0.02 se frena antes del submit (--forzar-submit para seguir).
  5. Submit del kernel a la competencia (MCP) y espera del private score.
  6. Fila en experiments/log.csv y resumen en experiments/<EXP>/salida/resultado.txt.

El MCP se usa para estado, salidas, submit y scores. El push del kernel se hace con la CLI de Kaggle porque
save_notebook del MCP no adjunta las fuentes de datos (competencia y kernels de preprocesamiento).
"""
import argparse, csv, datetime, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)
COMP = "petfinder-adoption-prediction"
TEXTO = "nickel/examen-lab-ii-preprocesado-texto"  # 7 emociones de la descripcion (publico, de la catedra)
ESPERA = 30


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ 1. conexion (MCP)
def buscar_mcp_json():
    d = AQUI
    while True:
        p = os.path.join(d, ".mcp.json")
        if os.path.exists(p):
            return p
        if os.path.dirname(d) == d:
            raise SystemExit("No encontre .mcp.json con el servidor 'kaggle' en ninguna carpeta de arriba.")
        d = os.path.dirname(d)


class Kaggle:
    def __init__(self):
        cfg = json.load(open(buscar_mcp_json(), encoding="utf-8"))["mcpServers"]["kaggle"]
        self.url = cfg["url"]
        self.auth = cfg["headers"]["Authorization"]
        self.token = self.auth.split(" ", 1)[1]
        self.sid = None
        self.sid, _ = self._post({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                                  "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                             "clientInfo": {"name": "orquestador-lab2", "version": "1"}}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.user = self._usuario()
        log("MCP conectado como", self.user)

    def _post(self, payload):
        h = {"Authorization": self.auth, "Content-Type": "application/json",
             "Accept": "application/json, text/event-stream", "User-Agent": "orquestador-lab2/1.0"}
        if self.sid:
            h["Mcp-Session-Id"] = self.sid
            h["MCP-Protocol-Version"] = "2025-06-18"
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), method="POST", headers=h)
        with urllib.request.urlopen(req, timeout=600) as r:
            raw, sid = r.read().decode("utf-8"), r.headers.get("Mcp-Session-Id")
        out = None
        for line in raw.splitlines():
            if line.startswith("data:"):
                out = json.loads(line[5:])
        if out is None and raw.strip():
            out = json.loads(raw)
        return sid, out

    def tool(self, name, **request):
        _, r = self._post({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": name, "arguments": {"request": request}}})
        if "error" in r:
            raise RuntimeError(f"{name}: {r['error']}")
        txt = "".join(c.get("text", "") for c in r["result"]["content"])
        if r["result"].get("isError"):
            raise RuntimeError(f"{name}: {txt[:500]}")
        try:
            return json.loads(txt)
        except ValueError:
            return txt

    def _usuario(self):
        try:
            p = self.tool("get_user_profile")
            return p.get("userName") or p.get("user_name") or p.get("displayName") or "?"
        except Exception:
            return os.environ.get("KAGGLE_USER", "ucatalano")

    def estado(self, slug):
        try:
            return self.tool("get_notebook_session_status", userName=self.user, kernelSlug=slug).get("status", "?")
        except Exception as e:
            # un kernel que todavia no existe devuelve "Permission 'kernels.get' was denied"
            return "NO_EXISTE" if re.search(r"not found|404|denied", str(e), re.I) else f"ERROR_MCP {e}"

    def version(self, slug):
        return int(self.tool("get_notebook_info", userName=self.user, kernelSlug=slug)["metadata"]["current_version_number"])

    def bajar(self, slug, archivo, destino):
        r = self.tool("download_notebook_output", ownerSlug=self.user, kernelSlug=slug, filePath=archivo, hasFilePath=True)
        with urllib.request.urlopen(r["url"], timeout=600) as resp:
            open(destino, "wb").write(resp.read())

    def submit(self, slug, version, descripcion):
        return self.tool("create_code_competition_submission", competitionName=COMP, kernelOwner=self.user,
                         kernelSlug=slug, kernelVersion=version, hasKernelVersion=True,
                         fileName="submission.csv", hasFileName=True,
                         submissionDescription=descripcion, hasSubmissionDescription=True)

    def envios(self, n=20):
        return self.tool("search_competition_submissions", competitionName=COMP, sortBy="Date", group="All",
                         pageSize=n, hasPageSize=True).get("submissions", [])


# ------------------------------------------------------------------ CLI de Kaggle (solo para push)
def comando_cli():
    if os.environ.get("KAGGLE_CLI"):
        return os.environ["KAGGLE_CLI"].split()
    uv = shutil.which("uv") or os.path.join(tempfile.gettempdir(), "kvenv", "Scripts", "uv.exe")
    if os.path.exists(uv):  # la CLI >= 1.8 acepta el token KGAT del MCP; uv la baja sola
        return [uv, "run", "-q", "--no-project", "--python", "3.12", "--with", "kaggle>=1.8", "--with", "six", "kaggle"]
    if shutil.which("kaggle"):
        return ["kaggle"]
    raise SystemExit("Falta la CLI de Kaggle: instalar uv (pip install uv) o 'pip install \"kaggle>=1.8\"'.")


def cli(kg, *args):
    env = dict(os.environ, KAGGLE_API_TOKEN=kg.token, PYTHONIOENCODING="utf-8")
    tmp = tempfile.gettempdir()
    for var, carpeta in (("UV_CACHE_DIR", "uvcache"), ("UV_PYTHON_INSTALL_DIR", "uvpy")):
        if var not in env and os.path.isdir(os.path.join(tmp, carpeta)):  # donde ya quedo instalado Python 3.12
            env[var] = os.path.join(tmp, carpeta)
    p = subprocess.run(comando_cli() + list(args), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} fallo:\n{out}")
    return out


def push(kg, slug, codigo, config, gpu, internet, kernel_sources):
    """Sube kernels/<codigo> como script con CONFIG inyectado arriba. Devuelve el numero de version."""
    d = tempfile.mkdtemp(prefix=f"push-{slug}-")
    src = open(os.path.join(AQUI, "kernels", codigo), encoding="utf-8").read()
    open(os.path.join(d, codigo), "w", encoding="utf-8").write(f"CONFIG = {config!r}\n\n" + src)
    json.dump({"id": f"{kg.user}/{slug}", "title": slug, "code_file": codigo, "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": internet,
               "dataset_sources": [], "competition_sources": [COMP], "kernel_sources": kernel_sources},
              open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    args = ["kernels", "push", "-p", d] + (["--accelerator", "NvidiaTeslaT4"] if gpu else [])
    out = cli(kg, *args)
    log(slug, "->", out.strip().splitlines()[-1] if out.strip() else out)
    m = re.search(r"[Vv]ersion\s*(?:number)?\s*:?\s*(\d+)", out)
    return int(m.group(1)) if m else None


def esperar(kg, slugs, limite_min=240):
    pendientes, t0 = set(slugs), time.time()
    time.sleep(ESPERA)
    while pendientes:
        for s in sorted(pendientes):
            st = kg.estado(s)
            if st.upper() in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED", "FAILED"):
                log(s, st)
                if st.upper() != "COMPLETE":
                    raise SystemExit(f"{s} termino en {st}: ver el log en https://www.kaggle.com/code/{kg.user}/{s}")
                pendientes.discard(s)
        if pendientes:
            if time.time() - t0 > limite_min * 60:
                raise SystemExit(f"Se paso el limite de {limite_min} min esperando {pendientes}")
            log("corriendo:", ", ".join(f"{s}={kg.estado(s)}" for s in sorted(pendientes)))
            time.sleep(ESPERA)


# ------------------------------------------------------------------ 2. preprocesamiento
def preprocesar(kg, modelos, rehacer, solo_primera):
    lanzados = []
    for m in modelos:
        slug = f"lab2-prep-{m}"
        st = kg.estado(slug)
        if st == "COMPLETE" and not rehacer:
            log(slug, "ya esta COMPLETE: se reusa")
            continue
        if st in ("RUNNING", "QUEUED"):
            log(slug, "ya esta corriendo: espero")
        else:
            gpu = m != "stats"
            push(kg, slug, "prep.py", {"modelo": m, "solo_primera": solo_primera}, gpu=gpu, internet=gpu, kernel_sources=[])
        lanzados.append(slug)
    if lanzados:
        esperar(kg, lanzados)
    return [f"{kg.user}/lab2-prep-{m}" for m in modelos]


# ------------------------------------------------------------------ 3-6. modelo, regla, submit, log
def decision(r):
    if r["delta_mean"] > 0.02:
        return f"sospechar y frenar (delta {r['delta_mean']:+.4f} > 0.02)"
    if r["delta_mean"] >= 0.004 and r["folds_mejoran"] >= 4:
        return f"aceptar (delta {r['delta_mean']:+.4f}, mejora en {r['folds_mejoran']}/5)"
    return f"descartar (delta {r['delta_mean']:+.4f}, mejora en {r['folds_mejoran']}/5)"


def anotar_log(exp, descripcion, r, dec):
    p = os.path.join(REPO, "experiments", "log.csv")
    filas = list(csv.reader(open(p, encoding="utf-8"))) if os.path.exists(p) else [
        ["exp_id", "fecha", "descripcion", "qwk_cv_mean", "qwk_cv_std", "decision"]]
    filas = [f for f in filas if not f or f[0] != exp]
    filas.append([exp, datetime.date.today().isoformat(), descripcion, f"{r['qwk_mean_exp']:.4f}", f"{r['qwk_std_exp']:.4f}", dec])
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(filas)


def esperar_private(kg, descripcion, limite_min=60):
    t0 = time.time()
    while time.time() - t0 < limite_min * 60:
        for s in kg.envios():
            if s.get("description") == descripcion:
                st = str(s.get("status", "")).upper()
                if st == "COMPLETE":
                    return s.get("private_score"), s
                if st == "ERROR":
                    raise SystemExit(f"El submit fallo en Kaggle: {s}")
                log("submit", st)
                break
        time.sleep(ESPERA)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", help="EXP-###")
    ap.add_argument("--img", default="clip-l14", help="clip-l14 | dinov2-b | dinov2-l | siglip-b")
    ap.add_argument("--agg", default="first", choices=["first", "mean"], help="foto 1 o promedio de todas")
    ap.add_argument("--base", default="effb0_first", help="imagen del baseline (EXP-026)")
    ap.add_argument("--emo", default="catedra", help="emociones de la variante: catedra | mdeberta (lab2-prep-emo-mdeberta)")
    ap.add_argument("--emo-base", default="catedra", help="emociones del baseline")
    ap.add_argument("--drop", default="", help="columnas que la variante no usa, separadas por coma (ej. RescuerID)")
    ap.add_argument("--drop-base", default="", help="columnas que el baseline no usa")
    ap.add_argument("--txt-score", default="", help="txt_score de la variante desde embeddings: e5 | e5l (lab2-prep-txt-*)")
    ap.add_argument("--txt-score-base", default="", help="txt_score del baseline")
    ap.add_argument("--n-svd", type=int, default=20)
    ap.add_argument("--rehacer", action="store_true", help="recalcular los preprocesamientos aunque existan")
    ap.add_argument("--sin-submit", action="store_true")
    ap.add_argument("--forzar-submit", action="store_true", help="subir aunque el delta sea > 0.02")
    ap.add_argument("--solo-scores", action="store_true", help="no corre nada: busca el private del submit de este EXP")
    a = ap.parse_args()

    kg = Kaggle()
    exp, slug = a.exp, f"lab2-{a.exp.lower()}"
    salida = os.path.join(REPO, "experiments", exp, "salida")
    os.makedirs(salida, exist_ok=True)
    lista = lambda t: [c.strip() for c in t.split(",") if c.strip()]
    base = {"img": a.base, "emo": a.emo_base, "drop": lista(a.drop_base), "txt_score": a.txt_score_base}
    variante = {"img": f"{a.img}_{a.agg}", "emo": a.emo, "drop": lista(a.drop), "txt_score": a.txt_score}
    corto = lambda d: (f"img {d['img']}, emo {d['emo']}" + (f", sin {'+'.join(d['drop'])}" if d["drop"] else "")
                       + (f", txt_score {d['txt_score']}" if d["txt_score"] else ""))
    descripcion = f"{exp} {corto(variante)} (base: {corto(base)})"

    if not a.solo_scores:
        modelos = ["stats", a.base.rsplit("_", 1)[0], a.img] + [f"emo-{e}" for e in (a.emo_base, a.emo) if e != "catedra"]                   + [f"txt-{t}" for t in (a.txt_score_base, a.txt_score) if t]
        fuentes = preprocesar(kg, list(dict.fromkeys(modelos)), a.rehacer, solo_primera=False)
        config = {"exp": exp, "n_svd": a.n_svd, "base": base, "variante": variante}
        push(kg, slug, "modelo.py", config, gpu=False, internet=False, kernel_sources=[TEXTO] + fuentes)
        esperar(kg, [slug], limite_min=90)
        for f in ("results.json", "resultado.txt", "submission.csv"):
            kg.bajar(slug, f, os.path.join(salida, f))
        r = json.load(open(os.path.join(salida, "results.json")))
        print(open(os.path.join(salida, "resultado.txt"), encoding="utf-8").read())
        dec = decision(r)
        anotar_log(exp, descripcion, r, dec)
        log("log.csv:", dec)
        if a.sin_submit:
            return
        if r["delta_mean"] > 0.02 and not a.forzar_submit:
            raise SystemExit("Delta > +0.02: AGENTS.md pide frenar y revisar antes de seguir. "
                             "Si se reviso y es real, volver a correr con --forzar-submit.")
        version = kg.version(slug)
        log("submit", slug, "version", version)
        try:
            kg.submit(slug, version, descripcion)
        except Exception as e:  # respaldo: la CLI
            log("submit por MCP fallo, uso la CLI:", e)
            cli(kg, "competitions", "submit", COMP, "-k", f"{kg.user}/{slug}", "-v", str(version),
                "-f", "submission.csv", "-m", descripcion)

    priv, s = esperar_private(kg, descripcion)
    linea = f"Kaggle: private {priv} (EXP-026: 0.42045) | {s.get('url', '') if s else 'sin submit encontrado'}"
    log(linea)
    with open(os.path.join(salida, "resultado.txt"), "a", encoding="utf-8") as f:
        f.write(linea + "\n")


if __name__ == "__main__":
    main()
