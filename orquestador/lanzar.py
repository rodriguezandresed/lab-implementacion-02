"""Lanza kernels de preprocesamiento o modelado en Kaggle (sin bloquear si se pasa --no-wait)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orquestador as orq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--codigo", required=True, help="nombre del archivo dentro de orquestador/kernels/")
    ap.add_argument("--config", default="{}", help="JSON de CONFIG")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--internet", action="store_true")
    ap.add_argument("--sources", default="", help="kernel_sources separados por coma (user/slug o solo slug para usuario propio)")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--limite-min", type=int, default=180)
    a = ap.parse_args()

    kg = orq.Kaggle()
    cfg = json.loads(a.config)
    sources = []
    for s in [x.strip() for x in a.sources.split(",") if x.strip()]:
        sources.append(s if "/" in s else f"{kg.user}/{s}")
    orq.push(kg, a.slug, a.codigo, cfg, gpu=a.gpu, internet=a.internet, kernel_sources=sources)
    if a.wait:
        orq.esperar(kg, [a.slug], limite_min=a.limite_min)


if __name__ == "__main__":
    main()
