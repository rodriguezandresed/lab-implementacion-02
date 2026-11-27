# Orquestador

Corre un experimento de imagen de punta a punta en Kaggle desde la PC: preprocesamiento → modelo con validación → kappa → submit → private.

```
python orquestador/orquestador.py EXP-030 --img clip-l14             # foto 1
python orquestador/orquestador.py EXP-031 --img clip-l14 --agg mean  # promedio de todas las fotos
python orquestador/orquestador.py EXP-032 --img dinov2-b
python orquestador/orquestador.py EXP-030 --img clip-l14 --solo-scores   # solo busca el private
```

Opciones:
- Baseline y variante: `--base effb0_first` (imagen del baseline), `--img`/`--agg` (imagen de la variante), `--emo`/`--emo-base` (`catedra` o `mdeberta`), `--drop`/`--drop-base` (columnas que no entran, ej. `RescuerID`), `--txt-score`/`--txt-score-base` (`e5` o `e5l`: ridge sobre embeddings de texto dentro del fold).
- Corrida: `--sin-submit`, `--forzar-submit` (subir aunque el delta sea > +0.02), `--rehacer` (recalcular los preprocesamientos), `--n-svd 20`, `--solo-scores`.

Ejemplo (EXP-035): `python orquestador/orquestador.py EXP-035 --img dinov2-s --base dinov2-s_first --drop RescuerID --drop-base RescuerID --txt-score e5`

Preprocesamientos de texto: `lab2-prep-emo-mdeberta` (7 emociones zero-shot multilingües) y `lab2-prep-txt-e5` (embeddings `multilingual-e5-base`).

## Qué hace

1. **Conexión**: lee el token de `.mcp.json` (el del MCP de Kaggle, lo busca hacia arriba desde esta carpeta) y abre una sesión MCP. Usuario: el dueño del token (`ucatalano`).
2. **Preprocesamiento**: un kernel por modelo en la cuenta, `lab2-prep-<modelo>`, con `kernels/prep.py` (GPU T4 + internet para bajar los pesos). Si ya terminó bien se reusa.
   - `lab2-prep-stats`: `img_stats_*` y `txt_stats_*` (las columnas de EXP-019).
   - `lab2-prep-effb0`: EfficientNetB0 foto 1, el embedding del baseline EXP-026.
   - `lab2-prep-<img>`: `img_<img>_first_*` (foto 1) e `img_<img>_mean_*` (promedio de todas las fotos).
3. **Modelo**: kernel `lab2-exp-###` con `kernels/modelo.py` (sin internet, como pide la competencia). Es EXP-026 corrido dos veces con folds agrupados por `RescuerID`: baseline con EfficientNetB0 y variante con el embedding nuevo. Después entrena la variante con `KFold(5)` para el submission.
4. **Kappa**: baja `results.json`, `resultado.txt` y `submission.csv` a `experiments/EXP-###/salida/` y anota la fila en `experiments/log.csv` con la decisión del spec.
5. **Regla de AGENTS.md**: si el delta es > +0.02, frena antes del submit.
6. **Submit** del kernel a la competencia por MCP (si falla, por CLI) y espera el private. Lo agrega a `resultado.txt`.

## Modelos de imagen disponibles

| `--img` | Modelo | Dims |
|---|---|---|
| `clip-l14` | openai/clip-vit-large-patch14 | 768 |
| `dinov2-s` | facebook/dinov2-small (ViT-S/14) | 384 |
| `dinov2-b` | facebook/dinov2-base | 768 |
| `dinov2-l` | facebook/dinov2-large | 1024 |
| `siglip-b` | google/siglip-base-patch16-224 | 768 |
| `effb0` | EfficientNetB0 keras (baseline), backend torch en GPU | 1280 |

Todos los preprocesamientos con modelo corren en GPU T4 y fallan si no la encuentran (no hay vuelta a CPU).

Para agregar otro: una entrada en `HF` dentro de `kernels/prep.py`.

## Requisitos

- `.mcp.json` con el servidor `kaggle` y su token en alguna carpeta de arriba.
- `uv` (`pip install uv`): el push usa `kaggle>=1.8` vía `uv run`, que acepta el token del MCP. Con `KAGGLE_CLI` se puede usar otro comando.
- El push es con la CLI porque `save_notebook` del MCP no adjunta las fuentes de datos. Todo lo demás va por el MCP.
- Los inputs del equipo (`rodriguezandresed/...`) son privados para otra cuenta; por eso el orquestador los recalcula en la propia. Las emociones salen del kernel público de la cátedra (`nickel/examen-lab-ii-preprocesado-texto`).
