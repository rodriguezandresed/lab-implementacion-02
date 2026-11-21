# Lab Implementación II: PetFinder Adoption Prediction

Predecir `AdoptionSpeed` (0 a 4) con QWK como métrica.

## Mejor modelo: `experiments/EXP-017/`

Private score **0.38381** (late submission). Es el baseline (`mdm-2026.ipynb`) con dos cambios:

1. Las 1280 columnas `img_feat_*` se reducen a 20 con `TruncatedSVD`, ajustado dentro de cada fold.
2. `LGBMRegressor` en lugar del clasificador, y la predicción se corta en clases 0 a 4 con las proporciones de clase del train del fold (cortes por cuantiles).

Inputs (en `experiments/EXP-017/kernel-metadata.json`):

- `rodriguezandresed/examen-lab-ii-preprocesado-texto`: 7 emociones de `Description` (código en `preprocesado/texto/`).
- `rodriguezandresed/examen-lab-ii-preprocesado-img`: embeddings EfficientNetB0 de la foto 1 (código en `preprocesado/img/`). Esta versión arregla el `PetID`: la original usaba `p.strip("-1.jpg")`, que borraba los `1` al principio y al final del id y dejaba 1840 mascotas de train sin features de imagen. Con esos inputs el mismo notebook da 0.37716.

Para correrlo en Kaggle: `python experiments/correr.py EXP-017` (necesita `~/.kaggle/kaggle.json`). Después, `kaggle competitions submit -c petfinder-adoption-prediction -f submission.csv -k rodriguezandresed/exp-017-svd-cuantiles -v <version> -m "..."`.

## Resultados (private score)

| Notebook | Qué es | Private |
|---|---|---|
| **EXP-017** | SVD de imagen + cortes por cuantiles | **0.38381** |
| EXP-018 | lo mismo, entrenado con folds agrupados por `RescuerID` | 0.37608 |
| EXP-009 | solo cortes por cuantiles | 0.37051 |
| EXP-004 | `rescuer_count` | 0.32833 |
| EXP-001 | flag de nombre genérico | 0.30701 |
| EXP-007 | sentiment por oración | 0.30149 |
| baseline | `mdm-2026.ipynb` | 0.30074 |
| EXP-003 | flag sin nombre | 0.29860 |
| EXP-008 | solo SVD de imagen | 0.29702 |
| EXP-002 | baseline con folds agrupados | 0.28530 |
| EXP-006 | metadata de Google Vision | 0.28397 |
| EXP-005 | regresor + redondeo | 0.22240 |

EXP-009 y EXP-008 se subieron con las imágenes viejas (con el bug del `PetID`). EXP-010 a EXP-015 no se subieron: su CV no superaba al baseline.

CV por experimento (media y desvío de QWK, decisión): `experiments/log.csv`. Para estimar el leaderboard conviene el CV con folds agrupados por `RescuerID` (EXP-018: 0.3771); `KFold(5)` sobreestima unos 0.05.

## Estructura

- `mdm-2026.ipynb`: baseline.
- `experiments/EXP-###/`: spec (`EXP-###.md`), notebook, `kernel-metadata.json` y `salida/` (log de Kaggle, `submission.csv`, `resultado.txt`).
- `experiments/correr.py`: corre experimentos en Kaggle (máximo 2 sesiones GPU), baja la salida y anota `log.csv`.
- `preprocesado/`: notebooks que generan los parquet de texto e imagen.
- `docs/`: specs, checklist y `LEDGER.md` (pendientes).
