# Lab Implementación II: PetFinder Adoption Prediction

Predecir `AdoptionSpeed` (0 a 4) con QWK como métrica.

## Mejor modelo: `experiments/EXP-026/`

Private score **0.42045** (late submission). Es EXP-025 más una columna, `rescuer_count`: cuántas publicaciones tiene el mismo `RescuerID`, contado dentro de cada dataset (train en train, test en test), sin usar el target. Distingue refugios de particulares. Con folds agrupados por `RescuerID` sube de 0.4168 a 0.4256 y mejora los 5 folds (aceptado por el criterio); en el leaderboard, +0.0165.

EXP-025 es EXP-019 con los hiperparámetros del grid search (`petfinder-v2-gridsearch`): `LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, colsample_bytree=0.6)`, 300 árboles fijos en lugar de `early_stopping(5)`, que cortaba demasiado pronto y hacía caer el fold 4.

EXP-019 es EXP-017 más 14 columnas simples, sin modelo:

- Texto (`Description`): largo, palabras, fracción de caracteres chinos, fracción de palabras en malayo, fracción de mayúsculas, signos de exclamación, mención de teléfono, mención de precio.
- Foto 1: nitidez (varianza del Laplaciano), brillo, contraste, ancho, alto, aspecto.

EXP-017 es el baseline (`mdm-2026.ipynb`) con dos cambios:

1. Las 1280 columnas `img_feat_*` se reducen a 20 con `TruncatedSVD`, ajustado dentro de cada fold.
2. `LGBMRegressor` en lugar del clasificador, y la predicción se corta en clases 0 a 4 con las proporciones de clase del train del fold (cortes por cuantiles).

Inputs (en `experiments/EXP-026/kernel-metadata.json`, los mismos que EXP-019):

- `rodriguezandresed/examen-lab-ii-preprocesado-texto`: 7 emociones de `Description` (código en `preprocesado/texto/`).
- `rodriguezandresed/examen-lab-ii-preprocesado-img`: embeddings EfficientNetB0 de la foto 1 (código en `preprocesado/img/`). Esta versión arregla el `PetID`: la original usaba `p.strip("-1.jpg")`, que borraba los `1` al principio y al final del id y dejaba 1840 mascotas de train sin features de imagen.
- `rodriguezandresed/examen-lab-ii-preprocesado-texto-v2` y `rodriguezandresed/examen-lab-ii-preprocesado-img-v2`: las columnas nuevas (código en `preprocesado/texto-v2/` y `preprocesado/img-v2/`).

Para correrlo en Kaggle: `python experiments/correr.py EXP-026` (necesita `~/.kaggle/kaggle.json`). Después, `kaggle competitions submit -c petfinder-adoption-prediction -f submission.csv -k rodriguezandresed/exp-026-rescuer-count-sobre-025 -v <version> -m "..."`.

## Resultados (private score)

| Notebook | Qué es | Private |
|---|---|---|
| **EXP-026** | EXP-025 + `rescuer_count` | **0.42045** |
| EXP-027 | EXP-025 + CLIP | 0.40572 |
| EXP-025 | EXP-019 con los hiperparámetros del grid search, sin early stopping | 0.40393 |
| EXP-019 | EXP-017 + estadísticas simples de texto y foto | 0.40321 |
| EXP-029 | EXP-025 + preguntas zero-shot | 0.40159 |
| EXP-023 | EXP-019 + CLIP juntos | 0.40141 |
| EXP-024 | EXP-019 + detección de animales (RT-DETR: cuánto de la foto ocupa, cuántos hay) | 0.39997 |
| EXP-028 | EXP-025 + detección de animales | 0.39872 |
| EXP-021 | EXP-017 + puntajes CLIP de la foto 1 (tierno, jaula, borrosa, persona, varios animales) | 0.39926 |
| EXP-017 | SVD de imagen + cortes por cuantiles | 0.38381 |
| EXP-020 | EXP-017 + preguntas zero-shot sobre `Description` (DeBERTa-v3 NLI) | 0.38075 |
| v5b v2 | línea vieja: selección de columnas con algoritmo genético | 0.37171 |
| EXP-018 | EXP-017 entrenado con folds agrupados por `RescuerID` | 0.37608 |
| EXP-009 | solo cortes por cuantiles | 0.37051 |
| v3 | línea vieja: regresión + umbrales | 0.36805 |
| v5b v1 | línea vieja: construcción de columnas con algoritmo genético | 0.36207 |
| EXP-004 | `rescuer_count` | 0.32833 |
| EXP-001 | flag de nombre genérico | 0.30701 |
| EXP-007 | sentiment por oración | 0.30149 |
| baseline | `mdm-2026.ipynb` | 0.30074 |
| EXP-003 | flag sin nombre | 0.29860 |
| EXP-008 | solo SVD de imagen | 0.29702 |
| EXP-002 | baseline con folds agrupados | 0.28530 |
| EXP-006 | metadata de Google Vision | 0.28397 |
| EXP-005 | regresor + redondeo | 0.22240 |

EXP-009 y EXP-008 se subieron con las imágenes viejas (con el bug del `PetID`). EXP-010 a EXP-015 no se subieron: su CV no superaba al baseline. EXP-022 (`txt_score`: ridge sobre embeddings `multilingual-e5-base`, dentro del fold) no se subió: empeoraba en los dos CV.

EXP-023 (EXP-019 + CLIP) tuvo el mejor CV agrupado (0.3897) pero no superó a EXP-019 en el leaderboard (0.40141 contra 0.40321): las dos fuentes no se suman. Lo mismo con la detección de animales (EXP-024): sube los dos CV pero no el leaderboard. Re-probados sobre EXP-025 (sin el ruido del early stopping, EXP-027 a 029): CLIP, detección y zero-shot se descartan por CV agrupado; CLIP sube +0.002 en el leaderboard, dentro del ruido.

Lo que más sumó: los cortes por cuantiles (+0.07), después las columnas simples (+0.02) y `rescuer_count` (+0.017); CLIP (+0.015) y la detección suman solas pero no encima de las columnas simples. Los modelos de texto (zero-shot, embeddings) no mejoraron el leaderboard.

CV por experimento: `experiments/log.csv`. Ningún CV predice solo el leaderboard: con folds agrupados por `RescuerID` EXP-019 y EXP-021 daban "descartar" (fold 4 cae) y en el leaderboard fueron los dos mejores; `KFold(5)` sí marcó la mejora (0.4501 y 0.4460 contra 0.4320) pero sobreestima el nivel en unos 0.05. Buena parte del ruido del CV agrupado venía del `early_stopping(5)`: sin él (EXP-025) los 5 folds quedan entre 0.385 y 0.439.

Línea vieja (`petfinder-v3/v4/v5b`, antes de los EXP): el algoritmo genético sirvió para elegir columnas (v5b v2, +0.004 sobre v3) y no para construir fórmulas (v5b v1, -0.006). Registro: `docs/REGISTRO_GENETICO.md`.

## Estructura

- `mdm-2026.ipynb`: baseline.
- `experiments/EXP-###/`: spec (`EXP-###.md`), notebook, `kernel-metadata.json` y `salida/` (log de Kaggle, `submission.csv`, `resultado.txt`).
- `experiments/correr.py`: corre experimentos en Kaggle (máximo 2 sesiones GPU), baja la salida y anota `log.csv`.
- `preprocesado/`: notebooks que generan los parquet de texto e imagen (`texto/`, `img/`: los de la cátedra; `texto-v2/`, `img-v2/`, `img-deteccion/`: features nuevas).
- `docs/`: specs, checklist y `LEDGER.md` (pendientes).
