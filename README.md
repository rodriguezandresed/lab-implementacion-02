# Lab Implementación II: PetFinder Adoption Prediction

Predecir `AdoptionSpeed` (0 a 4) con QWK como métrica.

## Mejor modelo: `experiments/EXP-019/`

Private score **0.40321** (late submission). Es EXP-017 (ver abajo) más 14 columnas simples, sin modelo:

- Texto (`Description`): largo, palabras, fracción de caracteres chinos, fracción de palabras en malayo, fracción de mayúsculas, signos de exclamación, mención de teléfono, mención de precio.
- Foto 1: nitidez (varianza del Laplaciano), brillo, contraste, ancho, alto, aspecto.

EXP-017 es el baseline (`mdm-2026.ipynb`) con dos cambios:

1. Las 1280 columnas `img_feat_*` se reducen a 20 con `TruncatedSVD`, ajustado dentro de cada fold.
2. `LGBMRegressor` en lugar del clasificador, y la predicción se corta en clases 0 a 4 con las proporciones de clase del train del fold (cortes por cuantiles).

Inputs (en `experiments/EXP-019/kernel-metadata.json`):

- `rodriguezandresed/examen-lab-ii-preprocesado-texto`: 7 emociones de `Description` (código en `preprocesado/texto/`).
- `rodriguezandresed/examen-lab-ii-preprocesado-img`: embeddings EfficientNetB0 de la foto 1 (código en `preprocesado/img/`). Esta versión arregla el `PetID`: la original usaba `p.strip("-1.jpg")`, que borraba los `1` al principio y al final del id y dejaba 1840 mascotas de train sin features de imagen.
- `rodriguezandresed/examen-lab-ii-preprocesado-texto-v2` y `rodriguezandresed/examen-lab-ii-preprocesado-img-v2`: las columnas nuevas (código en `preprocesado/texto-v2/` y `preprocesado/img-v2/`).

Para correrlo en Kaggle: `python experiments/correr.py EXP-019` (necesita `~/.kaggle/kaggle.json`). Después, `kaggle competitions submit -c petfinder-adoption-prediction -f submission.csv -k rodriguezandresed/exp-019-stats-texto-img -v <version> -m "..."`.

## Resultados (private score)

| Notebook | Qué es | Private |
|---|---|---|
| **EXP-019** | EXP-017 + estadísticas simples de texto y foto | **0.40321** |
| EXP-021 | EXP-017 + puntajes CLIP de la foto 1 (tierno, jaula, borrosa, persona, varios animales) | 0.39926 |
| EXP-017 | SVD de imagen + cortes por cuantiles | 0.38381 |
| EXP-020 | EXP-017 + preguntas zero-shot sobre `Description` (DeBERTa-v3 NLI) | 0.38075 |
| EXP-018 | EXP-017 entrenado con folds agrupados por `RescuerID` | 0.37608 |
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

EXP-009 y EXP-008 se subieron con las imágenes viejas (con el bug del `PetID`). EXP-010 a EXP-015 no se subieron: su CV no superaba al baseline. EXP-022 (`txt_score`: ridge sobre embeddings `multilingual-e5-base`, dentro del fold) no se subió: empeoraba en los dos CV.

En curso: EXP-023 (EXP-019 + CLIP juntos) y una detección de animales con DETR (cuánto de la foto ocupa el animal, cuántos hay).

Lo que más sumó: los cortes por cuantiles (+0.07), después las columnas simples (+0.02) y CLIP (+0.015). Los modelos de texto (zero-shot, embeddings) no mejoraron el leaderboard.

CV por experimento: `experiments/log.csv`. Ningún CV predice solo el leaderboard: con folds agrupados por `RescuerID` EXP-019 y EXP-021 daban "descartar" (fold 4 cae) y en el leaderboard fueron los dos mejores; `KFold(5)` sí marcó la mejora (0.4501 y 0.4460 contra 0.4320) pero sobreestima el nivel en unos 0.05.

## Estructura

- `mdm-2026.ipynb`: baseline.
- `experiments/EXP-###/`: spec (`EXP-###.md`), notebook, `kernel-metadata.json` y `salida/` (log de Kaggle, `submission.csv`, `resultado.txt`).
- `experiments/correr.py`: corre experimentos en Kaggle (máximo 2 sesiones GPU), baja la salida y anota `log.csv`.
- `preprocesado/`: notebooks que generan los parquet de texto e imagen (`texto/`, `img/`: los de la cátedra; `texto-v2/`, `img-v2/`, `img-deteccion/`: features nuevas).
- `docs/`: specs, checklist y `LEDGER.md` (pendientes).
