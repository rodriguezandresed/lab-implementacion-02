# Lab Implementación II: PetFinder Adoption Prediction

Predecir `AdoptionSpeed` (0 a 4) con QWK como métrica.

## Mejor modelo

- **Mejor private score: EXP-032, 0.44273** (DINOv2-S, promedio de todas las fotos). Es la submission que cuenta para la nota.
- **Base vigente por la regla: EXP-040, private 0.44183** (CLIP-L/14, promedio de todas las fotos, sobre EXP-033). Es el único cambio que pasó el criterio medido con 5 particiones agrupadas (delta +0.0069, mejora en 5 de 5). Los experimentos que siguen se construyen sobre ella.
- EXP-032 y EXP-040 están a 0.001 en el private: una diferencia de ese tamaño es ruido (el private tiene 3.972 mascotas).

## Cola sobre EXP-033 con 5 particiones (EXP-036 a EXP-043)

Con una sola partición agrupada, el promedio de 5 folds se mueve unos ±0.004 según cómo caen los rescatistas, así que deltas de ±0.003 no se podían decidir. Desde EXP-037 cada cambio se mide con `StratifiedGroupKFold(5)` repetido con 5 seeds (42, 1, 2, 3, 4): se acepta si el delta medio de las 25 folds es ≥ +0.004 y mejora en ≥ 4 de 5 seeds. Lo corre `experiments/cola.py` (kernels armados desde `orquestador/kernels/modelo.py`, inputs recalculados en `rodriguezandresed`).

| Exp | Cambio (sobre EXP-033) | Delta 5 particiones | Seeds que mejoran | Private | Decisión |
|---|---|---|---|---|---|
| EXP-037 | desempate de 031, 032, 033, 035 | 032 vs 031 +0.0028, 035 vs 033 +0.0037 | 4/5, 5/5 | (sin submit) | empates |
| EXP-036 | emociones sobre la traducción al inglés (malayo, chino) | +0.0002 | 2/5 | 0.43213 | empate |
| EXP-038 | foto promedio + `txt_score` e5 | +0.0037 | 5/5 | 0.44165 | empate |
| EXP-039 | DINOv2-B foto 1 | +0.0026 | 5/5 | 0.43231 | empate |
| **EXP-040** | **CLIP-L/14 promedio de fotos** | **+0.0069** | **5/5** | **0.44183** | **aceptado** |
| EXP-041 | promedio de 5 seeds de LightGBM | +0.0023 | 5/5 | 0.43105 | empate |
| EXP-042 | hiperparámetros re-tuneados (grid de 36) | +0.0034 | 5/5 | 0.43159 | empate |
| EXP-043 | selección genética (saca 9 de 41 columnas) | +0.0026 | 4/5 | 0.42818 | empate |

Para correr la cola: `python experiments/cola.py [EXP-###...]` (necesita `~/.kaggle/kaggle.json`).

## Cómo se llegó a EXP-026

EXP-026: private score **0.42045** (late submission). Es EXP-025 más una columna, `rescuer_count`: cuántas publicaciones tiene el mismo `RescuerID`, contado dentro de cada dataset (train en train, test en test), sin usar el target. Distingue refugios de particulares. Con folds agrupados por `RescuerID` sube de 0.4168 a 0.4256 y mejora los 5 folds (aceptado por el criterio); en el leaderboard, +0.0165.

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

## Imagen y texto con el orquestador (EXP-030 a EXP-035)

`orquestador/orquestador.py` corre todo en Kaggle desde la PC: preprocesa (un kernel por modelo, GPU T4), corre el modelo con folds agrupados contra el baseline en la misma sesión, baja el kappa, anota `experiments/log.csv`, hace el submit y espera el private. Ver `orquestador/README.md`. Los inputs se recalculan en la cuenta propia (los kernels de `rodriguezandresed` son privados); el baseline EXP-026 re-medido da 0.4245 agrupado (original 0.4256).

| Exp | Cambio | Agrupado (delta) | Private | Decisión |
|---|---|---|---|---|
| EXP-030 | CLIP ViT-L/14 foto 1 en lugar de EfficientNetB0 | 0.4350 (+0.0105, 4/5) | 0.42405 | aceptado |
| EXP-031 | DINOv2 ViT-S/14 foto 1 en lugar de EfficientNetB0 | 0.4367 (+0.0122, 5/5) | 0.43608 | aceptado |
| EXP-032 | EXP-031 con el promedio de todas las fotos | 0.4391 (+0.0024, 2/5) | **0.44273** | descartado por CV |
| EXP-033 | EXP-031 sin `RescuerID` como feature | 0.4392 (+0.0025, 3/5) | 0.42944 | aceptado (higiene; KFold 0.4804 → 0.4623) |
| EXP-034 | EXP-033 con emociones multilingües (mDeBERTa zero-shot) | 0.4364 (−0.0028, 2/5) | 0.42962 | descartado |
| EXP-035 | EXP-033 + `txt_score` (ridge sobre e5 dentro del fold) | 0.4420 (+0.0028, 3/5) | 0.43393 | descartado por CV |

Base por la regla después de esta tanda: EXP-033. EXP-032 y EXP-035 se desempataron en EXP-037 (abajo).

## Resultados (private score)

| Notebook | Qué es | Private |
|---|---|---|
| **EXP-032** | DINOv2-S, promedio de todas las fotos (orquestador) | **0.44273** |
| EXP-040 | CLIP-L/14 promedio de fotos sobre EXP-033 (base vigente) | 0.44183 |
| EXP-038 | EXP-033 + foto promedio + `txt_score` | 0.44165 |
| EXP-039 | DINOv2-B foto 1 sobre EXP-033 | 0.43231 |
| EXP-036 | EXP-033 + emociones traducidas | 0.43213 |
| EXP-042 | EXP-033 con hiperparámetros re-tuneados | 0.43159 |
| EXP-041 | EXP-033 con 5 seeds de LightGBM | 0.43105 |
| EXP-043 | EXP-033 con selección genética | 0.42818 |
| EXP-031 | DINOv2-S foto 1 en lugar de EfficientNetB0 | 0.43608 |
| EXP-035 | EXP-033 + `txt_score` e5 | 0.43393 |
| EXP-034 | EXP-033 + emociones mDeBERTa | 0.42962 |
| EXP-033 | EXP-031 sin `RescuerID` | 0.42944 |
| EXP-030 | CLIP ViT-L/14 foto 1 | 0.42405 |
| EXP-026 | EXP-025 + `rescuer_count` | 0.42045 |
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
