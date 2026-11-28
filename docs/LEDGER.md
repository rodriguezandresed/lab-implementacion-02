# LEDGER: pendientes

Lo que está abierto. Cuando algo se resuelve, se tacha o se borra y el resultado va a su lugar (regla en `AGENTS.md`, en la raíz del lab; chequeo en `checklist-verificacion.md`, experimento en `experiments/log.csv`).

## Hipótesis a probar
Cada una se corre como un experimento con `spec-experimento.md`.

- [x] Categóricas: ya está en el baseline (`mdm-2026.ipynb` convierte esas columnas a `category`). Sin experimento.
- [x] Rescatista como `category`: ya está en el baseline (`RescuerID` está en la lista de `category`). Sin experimento.

- [x] Folds agrupados: `experiments/EXP-002/EXP-002.md`. QWK con `StratifiedGroupKFold` por `RescuerID` = 0.3083 ± 0.029, contra 0.3834 ± 0.0098 con `KFold(5)`.
- [ ] Confirmar con folds agrupados los deltas que se acepten con `KFold(5)` (lo pide EXP-002 si la caída es grande: fue de 0.075).

### Banco de hipótesis de la cátedra (Clase 2, `exp_bank/`)
Cada fila es un spec distinto.

- [x] Imágenes · metadata: `experiments/EXP-006/EXP-006.md`. Descartado: delta -0.0153, mejora en 0 de 5 folds.
- [x] Sentiment: `experiments/EXP-007/EXP-007.md`. Descartado: delta -0.0027, mejora en 2 de 5 folds.
- [x] Rescatista (count): `experiments/EXP-004/EXP-004.md`. Descartado: delta +0.0002, mejora en 3 de 5 folds, desvío 0.0223.
- [x] Thresholds, paso 1: regresor + redondeo `experiments/EXP-005/EXP-005.md`. Descartado: 0.3105 contra 0.3834 del clasificador.
- [ ] Thresholds, pasos 2 y 3: cuantiles `experiments/EXP-009/EXP-009.md` = 0.4248 ± 0.0261; optimizados `experiments/EXP-010/EXP-010.md` = 0.3999 ± 0.0842 (fold_1 en 0.2507). Delta contra regresor + redondeo > 0.02: frenado por la regla. Contra el clasificador, EXP-009 da +0.041. Falta decidir si se acepta.
- [x] Imágenes · CNN: `experiments/EXP-008/EXP-008.md`. Aceptado: delta +0.0041, mejora en 4 de 5 folds, desvío 0.0061. Verificado: `img_feat_*` ya son embeddings de EfficientNetB0; el experimento los reduce con SVD dentro del fold.
- [x] Nombre genérico: `experiments/EXP-001/EXP-001.md`. Descartado: delta -0.0028, mejora en 0 de 5 folds.
- [x] Sin nombre: `experiments/EXP-003/EXP-003.md`. Descartado: delta -0.0002, mejora en 1 de 5 folds.

### Robustez y combinación
- [x] Ruido de la partición: `experiments/EXP-016/EXP-016.md`. Con 5 particiones `KFold(5, shuffle=True)`: SVD delta +0.0146 (mín +0.0061, 5 de 5 seeds, 24 de 25 folds); cuantiles delta +0.0469 (mín +0.0257, 5 de 5 seeds, 24 de 25 folds).
- [ ] SVD de imagen + cortes por cuantiles: `experiments/EXP-017/EXP-017.md`. 0.4341 ± 0.0139, delta +0.0508, 5 de 5 folds. Frenado por la regla de > 0.02: falta decidir si se acepta. Late submission (2026-11-20): private score 0.37716; el baseline (salida de EXP-016, `test_preds` sin tocar) da 0.30074. Con las imágenes arregladas (v2, 2026-11-20): CV 0.4320, folds agrupados 0.3771, private score 0.38381.
- [ ] Lo mismo con folds agrupados por `RescuerID`: `experiments/EXP-018/EXP-018.md`. 0.3662 ± 0.0151 contra 0.3083 ± 0.029 (repite EXP-002), delta +0.0579, 5 de 5 folds. Frenado por la regla de > 0.02.

### Features con modelos modernos (2026-11-21)
Todas sobre EXP-017, decididas con folds agrupados por `RescuerID` (baseline 0.3771). Inputs: `preprocesado/texto-v2/` y `preprocesado/img-v2/`.
- [x] Estadísticas simples de texto y foto: `experiments/EXP-019/EXP-019.md`. Agrupado 0.3744 (delta -0.0028, 4 de 5, fold 4 cae 0.038); KFold 0.4501. Private **0.40321** (mejor).
- [x] Preguntas zero-shot (DeBERTa-v3 NLI): `experiments/EXP-020/EXP-020.md`. Agrupado 0.3809 (delta +0.0038, 4 de 5, sube el desvío); KFold 0.4338. Private 0.38075.
- [x] Puntajes CLIP de la foto 1: `experiments/EXP-021/EXP-021.md`. Agrupado 0.3779 (delta +0.0007, 2 de 5); KFold 0.4460. Private 0.39926.
- [x] `txt_score` desde embeddings `multilingual-e5-base`: `experiments/EXP-022/EXP-022.md`. Agrupado 0.3750 (delta -0.0021, 3 de 5); KFold 0.4286. No subido.
- [x] EXP-019 + EXP-021 juntos: `experiments/EXP-023/EXP-023.md`. Agrupado 0.3897 (delta +0.0125, 4 de 5, sube el desvío); KFold 0.4400. Private 0.40141: no supera a EXP-019.
- [x] Detección de animales (RT-DETR, `preprocesado/img-deteccion/`) sobre EXP-019: `experiments/EXP-024/EXP-024.md`. Agrupado 0.3827 (delta +0.0083 contra EXP-019, 2 de 5, sube el desvío); KFold 0.4536. Private 0.39997: no supera a EXP-019.
- [ ] Hiperparámetros del grid search sin early stopping sobre EXP-019: `experiments/EXP-025/EXP-025.md`. Agrupado 0.4168 (delta +0.0425, 5 de 5, baja el desvío); KFold 0.4613. Private **0.40393** (mejor). Frenado por la regla de > 0.02: el salto es del CV (el `early_stopping(5)` del baseline cortaba pronto); en el leaderboard es +0.0007. Falta decidir si se acepta.
- [x] El CV agrupado dijo "descartar" a 019 y 021 y el leaderboard los puso primeros; caía el fold 4. Causa probable: `early_stopping(5)`. Con 300 árboles fijos (EXP-025) el fold 4 pasa de 0.328 a 0.385.

### Re-pruebas sobre EXP-025 (2026-11-23)
Baseline EXP-025, folds agrupados por `RescuerID` (0.4168), KFold 0.4613, private 0.40393.
- [x] `rescuer_count` (definición de EXP-004): `experiments/EXP-026/EXP-026.md`. Agrupado 0.4256 (delta +0.0087, 5 de 5): **aceptado**. KFold 0.4693. Private **0.42045** (mejor).
- [x] CLIP: `experiments/EXP-027/EXP-027.md`. Agrupado 0.4166 (delta -0.0003, 2 de 5): descartado. KFold 0.4651. Private 0.40572.
- [x] Detección: `experiments/EXP-028/EXP-028.md`. Agrupado 0.4139 (delta -0.0029, 0 de 5): descartado. KFold 0.4598. Private 0.39872.
- [x] Zero-shot: `experiments/EXP-029/EXP-029.md`. Agrupado 0.4177 (delta +0.0009, 3 de 5): descartado. KFold 0.4626. Private 0.40159.
### Embeddings de imagen más fuertes (2026-11-26, `orquestador/`)
Baseline EXP-026 re-medido con los inputs recalculados en `ucatalano`: agrupado 0.4245 (registro original 0.4256).
- [x] CLIP ViT-L/14 foto 1 en lugar de EfficientNetB0: `experiments/EXP-030/EXP-030.md`. Agrupado 0.4350 (delta +0.0105, 4 de 5; fold 3 -0.0013; desvío 0.0234 → 0.0250). KFold 0.4779. Private **0.42405** (mejor, +0.0036 sobre EXP-026): **aceptado**.
- [x] DINOv2 ViT-S/14 foto 1 en lugar de EfficientNetB0: `experiments/EXP-031/EXP-031.md`. Agrupado 0.4367 (delta +0.0122, 5 de 5; desvío 0.0234 → 0.0199). KFold 0.4804. Private **0.43608** (mejor, +0.0156 sobre EXP-026, +0.0120 sobre EXP-030): **aceptado**. Preprocesado en GPU T4, 4,5 min.
- [x] DINOv2-S promedio de todas las fotos sobre EXP-031: `experiments/EXP-032/EXP-032.md`. Agrupado 0.4391 contra 0.4367 (delta +0.0024, 2 de 5; casi todo el delta es el fold 4, +0.0155; desvío 0.0199 → 0.0147). Por la regla: **descartado**. Private **0.44273** (+0.0067 sobre EXP-031, el mejor). Otra vez el CV agrupado y el leaderboard no coinciden: falta medir con varias seeds de folds para decidir.
- [x] Sin `RescuerID` como feature, sobre EXP-031: `experiments/EXP-033/EXP-033.md`. Agrupado 0.4392 contra 0.4367 (delta +0.0025, 3 de 5): no empeora, **aceptado** por el criterio de higiene del spec. El CV KFold del modelo enviado cae de 0.4804 a 0.4623: confirma que memorizaba rescatistas. Private 0.42944 (−0.0066 contra EXP-031): falta saber si es ruido o si test comparte rescatistas con train (no verificado).
- [x] Emociones multilingües (mDeBERTa-v3 zero-shot, `lab2-prep-emo-mdeberta`, 7 min en T4) sobre EXP-033: `experiments/EXP-034/EXP-034.md`. Agrupado 0.4364 contra 0.4392 (delta −0.0028, 2 de 5, sube el desvío): **descartado**. Private 0.42962 (igual que EXP-033). Las emociones salen razonables (revisadas en el log), pero no aportan más que las de la cátedra.
- [x] `txt_score` de e5 (ridge dentro del fold, como EXP-022) sobre EXP-033: `experiments/EXP-035/EXP-035.md`. Agrupado 0.4420 contra 0.4392 (delta +0.0028, 3 de 5; desvío igual). Por la regla: **descartado**. Private 0.43393 (+0.0045 sobre EXP-033). Mismo patrón que EXP-032: CV casi plano y leaderboard mejor; falta medir con varias particiones.
- [x] Desempate con 5 particiones agrupadas (seeds 42, 1, 2, 3, 4): `experiments/EXP-037/EXP-037.md`. Inputs recalculados en `rodriguezandresed`; la seed 42 reproduce exacto EXP-031 a 035. 032 vs 031: +0.0028 (4 de 5 seeds, 16 de 25 folds). 033 vs 031: +0.0003 (3 de 5). 035 vs 033: +0.0037 (5 de 5 seeds, 17 de 25 folds). Los tres son **empate** por la regla (delta < +0.004). Promedio de las 25 folds: 031 0.4345, 032 0.4373, 033 0.4349, 035 0.4386. Con una sola partición el ruido del promedio es del orden de ±0.004: deltas de ese tamaño no se deciden con 5 folds.
- [x] Emociones de la cátedra sobre la traducción al inglés de las descripciones en malayo o chino (422 train, 75 test; NLLB-200, `preprocesado/texto-trad/`), sobre EXP-033: `experiments/EXP-036/EXP-036.md`. Control del preprocesado: con el texto original reproduce la cátedra (máx |dif| 1.2e-06). Con 5 particiones: delta +0.0002 (2 de 5 seeds, 12 de 25 folds): **empate**. KFold 0.4621 contra 0.4623. Private 0.43213 (+0.0027 sobre EXP-033; mejor sigue EXP-032, 0.44273). Junto con EXP-034: arreglar el idioma de las emociones no aporta.
- [x] `rescuer_count` sobre train + test juntos: no se corre. Train y test no comparten rescatistas (0 de 1.518), así que el conteo da igual en las dos formas.
- [x] Siguiente base después de EXP-037: EXP-033 (los empates no la cambian). Se probó la combinación en EXP-038.
- [x] CLIP promedio de todas las fotos: EXP-040, abajo.

- [ ] Siguiente base: EXP-026. Pendientes sobre ella: promedio de seeds, re-tunear hiperparámetros con las columnas nuevas, selección genética sobre todas las columnas.

### Cola sobre EXP-033 con 5 particiones (2026-11-27, `experiments/cola.py`)
- [x] EXP-038 foto promedio + txt_score e5: `experiments/EXP-038/EXP-038.md`. 5 particiones: delta +0.0037 (por seed +0.0011 a +0.0062; 5 de 5 seeds, 17 de 25 folds): **empate**. KFold 0.4651. Private 0.44165.
- [x] EXP-039 DINOv2-B foto 1: `experiments/EXP-039/EXP-039.md`. 5 particiones: delta +0.0026 (por seed +0.0003 a +0.0057; 5 de 5 seeds, 16 de 25 folds): **empate**. KFold 0.4601. Private 0.43231.
- [x] EXP-040 CLIP-L14 promedio de fotos: `experiments/EXP-040/EXP-040.md`. 5 particiones: delta +0.0069 (por seed +0.0026 a +0.0097; 5 de 5 seeds, 19 de 25 folds): **aceptar**. KFold 0.4683. Private 0.44183.
- [x] EXP-041 promedio de 5 seeds de LightGBM: `experiments/EXP-041/EXP-041.md`. 5 particiones: delta +0.0023 (por seed +0.0006 a +0.0049; 5 de 5 seeds, 16 de 25 folds): **empate**. KFold 0.4645. Private 0.43105.
- [x] EXP-042 hiperparametros re-tuneados: `experiments/EXP-042/EXP-042.md`. 5 particiones: delta +0.0034 (por seed +0.0010 a +0.0059; 5 de 5 seeds, 17 de 25 folds): **empate**. KFold 0.4690. Private 0.43159. sin la seed 42 (con la que se eligio): +0.0032; params {"n_estimators": 300, "learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 50}.
- [x] EXP-043 seleccion genetica de columnas: `experiments/EXP-043/EXP-043.md`. 5 particiones: delta +0.0026 (por seed -0.0013 a +0.0041; 4 de 5 seeds, 16 de 25 folds): **empate**. KFold 0.4597. Private 0.42818. sin la seed 42 (con la que se eligio): +0.0022; saca 9 de 41: Type, Color2, FurLength, anger, txt_palabras, txt_frac_chino, txt_frac_malayo, txt_exclamaciones, img_ancho.
- Lectura: todo lo que se probó es positivo o neutro en casi todas las particiones, pero solo CLIP promediado (EXP-040) pasa el umbral. Los cambios de imagen dan los mayores privates (040 0.44183, 038 0.44165, 032 0.44273).
- [x] Siguiente base: **EXP-040** (CLIP-L/14 promedio de fotos, aceptado). Se probaron sobre ella `txt_score`, 5 seeds, params de EXP-042 y todo junto (abajo).

### Sobre EXP-040 con 5 particiones (2026-11-27, `experiments/cola.py`)
- [x] EXP-044 txt_score e5: `experiments/EXP-044/EXP-044.md`. 5 particiones: delta -0.0001 (por seed -0.0025 a +0.0022; mejora en 3 de 5 seeds, 14 de 25 folds): **empate**. KFold 0.4708. Private 0.44237.
- [x] EXP-045 promedio de 5 seeds de LightGBM: `experiments/EXP-045/EXP-045.md`. 5 particiones: delta +0.0006 (por seed -0.0019 a +0.0037; mejora en 3 de 5 seeds, 14 de 25 folds): **empate**. KFold 0.4714. Private 0.43932.
- [x] EXP-046 hiperparametros de EXP-042: `experiments/EXP-046/EXP-046.md`. 5 particiones: delta -0.0006 (por seed -0.0030 a +0.0026; mejora en 2 de 5 seeds, 9 de 25 folds): **empate**. KFold 0.4765. Private 0.44039.
- [x] EXP-047 txt_score + 5 seeds + params de EXP-042 (todo junto): `experiments/EXP-047/EXP-047.md`. 5 particiones: delta +0.0025 (por seed +0.0003 a +0.0057; mejora en 5 de 5 seeds, 13 de 25 folds): **empate**. KFold 0.4791. Private 0.43950.
- [x] EXP-048 sin texto (ablacion: sin 7 emociones ni 8 txt_stats): `experiments/EXP-048/EXP-048.md`. 5 particiones: delta -0.0036 (por seed -0.0070 a +0.0009; mejora en 1 de 5 seeds, 6 de 25 folds): **empate en el borde**. KFold 0.4653. Private 0.42548.
- Lectura: sobre EXP-033 los tres pasos mejoraban en 5 de 5 seeds; sobre la base CLIP no se distinguen del ruido (3/5, 3/5, 2/5). Juntos (047) mejoran en 5 de 5 seeds pero +0.0025, y el private baja (0.43950). CLIP promediado parece cubrir lo que aportaban.
- Sacar el texto (048) empeora en 4 de 5 seeds (−0.0036, en el borde del umbral) y es lo que más baja el private (−0.016): el procesamiento de texto se gana su lugar, aunque cambiar el modelo de emociones (034, 036) no aporte.
- [ ] Siguiente base: EXP-040. Mejor private: EXP-032 (0.44273); EXP-044 0.44237, EXP-040 0.44183 (diferencias de ruido).

### Pipeline SOTA multi-modal y Stacking L2 (2026-11-27 / 2026-11-28, `orquestador/kernels/`)
Todos validados estrictamente en `StratifiedGroupKFold(5)` por `RescuerID` (seed 42), sin leakage (SVD, TF-IDF, stats por categoría, probes Cholesky anidados y cortes de cuantiles ajustados solo en `X_train` de cada fold).

- [x] `EXP-049` (`lab2-exp-036-fast`): `DINOv2-L` + `BGE-M3` (descripción + perfil) + `E5-L` + TF-IDF multi-vista + JSON Vision/Sentiment completos + Gold Tabular FE (3 GBDTs). CV agrupado **0.4748 ± 0.0219** (RMSE `1.0190`), Private **0.47040**.
- [x] `EXP-050` (`lab2-exp-036-chol`): reemplazo de `RidgeCV`/`LogisticRegression` por ecuaciones normales Cholesky multi-target (~150× más rápido) + 6 GBDTs (`lgbm_cls` EV primero con `0.4792`). CV agrupado **0.4853 ± 0.0229** (RMSE `1.0157`), Private **0.47776**.
- [x] `EXP-051` (`lab2-exp-037`): + `SigLIP-SO400M-384` + 12 prompts visuales zero-shot + alineación imagen-texto + 2 seeds. CV agrupado **0.4964 ± 0.0214** (RMSE `1.0071`), Private **0.48549**.
- [x] `EXP-052` (`lab2-exp-038`): + `lgbm_ordinal` (4 clasificadores binarios acumulativos $\sum_{k=1}^4 P(y \ge k)$, solo `0.4931` / RMSE `1.0079`) + `catboost_cls` + `xgb_cls`. CV agrupado **0.4967 ± 0.0248** (RMSE `1.0064`).
- [x] `EXP-053` (`lab2-exp-039`) y `EXP-054` (`lab2-exp-040-knn`): Vision Trinity + `DeBERTa-v3` en L1 (y LORO kNN en 054). CV `0.4937 ± 0.0198` y `0.4950 ± 0.0222` (RMSE `1.0096` / `1.0097`). Hallazgo: inyectar OOF débiles/ruidosos (`DeBERTa-v3`, kNN) directo en árboles L1 diluye splits; reservar diversidad neuronal para el stacker L2.
- [x] `EXP-055` (`lab2-exp-041-giant`): Vision Quartet limpio (`DINOv2-Giant 1.1B` + `SigLIP-SO400M` + `ConvNeXt-L` + `DINOv2-L`) sin kNN/DeBERTa en L1, 7 GBDTs × 2 seeds. CV agrupado **0.4971 ± 0.0235** (RMSE `1.0064`); mejores modelos individuales: `lgbm_ordinal` (`0.4959`) y `xgb_ordinal` (`0.4957`).
- [x] Modelos neuronales/transformers standalone (todos en el mismo `SGKF(5)` seed 42): `lab2-ft-mm-nn` (`0.4309`), `lab2-ft-mm-nn-giant` (`0.4500`), `lab2-ft-nn-v2` (`0.4711`), `lab2-ft-nn-v3` (`PetResNetV2` con `DINOv2-Giant`, **0.4736** / cabeza ordinal `0.4741`), `lab2-ft-deberta` (`0.3770`), `lab2-ft-vision-e2e` (`ConvNeXt-Small` + flip TTA, `0.3715`).
- [x] `EXP-056` (`lab2-stack-v2`): Level-2 RankAvg sobre 37 modelos (8 kernels). CV **0.4974 ± 0.0199**, Private **0.48782**.
- [x] `EXP-057` (`lab2-stack-final`): Level-2 NNLS sobre 50 modelos (10 kernels: ~82% GBDTs + 15.0% PyTorch NNs + 1.8% E2E ConvNeXt-S + 1.4% DeBERTa-v3). CV **0.4976 ± 0.0210** (`QWK_sh` `0.4991`), Private **0.49016** (+0.03678 sobre el #1 histórico `0.45338`).
- [ ] Opcionales ya cacheados en Kaggle (`mmaceoaustral`) para una futura iteración si se retoma: embeddings `lab2-prep-siglip2` (`SigLIP2-SO400M`) y fine-tune E2E `lab2-ft-swin-e2e` (`Swin-Base`).

### Variantes de elecciones propias
- [x] Vision extendido (crop hints, cantidad de colores) sobre EXP-006: `experiments/EXP-011/EXP-011.md`. Descartado: delta +0.0070 contra EXP-006, sube el desvío.
- [x] Sentiment del documento: `experiments/EXP-012/EXP-012.md`. Descartado: delta -0.0093, mejora en 2 de 5 folds.
- [x] SVD de imagen con 10 y 50 componentes: `experiments/EXP-013/EXP-013.md` (delta +0.0022, 3 de 5), `experiments/EXP-014/EXP-014.md` (delta -0.0126, 0 de 5). Descartados.
- [x] Nombre genérico con "baby": `experiments/EXP-015/EXP-015.md`. Descartado: delta +0.0031 contra EXP-001, mejora en 1 de 5 folds.

## Bug en los inputs
- [x] `examen-lab-ii-preprocesado-img` arma el id con `p.strip("-1.jpg")`, que borra también los `1` al principio y al final del PetID. 1840 de 14652 mascotas con foto en train y 463 de 3858 en test quedan con `img_feat_*` en NaN después del `join` (verificado contra la salida de nickel). Arreglo: `petId = p[:-len("-1.jpg")]` en las dos celdas (train y test). Arreglado en `rodriguezandresed/examen-lab-ii-preprocesado-img` v1 (0 ids rotos); los experimentos apuntan ahí. Re-corridos 017 y 018; falta 008, 013, 014, 016.

## Decisiones: el código no cumple las reglas
Las reglas de `AGENTS.md` y `checklist-verificacion.md` coinciden; los notebooks no. Para cada una: arreglar el código o anotar la excepción en `AGENTS.md` con el motivo.

- [ ] v5b calibra los umbrales sobre todas las predicciones OOF juntas, no dentro de cada fold.
- [ ] v5b ajusta TF-IDF y los dos SVD sobre train + test, y calcula `rescuer_count` con los conteos de test.
- [ ] `mdm-2025.ipynb` usa `KFold` sin grupos ni semilla: el mismo rescatista cae en train y valid.

## Sin verificar: posibles reglas
- [ ] Early stopping con el fold de valid en `eval_set`: ¿sesga el QWK OOF? Medirlo corriendo el loop de `mdm-2025.ipynb` dos veces con los mismos folds (early stopping sobre valid vs. `n_estimators` fijo) y comparar. Revisar también si la cátedra usa este esquema a propósito. Solo pasa a `AGENTS.md` si se confirma.

## Sin verificar
- [ ] Correr en Kaggle el loop nuevo de `mdm-2025.ipynb` y anotar el QWK (el último registrado, 0,350, es del loop anterior).
- [ ] En ese loop: sobra el primer `learner.fit(X_train, y_train, )`; el segundo `fit` reentrena desde cero.
- [ ] En ese loop: el early stopping también mira `multi_logloss`. Para parar solo por QWK: `LGBMClassifier(metric="None")`.

## Limpieza de `AGENTS.md`
- [ ] Completar o borrar los `<< >>` y las instrucciones de plantilla (líneas 3 a 5).
- [ ] Estructura y comandos describen `src/`, `configs/` y `pytest`; el proyecto son notebooks de Kaggle (v0 a v5b).
