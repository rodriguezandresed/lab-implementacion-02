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
- [ ] Siguiente base: EXP-033 (vigente por regla). Pendiente: desempatar EXP-032 con repeticiones de la partición; probar DINOv2-B.
- [ ] CLIP promedio de todas las fotos (`--agg mean`, ya calculado en `lab2-prep-clip-l14`).

- [ ] Siguiente base: EXP-026. Pendientes sobre ella: promedio de seeds, re-tunear hiperparámetros con las columnas nuevas, selección genética sobre todas las columnas.

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
