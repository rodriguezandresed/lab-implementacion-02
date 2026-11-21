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
- [ ] EXP-019 + EXP-021 juntos: `experiments/EXP-023/EXP-023.md`. Corriendo.
- [ ] Detección de animales (DETR): `preprocesado/img-deteccion/`. Corriendo; el experimento va después.
- [ ] El CV agrupado dijo "descartar" a 019 y 021 y el leaderboard los puso primeros: revisar por qué cae el fold 4 (¿early stopping de 5 rondas inestable?).

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
