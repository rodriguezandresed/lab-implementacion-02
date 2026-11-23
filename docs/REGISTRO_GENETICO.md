# Registro de corridas del algoritmo genético (PetFinder)

Una fila por corrida. El kappa es out-of-fold (5 folds `StratifiedGroupKFold`, `SEED = 42`, `LGBMRegressor(**BEST_PARAMS)`, umbrales por cuantiles). El LB público se completa a mano después de enviar.

Ojo: el kappa de las variantes genéticas es optimista, porque se eligieron mirando estos mismos folds. Diferencias de ±0,003 están dentro del ruido.

## Referencias (sin algoritmo genético)

| Notebook | Variante | Columnas | Kappa OOF |
|---|---|---|---|
| v4 | sin componentes de texto | 26 | 0,3955 |
| v4 | con TF-IDF + SVD | 46 | 0,3879 |
| v5b | solo TF-IDF (v4) | 46 | 0,3879 |
| v5b | solo embeddings | 46 | 0,3880 |
| v5b | TF-IDF + embeddings | 66 | 0,3842 |

Las 26 columnas base (sin componentes SVD): `Type`, `Age`, `Breed1`, `Breed2`, `Gender`, `Color1`, `Color2`, `Color3`, `MaturitySize`, `FurLength`, `Vaccinated`, `Dewormed`, `Sterilized`, `Health`, `Quantity`, `Fee`, `State`, `VideoAmt`, `PhotoAmt`, `rescuer_count`, `sin_nombre`, `desc_len`, `desc_words`, `desc_upper_ratio`, `desc_exclam`, `desc_vacia`.

## Corridas

| # | Fecha | Notebook | Algoritmo | Base | Config | Kappa OOF | Columnas | LB público | Notas |
|---|---|---|---|---|---|---|---|---|---|
| 1 | | v4 | selección | 46 (con TF-IDF) | pob 10, gen 6 | 0,3898 | 33 de 46 | | arrancó en 0,3879 (todas); no llegó a sacar las 20 SVD |
| 2 | | v4 | selección | 26 (sin texto) | pob 10, gen 6 | 0,3955 | 26 de 26 | | no sacó ninguna: ningún subconjunto mejora a las 26 |
| 3 | | v5b | selección | 46 (solo embeddings) | pob 10, gen 3 | 0,3905 | 42 de 46 | | antes de sembrar el individuo sin embeddings |
| 4 | | v5b (versión 1) | selección | 46 (solo embeddings) | pob 10, gen 3, sin semilla | 0,3929 | 39 de 46 | | afuera: `emb_svd_0`, `emb_svd_3`, `emb_svd_7`, `emb_svd_13`, `emb_svd_14`, `emb_svd_17`, `sin_nombre` |
| 5 | | v5b (versión 1) | construcción | 46 (solo embeddings), fórmulas sobre las 26 | pob 10, gen 3, barrido n_feat | 0,3935 | 46 + 10 | 0,36207 (private) | **enviada** (versión 1); fórmulas abajo |
| 6 | | v5b (versión 2) | selección | 46 (solo embeddings) | pob 20, gen 20, con semilla sin embeddings | **0,3965** | 25 de 46 | **0,37171** (private) | **enviada** (versión 2); 12,5 min; columnas abajo |
| 7 | | v5b (versión 2) | construcción | igual que la 5 | igual que la 5 | 0,3935 | 46 + 10 | | mismo resultado que la 5 (misma semilla, mismo código) |

## Columnas ganadoras

### Corrida 6 (selección, la mejor hasta ahora: 0,3965)

25 columnas: las 26 base menos `desc_vacia` y `desc_words`, más `emb_svd_4`.

```python
cols_ganadoras = ["Type", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                  "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health",
                  "Quantity", "Fee", "State", "VideoAmt", "PhotoAmt", "rescuer_count", "sin_nombre",
                  "desc_len", "desc_upper_ratio", "desc_exclam", "emb_svd_4"]
```

Lectura: arrancó en 0,3955 (el individuo sembrado sin embeddings) y ganó +0,001, dentro del ruido. Sacó `desc_vacia` y `desc_words`, que repiten lo que ya dice `desc_len`, y de las 20 componentes del embedding solo se quedó con una. En la práctica es el modelo de 26 columnas sin texto.

### Corridas 5 y 7 (construcción, barrido de n_feat)

Las 10 columnas nuevas del ganador (n_feat = 10), completas (salieron impresas en la corrida 7; la 5 dio lo mismo):

`Dewormed mas desc_exclam`, `Gender div Color2`, `Type por MaturitySize`, `rescuer_count menos Dewormed`, `VideoAmt por Gender`, `Breed2 por desc_upper_ratio`, `PhotoAmt por Color3`, `Color1 div Color3`, `Fee menos FurLength`, `Breed2 mas MaturitySize`

Barrido completo:

| n_feat | Kappa OOF | Columnas nuevas |
|---|---|---|
| 1 | 0,3921 | `FurLength div Sterilized` |
| 2 | 0,3915 | `Gender menos rescuer_count`, `desc_vacia div Age` |
| 3 | 0,3915 | `desc_vacia por desc_len`, `Gender menos rescuer_count`, … (truncado) |
| 5 | 0,3925 | `rescuer_count div Breed1`, `Vaccinated menos MaturitySize`(?), … (truncado) |
| 8 | 0,3926 | `Fee mas FurLength`, `Breed2 div MaturitySize`, … (truncado) |
| 10 | 0,3935 | las 10 de arriba |

Las listas de n_feat 3, 5 y 8 quedaron truncadas por la tabla de pandas; la celda de variantes solo imprime completa la del ganador.

Lectura: la curva es plana (0,3915 a 0,3935); la cantidad de fórmulas casi no importa. Las fórmulas no tienen sentido de negocio claro, señal de que ajustan ruido de estos folds.

## Pendiente

- [x] Corrida 6: v5b con la semilla sin embeddings y selección pob 20 × gen 20. Dio 0,3965 (esperado ≥ 0,3955).
- [x] LB (private, 2026-11-23): versión 1 (construcción) 0,36207, versión 2 (selección) 0,37171; v3 sin genético 0,36805. La selección suma +0,004; la construcción resta.
