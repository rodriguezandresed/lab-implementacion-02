# Spec de experimento — plantilla

> Un experimento = una hipótesis. Si necesitás la palabra "y" para describirlo, probablemente son dos.

```markdown
# EXP-###: <título corto>

## Hipótesis
Creemos que <cambio> mejora QWK porque <razón basada en datos o dominio>.

## Cambio exacto
- Qué se agrega/modifica (archivos, funciones, parámetros).

## Qué NO tocar
- Folds, métrica, semilla, features existentes, hiperparámetros del modelo.

## Evaluación
- Baseline vigente: EXP-### con QWK CV = <media ± desvío>.
- Medir: QWK por fold, media y desvío; importancia de las features nuevas.

## Criterio de decisión
- Aceptar si: delta media ≥ <umbral> y mejora en ≥ 4 de 5 folds.
- Descartar si: delta < <umbral> o aumenta el desvío.
- Sospechar (y frenar) si: delta > 0.02.

## Entregable del agente
- Código en <ruta>, fila en experiments/log.csv, resumen de ≤ 10 líneas.
```

---

# Ejemplo completo

```markdown
# EXP-007: Features de texto de Description (TF-IDF + SVD)

## Hipótesis
Creemos que el contenido de la descripción mejora QWK porque las publicaciones
que mencionan vacunación, carácter ("friendly", "playful") o urgencia ("urgent",
"need home") podrían adoptarse a otra velocidad, y hoy el modelo solo usa
el sentiment agregado.

## Cambio exacto
- Nuevo src/features/text_tfidf.py con build_text_features(train_df, valid_df) -> (X_tr, X_va).
- TF-IDF (word 1–2 gramas, min_df=3, max_features=20000, sublinear_tf=True) sobre Description
  (NaN → ""), luego TruncatedSVD(n_components=16, random_state=42).
- Fit de TF-IDF y SVD SOLO con train del fold.
- Agregar las 16 columnas svd_desc_0..15 al modelo LightGBM existente.

## Qué NO tocar
- Folds (data/folds.csv), métrica, semilla, resto de features, hiperparámetros de LightGBM.

## Evaluación
- Baseline vigente: EXP-005, QWK CV = 0.441 ± 0.012.
- Medir: QWK por fold, media, desvío; gain de las 16 features nuevas.

## Criterio de decisión
- Aceptar si: delta media ≥ +0.004 y mejora en ≥ 4/5 folds.
- Descartar si: delta < +0.004.
- Sospechar si: delta > +0.02 → revisar que TF-IDF no vea valid/test.

## Entregable del agente
- src/features/text_tfidf.py + test que verifique que el vocabulario no cambia al agregar filas de valid.
- Fila en experiments/log.csv y resumen de ≤ 10 líneas.
```

> Los números del ejemplo (0.441, umbrales) son ilustrativos: reemplazalos por los de tu baseline.

## Ideas de experimentos para PetFinder (una hipótesis cada una)
- Metadata de imágenes: promedio de `score` de labels de Google Vision y `dominantColors` de la primera foto.
- Sentiment: magnitude y score por oración (no solo el documento).
- Rescatista: cantidad de publicaciones por `RescuerID` (count, no target encoding).
- Thresholds: optimizar los cortes de regresión→clase dentro de cada fold (OptimizedRounder).
- Embeddings de imagen: CNN preentrenada sobre la foto 1 + SVD, con fit dentro del fold.
- Nombre: flags "sin nombre" / "nombre genérico" (No Name, Puppy, Kitty).
