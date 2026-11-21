# Checklist — revisar el trabajo del agente

> El agente es dueño de la implementación. Vos sos dueño/a de la verificación.

## 1. ¿Hizo lo que pedía el spec?
- [ ] El diff toca solo los archivos esperados.
- [ ] No cambió folds, métrica, semilla ni features existentes.
- [ ] No "arregló" cosas que no se le pidieron.

## 2. Leakage
- [ ] Todo `fit` (encoders, TF-IDF, SVD, scalers, thresholds) ocurre dentro del fold, sobre train del fold.
- [ ] `test.csv` no se usa salvo para el submission.
- [ ] Ninguna feature usa el target fuera de fold.
- [ ] `RescuerID` no cruza train y valid.

## 3. Resultados creíbles
- [ ] Reporta QWK por fold, media y desvío (no solo un número).
- [ ] El delta es razonable (> +0.02 de golpe = sospechar).
- [ ] Mejora en la mayoría de los folds, no solo en promedio.
- [ ] Puedo reproducir el número corriendo el comando yo mismo/a.

## 4. Entiendo el código
- [ ] Puedo explicar en 2 frases qué hace cada función nueva.
- [ ] Si no entiendo algo, le pido al agente que lo explique o lo simplifique *antes* de aceptarlo.

## 5. Registro y decisión
- [ ] Fila nueva en `experiments/log.csv`.
- [ ] Decisión explícita: aceptar / descartar / investigar, con una línea de por qué.
- [ ] Si se acepta, es el nuevo baseline.
