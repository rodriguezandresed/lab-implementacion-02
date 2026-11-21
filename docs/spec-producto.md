# Mini-spec de producto — plantilla

> Antes de construir, escribir. Con agentes, construir es barato: lo caro es construir lo equivocado.

```markdown
# <Nombre del producto>

## Usuario
¿Quién exactamente? (rol, contexto, frecuencia de uso)

## Problema / trabajo a resolver
¿Qué intenta lograr hoy y qué se lo impide?

## Decisión que habilita
¿Qué va a hacer distinto el usuario gracias a esto?

## Métrica de valor (no de modelo)
¿Cómo sabremos que generó valor? (ej. días hasta adopción, % adoptados a 30 días)
Métrica proxy offline: ¿qué podemos medir hoy con el dataset?

## Qué predecimos / generamos
¿Qué modelo o componente AI necesitamos y con qué input?

## MVP (lo mínimo para aprender)
3–5 bullets. ¿Qué dejamos explícitamente afuera?

## Riesgos y supuestos
- Supuesto más riesgoso: ...
- Cómo lo validamos barato: ...
```

---

# Ejemplo: Asistente de publicación para refugios

```markdown
# Asistente de publicación para refugios

## Usuario
Voluntario/a de un refugio chico que publica 5–20 mascotas por mes, sin experiencia en marketing.

## Problema / trabajo a resolver
Quiere que cada mascota se adopte rápido, pero no sabe qué hace "buena" una publicación.
Hoy escribe descripciones cortas, sube 1 foto y pone precio a ojo.

## Decisión que habilita
Antes de publicar, mejora lo que SÍ puede cambiar: cantidad y calidad de fotos,
texto de la descripción, precio (Fee), info de salud (vacunación, desparasitación, esterilización).
No puede cambiar: edad, raza, tamaño, color.

## Métrica de valor
- Real: mediana de días hasta adopción (requiere piloto con refugios).
- Proxy offline: mejora en la AdoptionSpeed predicha por nuestro modelo al aplicar las sugerencias,
  solo sobre features accionables.

## Qué predecimos / generamos
1. Modelo PetFinder (el de la competencia) como "simulador": predice AdoptionSpeed.
2. Contrafácticos: para cada feature accionable, ¿cuánto mejora la predicción si cambia?
   (ej. PhotoAmt 1 → 4, Fee 100 → 0, Vaccinated 3 → 1).
3. LLM que reescribe la descripción (cálida, específica, con datos de salud) → se re-evalúa con el modelo.

## MVP
- Input: una fila del dataset (una publicación existente).
- Output: top-3 sugerencias accionables ordenadas por impacto predicho + descripción reescrita.
- Interfaz: notebook o app Streamlit/Gradio de una pantalla.
- Fuera de alcance: login, carga de fotos reales, integración con PetFinder.

## Riesgos y supuestos
- Supuesto más riesgoso: que el modelo capture causalidad. Es correlacional: "más fotos" puede ser
  proxy de "rescatista más dedicado". → Mostrar sugerencias como hipótesis, no promesas.
- Riesgo: el LLM inventa datos (ej. "vacunado" cuando no lo está). → El prompt solo puede usar
  campos de la fila; validar que no agregue hechos.
- Riesgo: optimizar el texto para el modelo y no para humanos (Goodhart). → Revisión humana de 10 ejemplos.
```
