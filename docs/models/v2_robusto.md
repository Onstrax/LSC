# Modelo v2: robusto a la captura en webcam

**Modelo por defecto de la app desde el 20 nov. 2026**, por decisión del integrador (Acosta). La decisión se tomó **antes** de
tener la exactitud etiquetada en webcam; ver «Estado de la evidencia» al final. La app carga `artifacts_v2/` por defecto
(`api/constants.py: DEFAULT_ARTIFACTS_DIR`, `run-app`, `deployment/compose.yaml`); para usar v1: `run-app --artifacts artifacts` o
`LSC_ARTIFACTS_DIR=artifacts`. Una prueba (`tests/app/test_default_model.py`) mantiene alineadas todas las capas y verifica que
`export-artifacts` y `final-eval` siguen apuntando a v1.

**Estado:** mejora **posterior al congelamiento y a la evaluación de test de v1**. v1 sigue siendo el modelo
congelado (`reports/models/freeze.json`, `artifacts/`) y su cifra de test (exactitud 0,870 sobre 131 procesables,
`reports/models/test_evaluation.json`) es la del artículo para v1. **v2 no tiene cifra de test**: el test se abrió una sola vez
(con v1) y no se reabre (`evaluation.test_sealed: true`; `freeze` rechaza esta configuración).

## Por qué
Con la app real y webcam, v1 casi nunca superaba el umbral (0,94). En 27 intentos guardados localmente
(`run-app --debug-dump`; 23 pasaron calidad) se midió que la entrada en vivo difiere de la del corpus:

| Medida | Corpus (train) | Webcam (23 intentos de 1 persona) |
|---|---|---|
| Hombros / ancho de imagen | 0,20 (30 cuadros del corpus) | 0,37 |
| Cuadros con una sola mano | 27 % | 73 % |
| Cuadros con ambas manos | 70 % | 22 % |
| Secuencias con una mano casi siempre ausente (izq./der.) | 9 % / 6 % | 30 % / 35 % |
| Puntos de mano con y > 0,8 (zona baja) | 29 % | 0 % |

- Con ambas manos detectadas el modelo daba confianza ≥ 0,98; con una sola mano, `LOW_CONFIDENCE`.
- Causa: la persona está más cerca y la mano que no firma queda fuera del encuadre. El corpus casi siempre la muestra.
- La aumentación de v1 (`train_v1`: rotación ±5° y ruido 0,005) no simula recortes, manos ausentes ni escala.

**Límite de la medición:** son capturas de **una persona**, guardadas en `artifacts/debug_dump/` (contienen su rostro; ignoradas por Git,
no se publican). Por eso estas cifras no se reproducen desde el repositorio; el procedimiento sí: `run-app --debug-dump` y
`diagnose-dump`. Solo se usaron para fijar rangos de la aumentación; **no entran al entrenamiento ni a la selección**.

## Qué cambia en v2
- Aumentación `train_v2_robust` (`src/lsc_models/augment_robust.py`, perturbaciones en `perturb.py`): recorte inferior y lateral (la
  mano fuera de cuadro se quita **completa**, como en una detección real), una mano ausente toda la secuencia, pérdidas por cuadro,
  escala/traslación/rotación leves, ruido y repetición de un cuadro. Todo lo que sale **sigue siendo procesable** (cumple los criterios
  de calidad de la app).
- Mismas arquitecturas y tamaños que v1 (BiLSTM y Transformer compacto), mismo presupuesto. Dos configuraciones × 3 semillas.
- Configuración propia: `configs/train/experiments_v2.yaml`. **No se editó `experiments.yaml`**; una prueba verifica que los
  `config_hash` de v1 coinciden con los de `freeze.json`.
- Resultados en `reports/models_v2/`; paquete en `artifacts_v2/`. `artifacts/` (v1) no se toca.

## Cómo se evalúa sin el test
1. **Validación original** (63 secuencias procesables), contra v1 (exactitud 0,915 / F1 0,915).
2. **Batería de robustez sobre validación** (`robustness.py`): sin mano izquierda, sin mano derecha, recorte inferior a y = 1,0 / 0,7 / 0,5,
   una mano ausente por tramos, escala/traslación, y «webcam-like» (una mano ausente + recorte 0,7). Solo cuentan las muestras que la app
   aún aceptaría tras la perturbación. Comando: `python -m src.lsc_models robustness`.
3. **Selección** (fijada antes de entrenar): media de F1 macro en validación original y en validación perturbada (`val_mixed_f1`); desempate
   por menor p95; semilla desplegada 17.
4. **Sesión de webcam etiquetada** (una persona, N pequeño): diagnóstica; no se generaliza en el artículo.

### Metas declaradas antes de entrenar (se informan aunque no se cumplan)
- Exactitud en validación original: caída ≤ 0,03 respecto a v1.
- Exactitud en «sin mano derecha» y en «recorte 0,7»: mejora ≥ 0,15 respecto a v1.

## Qué NO se midió
- Test de v2 (sellado). Rechazo de negativos. Personas distintas a las del corpus y a la persona de la sesión de webcam.
- Que el modelo no puede recuperar la información de una mano ausente: en señas que difieren sobre todo por la segunda mano habrá un techo.

## Estado de la evidencia (20 nov. 2026)
- **Validación** (`reports/models_v2/robustness_comparison.csv`, 63 procesables, media de 3 semillas): `bilstm_robust` 0,942 en original
  (v1 desplegado 0,915) y 0,905 de media en los 8 escenarios perturbados (v1 desplegado 0,773). Las perturbaciones son del mismo tipo que
  las del entrenamiento y se usaron también para elegir: es robustez a esos fallos, no generalización a la webcam.
- **Sesión de webcam sin etiquetas** (111 intentos de una persona, todos cerca: hombros/ancho mediana 0,39). Con las mismas entradas,
  v1 acepta 60 % (62/103) y v2 72 % (74/103); v1 con el umbral de v2, 65 %. v1 y v2 coinciden en la seña en el 80 % de los intentos.
  «Aceptado» no es «acertado».
- **Pendiente:** exactitud por seña con el orden real de la sesión, ronda a mayor distancia, rechazo de negativos, test (sellado).
  La distribución del top-1 de v2 (Gustar 20, Licor 16, Buenos 4, Hola 6) sugiere confusiones entre señas aún sin medir.

## Declaraciones para el artículo
- v2 es el modelo por defecto de la app, adoptado antes de medir su exactitud etiquetada en webcam.
- v2 se entrenó **después** del congelamiento y del test de v1; no hay cifra de test para v2.
- Los rangos de la aumentación se fijaron con apoyo de capturas de webcam de una persona (no usadas para entrenar ni elegir).
