# Modelo v2 (robusto a la captura en webcam) — modelo OFICIAL

**Estado:** modelo oficial y por defecto de la app desde el **20 nov. 2026**, por decisión del integrador (Acosta).
v1 queda como línea base congelada (`reports/models/`, `artifacts/`).

> **Declaración obligatoria para el artículo: el split de test se usó DOS veces.** Primero con v1 (`reports/models/test_evaluation.json`);
> después con v2, por decisión explícita del integrador, registrada y verificable en `reports/models_v2/test_authorization.json`. v2 se diseñó
> **después** de conocer el resultado de test de v1; ninguna decisión de v2 (aumentación, rangos, escenarios de robustez, selección, umbral)
> se basó en predicciones ni métricas de test (se usaron el corpus de entrenamiento, la validación y capturas de webcam). Aun así, el test
> **dejó de ser una evaluación única**, y así debe informarse. v2 se evaluó en test **una sola vez** (`final-eval` rechaza repetirlo).

## Resultados en test (131 secuencias procesables de 140)

Fuentes: `reports/models/test_evaluation.json` (v1), `reports/models_v2/test_evaluation.json` (v2) y
`reports/models_v2/paired_test_comparison.json` (`python -m src.lsc_models compare-test`).

| | v1 (`transformer_aug`) | **v2 (`bilstm_robust`)** |
|---|---|---|
| Desplegado (semilla 17): exactitud / F1 macro | 0,870 / 0,872 | **0,931 / 0,931** |
| Desplegado: desempeño integral (rechazos = error) | 0,814 | 0,871 |
| 3 semillas, exactitud (media ± desv.) | 0,878 ± 0,008 | **0,921 ± 0,009** |
| Umbral → cobertura / exactitud entre aceptados | 0,940 → 0,809 / 0,962 | 0,908 → 0,794 / 0,971 |
| Baseline de mayoría (exactitud) | 0,099 | 0,099 |
| Metas (exactitud ≥ 0,85; F1 ≥ 0,80) | cumplidas | cumplidas |

Comparación **pareada** de los modelos desplegados, mismas 131 secuencias: v1 acierta 114, v2 acierta 122; **solo v2 acierta en 9, solo v1
en 1**, ambos en 113, ninguno en 8. Diferencia de exactitud +0,061 (IC95 % bootstrap +0,015 a +0,107; McNemar exacto p = 0,021).

**Cómo leerlo con prudencia:**
- Con n = 131, una secuencia vale 0,0076. La mejora es consistente (las 3 semillas de v2 superan a las 3 de v1) pero el intervalo es ancho.
- La semilla desplegada (17, fijada antes) fue la mejor de v2 (0,931 frente a 0,916 de las otras dos) y la peor de v1 (0,870 frente a 0,878 de
  media). La comparación por media de semillas (**0,921 frente a 0,878**) es la menos favorable para v2 y la más representativa; informar ambas.
- Las metas se cumplen, pero **no se midió el rechazo de negativos** ni se evaluó con personas ajenas al corpus.
- El umbral de v2 es **provisional** (ajustado en validación original + perturbada, ver abajo).

## Por qué existe v2
Con la app real y webcam, v1 casi nunca superaba el umbral (0,94). En 27 intentos guardados localmente (`run-app --debug-dump`; 23 pasaron
calidad) la entrada en vivo difería de la del corpus:

| Medida | Corpus (train) | Webcam (23 intentos de 1 persona) |
|---|---|---|
| Hombros / ancho de imagen | 0,20 (30 cuadros del corpus) | 0,37 |
| Cuadros con una sola mano | 27 % | 73 % |
| Cuadros con ambas manos | 70 % | 22 % |
| Secuencias con una mano casi siempre ausente (izq./der.) | 9 % / 6 % | 30 % / 35 % |
| Puntos de mano con y > 0,8 (zona baja) | 29 % | 0 % |

- Con ambas manos detectadas v1 daba confianza ≥ 0,98; con una sola mano, `LOW_CONFIDENCE`. La persona está más cerca y la mano que no
  firma queda fuera del encuadre; el corpus casi siempre la muestra. La aumentación de v1 (`train_v1`) no simula recortes ni manos ausentes.
- **Límite:** capturas de una persona, guardadas en `artifacts/debug_dump/` (contienen su rostro; ignoradas por Git, no se publican). Estas
  cifras no se reproducen desde el repositorio; el procedimiento sí (`run-app --debug-dump`, `diagnose-dump`). Solo fijaron rangos de la
  aumentación; **no entran al entrenamiento ni a la selección**.

## Qué cambia en v2
- Aumentación `train_v2_robust` (`src/lsc_models/augment_robust.py`; perturbaciones en `perturb.py`): recorte inferior y lateral (la mano fuera
  de cuadro se quita **completa**), una mano ausente toda la secuencia, pérdidas por cuadro, escala/traslación/rotación leves, ruido y
  repetición de un cuadro. Lo que sale **sigue siendo procesable** (cumple los criterios de calidad de la app).
- Mismas arquitecturas y tamaños que v1 (BiLSTM y Transformer compacto), mismo presupuesto; 2 configuraciones × 3 semillas.
- Configuración propia `configs/train/experiments_v2.yaml`. **`experiments.yaml` no se editó**; una prueba verifica que los `config_hash` de v1
  coinciden con `freeze.json`. Resultados en `reports/models_v2/`, paquete en `artifacts_v2/`.
- Selección (fijada antes de entrenar): media de F1 macro en validación original y perturbada (`val_mixed_f1`); desempate por menor p95;
  semilla desplegada 17. Eligió `bilstm_robust`.
- Umbral 0,908: `export-artifacts` lo ajusta sobre validación original + perturbada (50/50). Sigue siendo provisional.

## Evidencia previa al test (validación, `reports/models_v2/robustness_comparison.csv`)
63 secuencias procesables, media de 3 semillas, exactitud:

| | v1 desplegado | v2 `bilstm_robust` |
|---|---|---|
| Validación original | 0,915 | 0,942 |
| Sin mano derecha | 0,548 | 0,786 |
| Recorte inferior a y = 0,7 | 0,769 | 0,936 |
| «Webcam-like» (una mano ausente + recorte) | 0,655 | 0,905 |
| Media de 8 escenarios | 0,773 | 0,905 |

Las perturbaciones son del mismo tipo que las del entrenamiento y se usaron también para elegir: miden robustez a esos fallos, no
generalización a la webcam. Metas de v2 declaradas antes de entrenar: caída ≤ 0,03 en validación original (cumplida) y mejora ≥ 0,15 en «sin
mano derecha» y «recorte 0,7» (cumplida frente a v1 desplegado; frente al mejor v1 de «recorte 0,7», +0,135).

## Sesión de webcam (sin etiquetas)
111 intentos de una persona, todos a la misma distancia (hombros/ancho ≈ 0,39). Con las mismas entradas, v1 acepta el 60 % (62/103) y v2 el
72 % (74/103); v1 con el umbral de v2, 65 %. v1 y v2 coinciden en la seña en el 80 % de los intentos. «Aceptado» no es «acertado».

## Qué NO se midió
- Exactitud etiquetada en webcam (falta el orden real de la sesión), ronda a otra distancia, personas ajenas al corpus.
- Rechazo de negativos. El modelo no puede recuperar la información de una mano ausente: en señas que difieren sobre todo por la segunda
  mano habrá un techo. La distribución del top-1 en webcam (Gustar 20, Licor 16, Buenos 4, Hola 6) sugiere confusiones aún sin medir.

## Para reproducir
`run-all --config configs/train/experiments_v2.yaml --out reports/models_v2` → `aggregate` (mismo `--config` y `--out`) →
`robustness` → `export-artifacts --reports reports/models_v2 --out artifacts_v2` → `authorize-test` → `freeze` → `final-eval`
(mismos `--config`, `--reports`, `--artifacts`) → `compare-test`.
