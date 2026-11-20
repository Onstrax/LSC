# Metodología de modelado y resultados (Acosta)

> Estado: borrador con cifras reales. **Cada cifra sale del archivo que se indica** (`reports/models/` = v1,
> `reports/models_v2/` = v2); no hay ninguna cifra sin fuente. Los datos sintéticos no aparecen en este documento.
> Lo que no se midió se declara en la sección 5. Marcado **[PENDIENTE]**: lo que debe completar una persona.

## 1. Método

- **Datos:** subconjunto LSC70W: 10 señas aisladas (Hola, Buenos, Días, Tardes, Noches, Yo, Nombre, Años, Gustar, Licor), 70 personas,
  N = 700 secuencias (`artifacts/data/v1/features.npz`).
- **Particiones por persona** (sin fuga): train 490 secuencias / 49 personas; validación 70 / 7; test 140 / 14. Procesables (el extractor
  acepta la captura): 462 / 63 / 131.
- **Entrada:** 6 instantes × 49 puntos × (x, y, presencia) = (6, 147), float32, normalizada por hombros (extractor de Ortiz, `features_v1`).
  «Procesable» = hombros válidos en ≥ 5 cuadros y ≥ 1 mano en ≥ 4 cuadros (`contracts/feature_spec.json`).
- **Modelos:** BiLSTM (proyección 64 → BiLSTM 64 → softmax) y Transformer compacto (proyección 64, embedding posicional, 2 bloques de 4
  cabezas, FF 128, promedio temporal → softmax). ≈ 77 mil parámetros cada uno (76 810 y 77 578).
- **Presupuesto común:** Adam, lr 1e-3, lote 32, máximo 80 épocas, paciencia 12, restauración de los mejores pesos.
- **Referencia:** clasificador de clase mayoritaria.
- **Métricas:** exactitud y F1 macro sobre **procesables**; **desempeño integral** (los rechazos de calidad cuentan como error); y el
  rechazo de negativos, que **no se midió** (sección 5). Las metas (exactitud ≥ 0,85; F1 macro ≥ 0,80; p95 < 2 s) no son resultados.

### 1.1 Modelo v1 (línea base congelada)
- Aumentación `train_v1` (rotación global ±5° y ruido σ = 0,005, solo en entrenamiento).
- Diseño: 2 arquitecturas × {sin, con} aumentación × semillas 17, 29, 43 = 12 ejecuciones.
- Selección (fijada antes de abrir el test): F1 macro medio de validación; diferencia ≤ 0,01 → menor p95; semilla 17 desplegada.

### 1.2 Modelo v2 (robusto a la webcam; **modelo oficial**)
Se creó **después** de congelar y evaluar v1 en test, al ver que v1 casi nunca daba una sugerencia con webcam (sección 3).
- Aumentación `train_v2_robust`: recorte inferior/lateral que quita la mano completa, una mano ausente toda la secuencia, pérdidas por
  cuadro, escala/traslación/rotación leves, ruido y repetición de un cuadro. Lo aumentado sigue siendo procesable.
- Mismas arquitecturas, tamaños y presupuesto; 2 configuraciones (`bilstm_robust`, `transformer_robust`) × 3 semillas = 6 ejecuciones.
- Selección (fijada antes de entrenar): media de F1 macro en validación original y en validación perturbada; desempate por p95.
- Detalle y parámetros: `docs/models/v2_robusto.md`, `configs/train/experiments_v2.yaml`.

## 2. Resultados de v1

**Validación** (63 procesables; media ± desv. de 3 semillas). Fuente: `reports/models/ablation_summary.csv`.

| Configuración | Exactitud | F1 macro | Exactitud integral | p95 del modelo (ms) |
|---|---|---|---|---|
| BiLSTM sin aumentación | 0,915 ± 0,009 | 0,914 ± 0,009 | 0,824 | 39,9 |
| BiLSTM con aumentación | 0,905 ± 0,000 | 0,904 ± 0,001 | 0,814 | 51,2 |
| Transformer sin aumentación | 0,926 ± 0,009 | 0,925 ± 0,011 | 0,833 | 39,1 |
| **Transformer con aumentación** | 0,915 ± 0,009 | 0,915 ± 0,011 | 0,824 | 37,4 |

Selección (`reports/models/selection.json`): **empate técnico** (≤ 0,01) entre `transformer_noaug` y `transformer_aug`; ganó `transformer_aug`
por menor p95. La diferencia entre ambas es de 2 secuencias sobre 189 (3 semillas × 63): la validación no las distingue. Una secuencia de
validación vale 0,016 de exactitud.

**Sensibilidad a la pérdida de manos** (validación, `transformer_aug`, exactitud media; `reports/models/sensitivity_hand_dropout.csv`;
diagnóstico, no se ajustó nada con él):

| Fracción de cuadros sin mano | 0 | 0,17 | 0,33 | 0,50 | 0,67 |
|---|---|---|---|---|---|
| Se pierden ambas manos | 0,915 | 0,860 | 0,792 | 0,713 | 0,601 |
| Se pierde solo la izquierda | 0,915 | 0,904 | 0,892 | 0,880 | 0,865 |
| Se pierde solo la derecha | 0,915 | 0,867 | 0,839 | 0,790 | 0,728 |

Perder la mano derecha duele mucho más que perder la izquierda: el modelo depende sobre todo de la derecha.

**Test** (evaluación de v1; 131 procesables de 140). Fuente: `reports/models/test_evaluation.json`. Desplegado (semilla 17): exactitud
**0,870**, F1 macro **0,872**, integral 0,814. Por semilla: 0,870 / 0,885 / 0,878 (media 0,878 ± 0,008). Metas cumplidas. Baseline de
mayoría: 0,099. Con el umbral provisional 0,940: cobertura 0,809, exactitud entre aceptados 0,962.

## 3. Hallazgo en webcam y origen de v2

Con la app real, v1 casi nunca superaba el umbral. Se guardaron localmente 27 intentos de **una persona** (23 con calidad válida) y se
compararon con el corpus. Estas cifras provienen de capturas con rostro que **no se publican** y por tanto no son reproducibles desde el
repositorio (sí el procedimiento: `run-app --debug-dump`, `diagnose-dump`):

| Medida | Corpus (train) | Webcam |
|---|---|---|
| Hombros / ancho de imagen | 0,20 | 0,37 |
| Cuadros con una sola mano | 27 % | 73 % |
| Secuencias con una mano casi siempre ausente (izq./der.) | 9 % / 6 % | 30 % / 35 % |
| Puntos de mano en la zona baja (y > 0,8) | 29 % | 0 % |

Con ambas manos detectadas, v1 daba confianza ≥ 0,98; con una sola, `LOW_CONFIDENCE`. La persona estaba más cerca y la mano que no firma
quedaba fuera del encuadre, algo que el corpus casi nunca muestra. Estas capturas fijaron rangos de la aumentación de v2; **no se usaron para
entrenar ni para elegir**.

## 4. Resultados de v2

**Validación** (media ± desv. de 3 semillas). Fuente: `reports/models_v2/ablation_summary.csv`.

| Configuración | Exactitud | F1 macro | F1 macro perturbada | Mixto (criterio) | p95 (ms) |
|---|---|---|---|---|---|
| **BiLSTM robusto** | 0,942 ± 0,009 | 0,938 ± 0,008 | 0,887 | 0,913 | 52,7 |
| Transformer robusto | 0,910 ± 0,009 | 0,909 ± 0,008 | 0,878 | 0,893 | 48,5 |

Selección (`reports/models_v2/selection.json`): `bilstm_robust` por mayor F1 mixto (la diferencia con la otra supera la tolerancia).

**Batería de robustez sobre validación** (exactitud media; fuente: `reports/models_v2/robustness_comparison.csv`):

| Escenario | v1 desplegado | v2 `bilstm_robust` |
|---|---|---|
| Original | 0,915 | 0,942 |
| Sin mano izquierda | 0,767 | 0,899 |
| Sin mano derecha | 0,548 | 0,786 |
| Recorte inferior y = 0,7 | 0,769 | 0,936 |
| Una mano ausente por tramos | 0,837 | 0,959 |
| «Webcam-like» (una mano ausente + recorte) | 0,655 | 0,905 |
| **Media de los 8 escenarios** | 0,773 | 0,905 |

Estas perturbaciones son del mismo tipo que las del entrenamiento y se usaron también para elegir: miden robustez a esos fallos, **no**
generalización a la webcam.

**Test de v2** (131 procesables). Fuente: `reports/models_v2/test_evaluation.json`.

| | v1 (`transformer_aug`) | **v2 (`bilstm_robust`)** |
|---|---|---|
| Desplegado (semilla 17): exactitud / F1 macro | 0,870 / 0,872 | **0,931 / 0,931** |
| Desempeño integral (desplegado) | 0,814 | 0,871 |
| 3 semillas, exactitud (media ± desv.) | 0,878 ± 0,008 | **0,921 ± 0,009** |
| Umbral → cobertura / exactitud entre aceptados | 0,940 → 0,809 / 0,962 | 0,908 → 0,794 / 0,971 |
| Metas (exactitud ≥ 0,85; F1 ≥ 0,80) | cumplidas | cumplidas |

Comparación pareada (`reports/models_v2/paired_test_comparison.json`, mismas 131 secuencias): v1 acierta 114 y v2 122; solo v2 acierta en 9,
solo v1 en 1. Diferencia de exactitud +0,061 (IC95 % bootstrap +0,015 a +0,107; McNemar exacto p = 0,021).

**F1 por clase en test** (desplegados; `per_class_test_deployed.csv` de cada carpeta):

| Seña | v1 | v2 | n |
|---|---|---|---|
| Hola | 0,846 | 0,929 | 14 |
| Buenos | 1,000 | 0,963 | 13 |
| Días | 0,963 | 0,963 | 14 |
| Tardes | 0,842 | 0,900 | 10 |
| Noches | 0,923 | 0,960 | 12 |
| Yo | 0,690 | 0,929 | 14 |
| Nombre | 0,963 | 0,960 | 13 |
| Años | 0,963 | 0,929 | 14 |
| Gustar | 0,714 | 0,846 | 14 |
| Licor | 0,815 | 0,929 | 13 |

## 5. Discusión y límites

- **Errores por clase.** En v1, las clases más débiles en test fueron «Yo» (F1 0,690), «Gustar» (0,714), «Licor» (0,815), «Tardes» (0,842) y
  «Hola» (0,846); v2 las mejora (Yo 0,929, Gustar 0,846, Licor 0,929) pero empeora ligeramente «Buenos» (1,000 → 0,963) y «Años» (0,963 → 0,929).
  Con 10–14 secuencias por clase en test, una secuencia mueve el F1 de una clase varios puntos: **no se interpretan diferencias por clase**.
- **Validación frente a test.** v1: 0,915 → 0,878 (media de semillas); v2: 0,942 → 0,921. La validación (7 personas, 63 procesables) es
  pequeña; sobreestima ligeramente en ambos casos.
- **Selección débil en v1.** Ganó por un empate técnico decidido por una diferencia de latencia de menos de 2 ms; la regla se cumplió tal
  como estaba fijada, pero la elección no estaba respaldada por una diferencia de exactitud.
- **Resultado negativo.** `transformer_robust` no mejoró en validación original (0,910 frente a 0,915 de `transformer_aug`); la ganancia de v2
  proviene del BiLSTM con la aumentación robusta. La latencia del modelo de v2 es mayor (p95 52,7 ms frente a 37,4 ms), muy por debajo de la meta.
- **Tamaño de muestra.** Test: 131 procesables (una secuencia = 0,0076). La semilla desplegada de v2 (17) fue la mejor de sus tres (0,931 frente a
  0,916) y la de v1 la peor de las suyas; la comparación por media de semillas (0,921 frente a 0,878) es la más representativa.
- **Desempeño integral.** Cuando las capturas rechazadas por calidad cuentan como error, la exactitud baja a 0,814 (v1) y 0,871 (v2): ≈ 6 % de
  las secuencias de test (9 de 140) no son procesables.
- **Latencia.** El p95 del modelo solo (ms) está en las tablas. La meta de p95 < 2 s es de extremo a extremo y **[PENDIENTE]** su medición con
  la app real (`reports/app/`). Una medición local de servidor en 114 intentos de webcam dio p95 de ≈ 0,40 s en la extracción y ≈ 0,04 s en
  la predicción (capturas no publicadas, no reproducible desde el repositorio).

### 5.1 Lo que NO se midió
- **Rechazo de negativos** (gestos o personas fuera del vocabulario): no medido. El umbral es **provisional** y la interfaz presenta una
  **sugerencia que la persona confirma**; la probabilidad softmax no es certeza.
- **Exactitud etiquetada en webcam**: no se anotó el orden de la sesión, así que solo hay aceptación y confianza, no aciertos. Con las mismas 103
  entradas válidas de esa sesión (capturas locales no publicadas: no reproducible desde el repositorio), v1 acepta el 60 % y v2 el 72 % (v1 con
  el umbral de v2, 65 %); «aceptado» no es «acertado». Para una futura sesión: anotar el orden de las señas antes de empezar.
- **Personas ajenas al corpus**: ninguna evaluación externa salvo la sesión de una persona.
- **Generalización a otras cámaras, distancias o fondos.** Toda la sesión de webcam fue a una distancia (hombros/ancho ≈ 0,39).

### 5.2 Declaraciones obligatorias
1. **El split de test se usó DOS veces**: primero con v1 y después con v2, por decisión explícita del integrador (constancia en
   `reports/models_v2/test_authorization.json`). v2 se diseñó **conociendo** el resultado de test de v1; ninguna decisión de v2 se basó en datos de
   test, pero el test **dejó de ser una evaluación única**. Se informan ambos resultados.
2. v2 se adoptó **después** del congelamiento (previsto el 18 nov.; v1 se congeló el 19 nov., declarado) y de la evaluación de v1.
3. Los rangos de la aumentación de v2 se fijaron con apoyo de capturas de webcam de una sola persona (no usadas para entrenar ni elegir).
4. **[PENDIENTE]** Declaración de uso de IA generativa en el desarrollo, según las normas de la entrega (la hace el equipo).

## 6. Reproducibilidad
`prepare-data` → `sync-bundle-spec` → `run-all` → `aggregate` → `sensitivity` → `freeze` → `export-artifacts` → `final-eval` (v1) y, para v2,
`run-all/aggregate --config configs/train/experiments_v2.yaml --out reports/models_v2` → `robustness` → `export-artifacts --reports reports/models_v2
--out artifacts_v2` → `authorize-test` → `freeze` → `final-eval` → `compare-test` (ver `docs/models/v2_robusto.md`). Todos con
`python -m src.lsc_models <comando>`.
