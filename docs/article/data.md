# Sección de datos del artículo

Esta sección documenta el corpus y el pipeline de datos del proyecto LSC. Todos los conteos están **medidos sobre el corpus completo
(LSC70W, 70 personas)** y cada uno cita su archivo fuente. Las métricas del modelo no van aquí: están en `docs/article/modeling.md`.
No se llama «traducción completa» a la clasificación aislada de 10 señas.

> Actualizado el 20 nov. 2026. Una versión anterior de este documento describía solo las primeras 39 personas del corpus (387 muestras;
> reparto 27/4/8); ya no aplica. Las cifras de abajo son las del corpus completo con el que se entrenaron y evaluaron v1 y v2.

## 1. Corpus

- **Nombre / procedencia.** LSC70W (Lengua de Señas Colombiana). DOI `10.17632/9ssyn8tff5.2`, versión 2 (Mendeley Data; registrado sobre
  `configs/data/source.yaml`). Los subconjuntos LSC70AN y LSC70ANH (letras y números) se mantienen separados y **no se usan**: solo se procesa
  el subconjunto LSC70W declarado en la procedencia.
- **Clases.** 10 clases en orden canónico fijo (fuente: `contracts/labels.json`):
  `0 Hola, 1 Buenos, 2 Días, 3 Tardes, 4 Noches, 5 Yo, 6 Nombre, 7 Años, 8 Gustar, 9 Licor`.
- **Conteos crudos (fuente: `reports/data/inventory.csv`, 70 filas, una por persona).**
  - Personas (`Per01..Per70`): **70**.
  - Ejecuciones (carpetas de clase por persona, `n_subdirs`): **700** (70 personas × 10 clases; todas las personas tienen las 10 clases).
  - Imágenes `.jpg`: **4 200** (suma de `n_files`: 60 por persona = 10 clases × 6 cuadros).

## 2. Manifiesto: muestras válidas y exclusiones

Fuente: `data/interim/manifest.jsonl` y `data/interim/exclusions.csv` (comando `python -m src.lsc_data manifest ...`).

- **Muestras válidas (secuencias de 6 cuadros): 700** (700 filas del manifiesto; 70 firmantes distintos).
- **Excluidas: 0.** `exclusions.csv` contiene solo la cabecera.
- **Cobertura por clase:** **70 secuencias en cada una de las 10 clases** (una por persona).

## 3. Splits sin fuga de identidad

Fuente: `data/interim/splits.json` (comando `python -m src.lsc_data split --seed 20261022`).

- **Semilla:** `20261022`.
- **División por `signer_id` (personas):** train / val / test = **49 / 7 / 14** personas (suma 70). Ninguna persona aparece en más de un conjunto.
- **Muestras por conjunto (fuente: `features.npz`):** train **490**, val **70**, test **140**.
- **Incidentes de cobertura registrados:** **0** (`incidents` vacío en `splits.json`).
- Controles de intersección y de copias cruzadas por `sequence_sha256` activos; el comando devuelve exit 3 ante cualquier fuga (no ocurrió).

## 4. Paquete de características (bundle v1)

Fuente: `artifacts/data/v1/features.npz` + `feature_spec.json` + `exclusions.csv` (comando `python -m src.lsc_data features ...`, validado con
`python -m src.lsc_data validate --bundle artifacts/data/v1`).

- **Representación.** Cada muestra es una secuencia de 6 instantes; por instante se conservan 49 puntos (7 de pose + 21 mano izquierda + 21 mano
  derecha), cada uno como triple `(x, y, m)`. Forma final `(6, 147)`, `dtype=float32`.
- **Normalización por secuencia.** Centro = mediana de los puntos medios de hombros; escala = mediana de las distancias de hombros (comunes a los
  6 cuadros). El contrato exacto (orden de puntos, versiones, hash) vive en `contracts/feature_spec.json`; el `feature_spec_hash` identifica la versión.
- **Detector real.** MediaPipe Tasks (modo IMAGE), `.task` con procedencia verificada por SHA-256 (pose `4eaa5eb7…c4ad`, hands `fbc2a300…607cde`);
  registrada en `contracts/feature_spec.json` y `configs/data/features_v1.yaml`.
- **Conteos del bundle (fuente: `features.npz`).**
  - **N = 700** muestras (coincide con las válidas del manifiesto).
  - **`quality_pass = True`: 656** (train 462, val 63, test 131); **`quality_pass = False`: 44**. Una muestra con `quality_pass=False` no se
    elimina del bundle: se marca para que el clasificador no la use.
  - **Rechazos de calidad por clase** (suman 44): Hola 3, Buenos 4, Días 5, **Tardes 14**, **Noches 11**, Yo 2, Nombre 1, Años 1, Gustar 0, Licor 3.
    «Tardes» y «Noches» concentran 25 de los 44 rechazos (57 %, cuenta derivada de la lista anterior).

## 5. Criterio de calidad

Una muestra pasa (`quality_pass = True`) si tiene 6 cuadros decodificables, a lo sumo una persona por cuadro, hombros válidos en ≥ 5 cuadros y ≥ 1
mano asociada en ≥ 4 cuadros; en caso contrario se marca `quality_pass = False` con `reason_codes` del conjunto canónico de 9 códigos (idéntico en
Ortiz y en la API; ver `tests/contracts/test_feature_result_parity.py`). La app aplica la misma regla a las capturas en vivo.

## 6. Aumentación

- **v1, política `train_v1`** (solo entrenamiento): rotación global por secuencia (±5°, prob. 0,5) y ruido gaussiano (σ = 0,005, prob. 0,5), aplicados
  solo donde `m = 1`.
- **v2, política `train_v2_robust`** (solo entrenamiento): recorte inferior y lateral que quita la mano completa, una mano ausente toda la secuencia,
  pérdidas por cuadro, escala/traslación/rotación leves, ruido y repetición de un cuadro; lo aumentado sigue siendo procesable. Ver
  `docs/models/v2_robusto.md`.

Validación, prueba y producción **nunca** aplican aumentación.

## 7. Reproducibilidad y ética

- Ejecución determinista: desde cero produce los mismos `sample_id`, etiquetas y splits (semilla `20261022`). Empaquetado tipado sin pickle.
- No se versionan copias de consentimiento; `consent_ref` es un id opaco. Las rutas del manifiesto no contienen nombres personales.
- El corpus (`LSC70/`, imágenes de personas) y `artifacts/data` **no** se suben a Git.
- Las capturas de webcam del diagnóstico de v2 (`artifacts/debug_dump*/`) contienen el rostro de la persona que las hizo: se guardan solo en local,
  están ignoradas por Git y no se publican.

## 8. Alcance y límites

- Lo medido es **clasificación de 10 señas aisladas** de LSC70W, no traducción de lengua de señas ni reconocimiento continuo.
- **Desfase con la webcam.** El corpus muestra casi siempre ambas manos y a la persona a una distancia (hombros ≈ 0,20 del ancho de imagen); en una
  sesión de webcam de una persona se midió hombros ≈ 0,37 y una sola mano en la mayoría de los cuadros. Detalle y consecuencias en
  `docs/article/modeling.md` §3.
- Las 70 personas pertenecen a un único corpus con el mismo formato de imagen (640×480 en las muestras revisadas): no hay personas ajenas al corpus
  ni otras cámaras en la evaluación.

## 9. Pendientes

- **Captura externa y consentimientos** (protocolo de prueba física con personas ajenas al corpus): no realizada. Sin ella no hay evaluación externa
  y debe declararse como limitación.
- **Rechazo de negativos:** no medido (requiere capturas de gestos o personas fuera del vocabulario).

Ver `docs/data/blocks.md` para las condiciones de desbloqueo y `reports/data/do-batch2-report.md` para el resumen del lote 2
(**nota:** ese informe y `docs/data/corpus-grammar-assumption.md` conservan cifras y una regex de la etapa de 39 personas y están desactualizados).
