# Sección de datos del artículo (DO10, conteos reales del lote 2)

Esta sección documenta el corpus y el pipeline de datos del proyecto LSC. Los
conteos del corpus real y del paquete de características YA están medidos
(lote 2, FEAT-001/FEAT-003) y se reportan con su fuente. Las métricas del
MODELO (exactitud, matriz de confusión, etc.) NO se incluyen: son de Acosta y
quedan explícitamente pendientes (bloque DO10). No se inventan cifras
(design §24/§25). No se llama "traducción completa" a la clasificación aislada
de 10 señas.

## 1. Corpus

- **Nombre / procedencia.** LSC70W (Lengua de Señas Colombiana). DOI
  `10.17632/9ssyn8tff5.2`, versión 2 (Mendeley Data; registrado por `acquire`
  sobre `configs/data/source.yaml`). Los subconjuntos LSC70AN y LSC70ANH se
  mantienen separados; sólo se procesa el subset LSC70W declarado en la
  procedencia.
- **Clases.** 10 clases en orden canónico fijo (fuente: `contracts/labels.json`):
  `0 Hola, 1 Buenos, 2 Días, 3 Tardes, 4 Noches, 5 Yo, 6 Nombre, 7 Años,
  8 Gustar, 9 Licor`.
- **Conteos crudos (fuente: `reports/data/inventory.csv`).**
  - Personas (firmantes `Per01..Per39`): **39**.
  - Ejecuciones (carpetas de clase por persona, `n_subdirs` del inventario):
    **388** (37 personas × 10 clases + Per39 con 8 clases: faltan TARDES y YO).
  - Imágenes `.jpg` del corpus: **2323** (suma de `n_files` del inventario;
    Per01..Per38 con 60 cada una y Per39 con 43).

## 2. Manifiesto: muestras válidas y exclusiones

Fuente: `data/interim/manifest.jsonl` y `data/interim/exclusions.csv`
(comando `python -m src.lsc_data manifest ...`).

- **Muestras válidas (secuencias de 6 cuadros): 387.**
- **Excluidas: 1.** `lsc70_v2/Per39/6/0` (Per39 / NOMBRE, `class_id` 6), con
  `reason_code` **`INVALID_FRAME_COUNT`** ("se esperaban 6 frames, hay 1"). Es la
  única fila de `exclusions.csv`.
- **Cobertura por clase (secuencias válidas en el manifiesto)**, fuente
  `data/interim/manifest.jsonl`:

  | class_id | seña    | válidas |
  | --- | --- | --- |
  | 0 | Hola   | 39 |
  | 1 | Buenos | 39 |
  | 2 | Días   | 39 |
  | 3 | Tardes | 38 |
  | 4 | Noches | 39 |
  | 5 | Yo     | 38 |
  | 6 | Nombre | 38 |
  | 7 | Años   | 39 |
  | 8 | Gustar | 39 |
  | 9 | Licor  | 39 |

  Las clases 3 (Tardes) y 5 (Yo) quedan en 38 porque **Per39** no las tiene; la
  clase 6 (Nombre) queda en 38 por la exclusión de `Per39/NOMBRE` (1 cuadro).

## 3. Splits sin fuga de identidad

Fuente: `data/interim/splits.json` (comando
`python -m src.lsc_data split --seed 20261022`).

- **Semilla:** `20261022`.
- **División por `signer_id` (personas): train/val/test = 27 / 4 / 8**
  (suma = 39 personas). Ninguna persona aparece en más de un conjunto.
- **Incidentes de cobertura registrados:** 0 (la ausencia de TARDES/YO en Per39
  queda en `class_coverage`, no genera incidente de fuga; la semilla NO se
  recomputa).
- Controles de intersección y de copias cruzadas por `sequence_sha256` activos;
  el comando devuelve exit 3 ante cualquier fuga (no ocurrió).

## 4. Paquete de características (bundle v1)

Fuente: `artifacts/data/v1/features.npz` + `feature_spec.json` + `exclusions.csv`
(comando `python -m src.lsc_data features ...`, validado con
`python -m src.lsc_data validate --bundle artifacts/data/v1` → exit 0).

- **Representación.** Cada muestra es una secuencia de 6 instantes; por instante
  se conservan 49 puntos (7 de pose + 21 mano izquierda + 21 mano derecha), cada
  uno como triple `(x, y, m)`. Forma final `(6, 147)`, `dtype=float32`.
- **Normalización por secuencia.** Centro = mediana de los puntos medios de
  hombros; escala = mediana de las distancias de hombros (comunes a los 6
  cuadros). El contrato exacto (orden de puntos, versiones, hash) vive en
  `contracts/feature_spec.json`; el `feature_spec_hash` identifica la versión.
- **Detector real.** MediaPipe Tasks (modo IMAGE), `.task` con procedencia
  verificada por SHA-256 (pose `4eaa5eb7…c4ad`, hands `fbc2a300…607cde`);
  registrada en `contracts/feature_spec.json` y `configs/data/features_v1.yaml`.
- **Conteos del bundle (fuente: `features.npz`).**
  - **N = 387** muestras (coincide con las válidas del manifiesto).
  - **Reparto por split: train 267 / val 40 / test 80** (muestras, no personas).
  - **`quality_pass = True`: 371** muestras; **`quality_pass = False`: 16**.
  - Rechazos de calidad por clase (`quality_pass = False`):
    `{0 Hola: 2, 1 Buenos: 1, 2 Días: 1, 3 Tardes: 5, 4 Noches: 4, 5 Yo: 1,
    9 Licor: 2}`. Un `quality_pass=False` no elimina la muestra del bundle; la
    marca para que el clasificador no la use.

## 5. Criterio de calidad

Una muestra pasa (`quality_pass = True`) si tiene 6 cuadros decodificables, a lo
sumo una persona por cuadro, hombros válidos en ≥5 cuadros y ≥1 mano asociada en
≥4 cuadros; en caso contrario se marca `quality_pass = False` con `reason_codes`
del conjunto canónico de 9 códigos (idéntico en Ortiz y en la API; ver
`tests/contracts/test_feature_result_parity.py`).

## 6. Augmentation

Política `train_v1` sólo en entrenamiento: rotación global por secuencia
(±5°, prob 0.5) y ruido gaussiano (std 0.005, prob 0.5), aplicados sólo donde
`m = 1`. Validación, prueba y producción NUNCA aplican augmentation.

## 7. Reproducibilidad y ética

- Ejecución determinista: desde cero produce los mismos `sample_id`, etiquetas y
  splits (semilla `20261022`). Empaquetado tipado sin pickle.
- No se versionan copias de consentimiento; `consent_ref` es un id opaco. Las
  rutas del manifiesto no contienen nombres personales (design §7/§25).

## 8. Alcance e incidentes

- Lo medido es **clasificación de 10 señas aisladas** de LSC70W, no traducción
  de lengua de señas ni reconocimiento continuo.
- Incidente del corpus: `Per39` no aporta TARDES ni YO, y su `NOMBRE` tiene un
  solo cuadro (excluido). Esto reduce tres clases a 38 válidas; se documenta en
  lugar de rellenar o recortar.

## 9. Pendientes (bloques)

- **DO10 (Acosta).** Métricas del MODELO (exactitud, matriz de confusión,
  calibración) para completar la sección de resultados del artículo. Hoy sin
  métricas inventadas.
- **DO09.** Captura externa y consentimientos (protocolo de prueba física).
- **DO11.** Fusión de `testpaths` al `pyproject.toml`/CI raíz de Acosta.

Ver `docs/data/blocks.md` para las condiciones de desbloqueo exactas y
`reports/data/do-batch2-report.md` para el resumen del lote 2.
