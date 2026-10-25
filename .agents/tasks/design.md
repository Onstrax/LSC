# Diseño — Módulo de datos y extracción de características (Ortiz, 30 %)

Estado: revisión v3 (incorpora la segunda ronda de `design-review.json`,
veredicto `CHANGES_REQUESTED`; se resuelven sus 5 hallazgos — 1 HIGH, 1 MEDIUM,
3 NIT — contra el código real del repo, verificando `pytest.ini`,
`api/constants.py` y `api/adapters/*`). La ronda previa resolvió sus 8 hallazgos
(ver historial en §26). El detalle de respuesta por hallazgo está en §26.
Alcance de propiedad: sólo rutas de Ortiz según `AGENTS.md`. No se propone ni
un solo cambio en `app/`, `api/`, `deployment/` (salvo el archivo nuevo
`deployment/requirements-data.in`), `tests/app/`, `tests/e2e/`, `docs/app/`,
`pyproject.toml`, `pyproject.app.toml`, lockfile ni CI raíz. `api/schemas/openapi.json`
y `api/adapters/types.py` se usan como referencia de SÓLO LECTURA para alinear
el contrato.

---

## 1. Resumen

Este documento especifica, al nivel necesario para implementar sin tomar nuevas
decisiones de arquitectura, el primer lote ejecutable del módulo de Ortiz:
contratos de datos (`contracts/`), empaquetado y manifiesto del corpus
(`src/lsc_data/`), la matemática pura del extractor con detector inyectable
(`src/lsc_features/`), configuración versionada (`configs/data/`), la CLI
`python -m lsc_data`, la suite de pruebas rápida sin red (`tests/data`,
`tests/features`, `tests/contracts`) y las dependencias mínimas que se entregan
a Acosta (`deployment/requirements-data.in`).

El lote corre **sin red**, **sin el corpus real de Mendeley** y **sin archivos
`.task` reales de MediaPipe**. Toda la geometría se valida con detectores falsos
inyectables y fixtures sintéticos rotulados. Las tareas fuera de alcance (DO02,
DO05, DO07–DO10) se documentan como bloques con instrucciones exactas de
desbloqueo, pero no se implementan.

El diseño alinea tres invariantes contractuales que la API ya fija y que Ortiz
debe respetar byte a byte: forma `(6, 147)` (`api/constants.py`:
`SEQUENCE_SHAPE`), el conjunto exacto de `QUALITY_REASON_CODES`, y la firma
`extract_sequence(frames_rgb) -> FeatureResult` del `ExtractorProtocol`
(`api/adapters/types.py`). El `FeatureResult` de Ortiz debe ser estructuralmente
compatible con el dataclass que la API ya consume (campos `x`, `quality_pass`,
`reason_codes`, `valid_shoulders`, `frames_with_hand`, `feature_spec_hash`,
`elapsed_ms`).

Nota sobre un campo del contrato existente: en `api/adapters/types.py`
`valid_shoulders` está tipado como `bool`, mientras que la especificación de
Ortiz pide `valid_shoulders: int` (conteo de cuadros con hombros válidos). Se
resuelve así: el `FeatureResult` de `src/lsc_features` expone
`valid_shoulders: int`. La API lo consume por verdad/falsedad (`bool(count)`),
lo que es compatible sin tocar `api/`. Se registra como bloque de integración
DO08 (paridad con la API de Juan), a resolver cuando se unifique el contrato;
Ortiz NO modifica `api/`.

---

## 2. Stack tecnológico (fijo tras aprobación)

- Python 3.11, ejecución en CPU.
- numpy (álgebra y arrays empaquetados `.npz`).
- opencv-python-headless (decodificación de imágenes; sin GUI, apto para CPU/CI).
- mediapipe (Tasks; sólo en el camino real NO ejercitado en este lote).
- pydantic v2 (tipos de contrato y validación de configuración/manifiesto).
- jsonschema (validación de documentos contra JSON Schema en `validate`).
- PyYAML (lectura de configs YAML versionadas).
- pytest (suite rápida sin red).

Justificación de elección donde había alternativas:
- **opencv-python-headless** en vez de Pillow: el corpus real y los `.task`
  trabajan en espacio BGR/RGB con matrices numpy; headless evita dependencias de
  GUI y es estable en CI sin display. Se documenta que el extractor recibe RGB
  uint8 (la conversión BGR→RGB, si aplica, ocurre en el adaptador de adquisición,
  nunca dentro de la matemática pura).
- **pydantic v2** en vez de dataclasses puras: la API ya usa pydantic v2
  (`api/schemas/models.py`); mantener el mismo motor evita un segundo estilo de
  validación y facilita que Acosta reutilice los tipos. El `FeatureResult`
  concreto se mantiene como `@dataclass(frozen=True)` para ser idéntico al que la
  API ya importa, pero los tipos de manifiesto/split/reportes son modelos
  pydantic.
- **jsonschema** para `validate`: los esquemas se publican como JSON Schema en
  `contracts/schemas/` (consumibles por cualquier lenguaje), y pydantic valida en
  Python. Los dos derivan del mismo contrato; una prueba de `tests/contracts`
  comprueba que no divergen.

`deployment/requirements-data.in` (archivo NUEVO, único permitido bajo
`deployment/`) lista estas dependencias con versiones ancladas y se entrega a
Acosta como SOLICITUD de fusión a su lockfile raíz (que hoy no existe en el repo;
ver §22). Ortiz NO toca `pyproject.toml` (inexistente, de Acosta),
`pyproject.app.toml` (de Juan) ni `requirements-app.in`.

---

## 3. Árbol de archivos exacto (sólo rutas de Ortiz)

```
contracts/
  __init__.py
  feature_spec.json            # orden de puntos, forma, dtype, versión, hash canónico
  labels.json                  # 10 clases: class_id, code, display_text, raw_label
  models.py                    # tipos pydantic v2 del contrato (manifest, split, reports, specs)
  canonical.py                 # serialización canónica + SHA-256 (fuente única de hashing)
  schemas/
    manifest.schema.json       # JSON Schema de una línea de manifiesto
    split.schema.json          # JSON Schema de splits.json
    report.schema.json         # JSON Schema de ManifestReport/SplitReport/FeatureBuildReport
    feature_spec.schema.json   # JSON Schema de feature_spec.json
    labels.schema.json         # JSON Schema de labels.json
  fixtures/
    README.md                  # declara que TODO aquí es SINTÉTICO
    SYNTHETIC_valid_sample.jsonl     # manifiesto válido sintético
    SYNTHETIC_invalid_samples.jsonl  # casos inválidos rotulados por reason_code
    SYNTHETIC_splits.json
    SYNTHETIC_feature_result_6x147.npz  # x (6,147) consumible sin corpus
src/
  lsc_data/
    __init__.py                # exporta build_manifest, make_splits, API pública
    __main__.py                # CLI: python -m lsc_data <subcomando>
    cli.py                     # parser y despacho; mapeo de códigos de salida 0/2/3
    acquisition.py             # acquire sin red; verificación de archivo; source.json
    security.py                # extracción segura (temp dir, rechazos de rutas)
    inventory.py               # inventory -> reports/data/inventory.csv
    adapter_lsc70w.py          # gramática carpeta/nombre parametrizable por regex
    manifest.py                # build_manifest -> ManifestReport
    sequence.py                # reconstrucción de secuencias (agrupar/ordenar/validar)
    hashing.py                 # re-export de contracts.canonical para datos (sample_id, sequence_sha256)
    split.py                   # make_splits con rng(20261022); controles de fuga
    packaging.py               # features.npz, feature_spec.json, exclusions.csv
    validate.py                # validate --bundle (schema/formas/integridad/cobertura/fuga)
    errors.py                  # excepciones tipadas internas de datos
  lsc_features/
    __init__.py                # exporta FeatureExtractor, FeatureResult, build_features, augment_train
    types.py                   # FeatureResult (dataclass frozen idéntico a api/adapters/types)
    detector.py                # DetectorProtocol + FakeDetector + tipos de landmarks
    geometry.py                # px, mediana centro/escala, normalización, rotación
    association.py             # asociación mano-pose de costo mínimo
    quality.py                 # reglas de calidad, orden determinista de reason_codes
    temporal.py                # muestreo round_half_up de índices
    augment.py                 # augment_train (train_v1): rotación + ruido
    extractor.py               # FeatureExtractor(config, model_paths) con detector inyectable
    build.py                   # build_features(manifest, splits, extractor, out_dir)
    reason_codes.py            # enum/constantes de códigos (espejo de api/constants)
configs/
  data/
    source.yaml                # procedencia (URL, DOI, versión, hashes esperados)
    lsc70w.yaml                # gramática regex del corpus LSC70W + label_map
    features_v1.yaml           # umbrales, versiones, política de augment, detector cfg
tests/
  data/
    __init__.py
    conftest.py
    test_manifest_D.py         # D01..D12
    test_split_leakage.py      # refuerzos de fuga de D
    test_acquisition_security.py
    test_cli_exit_codes.py
  features/
    __init__.py
    conftest.py                # fixtures de FakeDetector y geometría sintética
    test_features_F.py         # F01..F12
  contracts/
    __init__.py
    test_feature_spec_hash.py  # hash canónico estable ante rename/fecha
    test_schema_pydantic_parity.py
    test_labels_canonical_order.py
    test_incompatible_config.py  # labels/hash divergentes -> fallo al inicio
    test_model_paths_contract.py # claves exactas {"pose","hands"}; config dict (§8.3, hallazgo 1)
    test_requirements_version.py # compatibilidad básica de versiones (sin pyproject raíz)
reports/
  data/
    .gitkeep                   # inventory.csv y reportes se generan aquí (algunos git-excluidos)
    README.md
docs/
  data/
    README.md
    progreso-DO.md             # bitácora por tarea DO (id, cambio, archivo, verificación, conteos, riesgo)
    corpus-grammar-assumption.md  # supuesto de gramática LSC70W a verificar contra el corpus real
    blocks.md                  # bloques fuera de alcance con instrucciones exactas
  article/
    data.md                    # sección de datos del artículo (estructura, sin métricas reales aún)
deployment/
  requirements-data.in         # ÚNICO archivo nuevo permitido bajo deployment/
.gitignore                     # data/ y artifacts/* YA ignorados; a lo sumo un comentario (ver §21)
```

Datos pesados viven en `data/raw/`, `data/interim/`, `data/processed/`,
`artifacts/data/` y quedan excluidos de git. `contracts/`, los fixtures
sintéticos y el código NO se excluyen.

---

## 4. Contratos y tipos pydantic (`contracts/models.py`)

Todos los modelos usan pydantic v2 con `model_config = ConfigDict(extra="forbid",
frozen=True)` salvo donde se indique mutabilidad. Los nombres de campo son
exactamente los de la especificación.

### 4.1 Etiquetas

`LabelEntry`: `class_id: int` (0..9), `code: str` (ascii sin tildes:
`hola, buenos, dias, tardes, noches, yo, nombre, anos, gustar, licor`),
`display_text: str` (con tildes: `Hola, Buenos, Días, ...`), `raw_label: str`
(etiqueta de origen sin alterar).

`LabelMap`: `schema_version: str`, `labels: list[LabelEntry]` de longitud 10 en
ORDEN CANÓNICO FIJO. Validadores: `class_id` consecutivos 0..9 en orden; el
orden NO se infiere de `os.listdir` ni alfabéticamente. `contracts/labels.json`
es el artefacto serializado de este modelo y es la fuente única del módulo de
datos (el `labels.json` de Acosta es la fuente para el modelo; una prueba de
contrato verifica que `code`/`class_id` coinciden cuando el de Acosta exista —
hoy bloque DO08).

### 4.2 Especificación de características

`FeatureSpec`: `schema_version`, `feature_version`, `point_order: list[str]`
(49 nombres en orden: `pose_0, pose_11, pose_12, pose_13, pose_14, pose_15,
pose_16, lh_0..lh_20, rh_0..rh_20`), `coordinate_system: str`,
`temporal_length: int = 6`, `features_per_frame: int = 147`,
`dtype: str = "float32"`, `missing_policy: dict`, `quality_rules: dict`,
`detector_versions: dict`, `model_asset_hashes: dict`, `normalization: dict`.
El `feature_spec_hash` NO se almacena dentro del objeto cuya firma se calcula
(ver §6); la fecha de construcción se guarda en un campo separado
`build_date` que NO entra al hash semántico.

### 4.3 Manifiesto (una línea JSON por muestra)

`ManifestRecord` (campos requeridos):
`schema_version, sample_id, source, source_version, signer_id, class_id,
raw_label, repetition_id, frame_paths: list[str] (6), frame_sha256: list[str]
(6), frame_indices: list[int] (6), width, height, n_frames, sequence_sha256,
structural_status`.

Validadores: `len(frame_paths)==len(frame_sha256)==len(frame_indices)==6`;
`frame_indices` únicos; rutas relativas a `data/raw/` (rechaza absolutas, `..`,
nombres personales — ver §7); `class_id` en 0..9 **o** `null` para negativos;
`n_frames == 6` para muestras válidas.

Campos opcionales para capturas propias:
`domain, participant_group, experience_lsc` (enum `fluent|interpreter|
instructed|unknown`), `capture_session, capture_protocol_version, consent_ref`
(id opaco, NUNCA la copia del consentimiento), `timestamps_ms: list[int]`,
`sample_type` (enum `known|rest|gesture|out_of_vocab`).

`NegativeRecord`: igual que `ManifestRecord` pero `class_id` es `null`; se
exporta en paquetes propios con `y=-1` y NUNCA entra al entrenamiento de 10
clases. El loader rechaza cualquier `y` fuera de 0..9.

`ExclusionRecord` (para `exclusions.csv`): `artifact_id, stage, reason_code,
detail`.

### 4.4 Splits

`SplitAssignment`: `train: list[str]`, `val: list[str]`, `test: list[str]`
(listas de `signer_id`), más `sample_ids` por split.
`SplitDoc`: `schema_version, seed: int, manifest_sha256: str, version: str,
signer_ids: SplitAssignment, sample_ids: {train,val,test}, class_coverage:
{split: {class_id: count}}, incidents: list[str]`. Es el artefacto `splits.json`.

### 4.5 Reportes

`ManifestReport`: `n_records, n_valid, n_excluded, by_class: dict[int,int],
by_signer: dict[str,int], exclusions_path, manifest_path, warnings`.
`SplitReport`: `n_train, n_val, n_test, n_signers, class_coverage, incidents,
splits_path`.
`FeatureBuildReport`: `n_total, n_quality_pass, n_failed, failed_sample_ids,
feature_spec_hash, bundle_path, per_reason_counts`.
`FeatureResult`: NO es pydantic; es el dataclass de §8.

---

## 5. JSON Schemas (`contracts/schemas/`)

Cada esquema es Draft 2020-12, `"additionalProperties": false`, con `required`
explícito. Se generan a mano (no auto-exportados) para ser idioma-agnósticos y
se mantienen sincronizados con los modelos pydantic mediante
`tests/contracts/test_schema_pydantic_parity.py`, que:

1. Construye instancias válidas e inválidas desde los modelos pydantic.
2. Valida cada una contra el JSON Schema correspondiente con `jsonschema`.
3. Exige que pydantic y jsonschema acepten/rechacen el MISMO conjunto.

- `manifest.schema.json`: una línea = un `ManifestRecord`. `class_id` admite
  `integer` 0..9 o `null`. Arrays de longitud fija 6 (`minItems/maxItems`).
- `split.schema.json`: estructura de `SplitDoc`; `seed` const 20261022.
- `report.schema.json`: `oneOf` de los tres reportes vía `report_type`.
- `feature_spec.schema.json`: campos de `FeatureSpec`; `features_per_frame`
  const 147, `temporal_length` const 6, `dtype` const `"float32"`.
- `labels.schema.json`: array de 10 `LabelEntry` con `class_id` 0..9.

---

## 6. Serialización canónica y SHA-256 (`contracts/canonical.py`)

Fuente ÚNICA de hashing para todo el módulo. Regla byte a byte, reproducible:

```python
def canonical_bytes(obj) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,      # UTF-8 real, tildes incluidas
        sort_keys=True,          # claves ordenadas
        separators=(",", ":"),   # sin espacios
        allow_nan=False,         # prohíbe NaN/Infinity
    ).encode("utf-8")

def sha256_hex(obj) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()
```

Reglas adicionales:
- Los `float` que entren al hash se serializan con `repr` determinista; para los
  campos del manifiesto evitamos floats en la representación canónica (usamos
  enteros px, strings de hash y enteros de índice). El `sample_id` y
  `sequence_sha256` nunca dependen de floats.
- Antes de hashear se eliminan los campos NO semánticos: `build_date`,
  `elapsed_ms`, rutas absolutas, y el propio campo `*_hash`.

### 6.1 `feature_spec_hash`
SHA-256 del `FeatureSpec` canónico (claves ordenadas, UTF-8, separadores fijos),
**excluyendo** `build_date` y cualquier campo `feature_spec_hash`. Una prueba
(`test_feature_spec_hash.py`) renombra un campo cosmético/cambia la fecha y
exige que el hash semántico NO cambie, y cambia `point_order` y exige que SÍ
cambie.

### 6.2 `sample_id`
SHA-256 de la representación canónica de: `source_version, signer_id, class_id,
repetition_id` y la lista ORDENADA de hashes de frame (`frame_sha256`). Objeto
canónico:
```json
{"class_id":C,"frame_sha256":[...6...],"repetition_id":R,"signer_id":"...","source_version":"..."}
```

### 6.3 `sequence_sha256`
SHA-256 SOLO sobre la lista ordenada de hashes de imagen (`frame_sha256`), SIN
identidad ni etiqueta. Detecta copias de la misma secuencia entre personas
distintas (dos muestras con `sequence_sha256` igual y `signer_id` distinto = copia
cruzada -> incidente/exclusión).

### 6.4 `manifest_sha256`
SHA-256 sobre la concatenación canónica ordenada por `sample_id` de las líneas
del manifiesto; lo consume `splits.json` y `validate`.

---

## 7. Adquisición segura (`src/lsc_data/acquisition.py`, `security.py`)

`acquire --config configs/data/source.yaml --archive OPCIONAL`:
- Sin `--archive`: NO descarga. Registra en log el bloque DO02 y las
  instrucciones exactas (URL, DOI `10.17632/9ssyn8tff5.2`, versión) y termina
  con código 0 si sólo verifica, o 2 si el config es inválido. Nunca simula una
  descarga.
- Con `--archive`: valida procedencia contra `source.yaml` (SHA-256, tamaño
  esperados) y extrae de forma segura. Rutas peligrosas detectadas en la
  extracción (`..`, absolutas, symlinks que escapan, ejecutables inesperados) ->
  rechazo, nada se mueve a destino, **exit 3 (integridad)**. `source.yaml`
  inválido -> **exit 2**.

Extracción segura (`security.py`):
1. Extraer a un directorio temporal.
2. Rechazar: rutas absolutas, componentes `..`, symlinks que escapen del
   destino, y ejecutables inesperados (lista de extensiones permitidas en config).
3. Sólo tras validar, mover a destino final.
4. Mismo input (mismo hash) no se duplica; input distinto NO sobrescribe
   (requiere nueva versión de procedencia en `source.yaml`).
5. `source.json` registra URL, DOI, versión, fecha UTC, tamaño y SHA-256.
6. Subconjuntos LSC70AN y LSC70ANH se mantienen separados de LSC70W; el adapter
   sólo procesa el subset declarado en config.

Secretos/identidades: las rutas del manifiesto no contienen nombres personales;
`consent_ref` es un id opaco. El módulo nunca lee ni versiona copias de
consentimiento.

---

## 8. Extractor: detector inyectable y FeatureResult

### 8.1 `FeatureResult` (`src/lsc_features/types.py`)
Dataclass `frozen` idéntico en forma al de `api/adapters/types.py`:
`x: np.ndarray (6,147) float32`, `quality_pass: bool`, `reason_codes: list[str]`,
`valid_shoulders: int`, `frames_with_hand: int`, `feature_spec_hash: str`,
`elapsed_ms: float`.

**Invariante de salida de forma/dtype (hallazgo 4).** El tipo documental no basta:
el extractor GARANTIZA POR CONSTRUCCIÓN que, antes de devolver cualquier
`FeatureResult` (incluso cuando `quality_pass=False`), `x` cumple
`x.shape == (6, 147)` y `x.dtype == np.float32`. `extract_sequence` construye `x`
con `np.zeros((6,147), dtype=np.float32)` y sólo escribe triples normalizados en
posiciones válidas; nunca retorna un arreglo de otra forma o dtype ni delega esa
coacción al llamante. `api/adapters/real.py` además hace
`np.asarray(result.x, dtype=np.float32)` como defensa, pero la regla es de Ortiz,
no de la API. `F02` (§20) verifica la invariante en el camino con y sin
`quality_pass`.

### 8.2 `DetectorProtocol` y `FakeDetector` (`src/lsc_features/detector.py`)

El `FeatureExtractor` NO carga detectores ni descarga en el import. La inicialización
explícita verifica `.task` y hashes (camino real NO ejercitado en este lote).
Para pruebas, el extractor acepta un detector inyectado que espeja la API IMAGE
de MediaPipe Tasks:

```python
@runtime_checkable
class DetectorProtocol(Protocol):
    def detect_pose(self, image_rgb: np.ndarray) -> PoseResult: ...
    def detect_hands(self, image_rgb: np.ndarray) -> HandsResult: ...
    def close(self) -> None: ...
```

Tipos de resultado que imitan MediaPipe Tasks (modo IMAGE):
- `PoseLandmark`: `x: float, y: float` (normalizados 0..1), `visibility: float`,
  `presence: float | None` (campo opcional). La validez del punto se decide con
  la regla por defecto fijada en §11 (`quality_rules.use_presence=false`): la
  ausencia del campo nunca invalida; su presencia exige `presence>=0.5`.
- `PoseResult`: `pose_landmarks: list[list[PoseLandmark]]` (una lista por persona;
  `num_poses=2`).
- `HandLandmark`: `x: float, y: float`, más `handedness` (`"Left"`/`"Right"`
  — se IGNORA para asignación; se usa anatomía por pose, no lado de pantalla).
- `HandsResult`: `hand_landmarks: list[list[HandLandmark]]` (21 puntos por mano),
  `handedness: list[...]` (`num_hands=2`).

`FakeDetector` implementa el protocolo devolviendo landmarks programados por el
test (listas de personas/manos construidas en memoria). Permite reproducir:
cero personas, una persona, dos personas (MULTIPLE_PEOPLE), manos ambiguas,
puntos no finitos, hombros ausentes, etc. Un `RealDetector` (no ejercitado)
envolvería `mediapipe.tasks.python.vision.PoseLandmarker`/`HandLandmarker` con
`running_mode=IMAGE`, `num_poses=2`, `num_hands=2`, thresholds 0.5; su carga de
`.task` queda documentada como bloque DO05.

### 8.3 `FeatureExtractor` (`src/lsc_features/extractor.py`)

Firma EXACTA (fijada contra el contrato que ya escribió Juan en
`api/adapters/real.py`, sólo lectura):

```python
FeatureExtractor(
    config: Mapping[str, Any],
    model_paths: Mapping[str, str | Path],
    detector: DetectorProtocol | None = None,
)
```

Esta firma es obligatoria porque `api/adapters/real.py` ya construye el extractor
así (verificado en el código real):

```python
model_paths = {
    "pose":  _require(self._artifacts_dir / "pose_landmarker.task", "pose .task"),
    "hands": _require(self._artifacts_dir / "hand_landmarker.task", "hand .task"),
}
self._impl = FeatureExtractor(feature_config or self._feature_spec, model_paths)
```

Reglas derivadas de ese contrato, FIJAS:

- `config` acepta un **`dict`/`Mapping` de `feature_spec`** (NO sólo un objeto
  pydantic). `api/adapters/real.py` pasa `feature_spec.json` ya cargado con
  `json.load` como `dict`; el extractor debe aceptarlo directamente. Internamente
  puede validarlo contra `FeatureSpec` (pydantic, §4.2) sin exigir que el llamante
  construya el modelo.
- `model_paths` DEBE contener EXACTAMENTE las claves obligatorias `"pose"` y
  `"hands"`, cuyos valores son rutas (`str | Path`) a `pose_landmarker.task` y
  `hand_landmarker.task` respectivamente. Claves faltantes o adicionales ⇒
  `CONFIG_MISMATCH` en `init()`. No se admiten listas ni tuplas.
- Si `detector` se inyecta, se usa tal cual y `model_paths` NO se abre ni se
  hashea (camino de pruebas; no se toca disco).
- Si `detector` es `None`, la verificación de existencia+hash de las DOS entradas
  (`"pose"`, `"hands"`) ocurre en un método `init()` explícito, separado del
  import (no se carga nada en import; §16). Sólo se validan esas dos entradas.
  (Camino real NO ejercitado en este lote; bloque DO05.)
- `extract_sequence(frames_rgb)`: valida 6 cuadros RGB uint8 de igual tamaño,
  ejecuta detección por cuadro vía el detector, aplica §9–§12 y devuelve
  `FeatureResult`. Mide `elapsed_ms`.

Prueba de contrato asociada (`tests/contracts/test_model_paths_contract.py`):
afirma que las claves esperadas de `model_paths` son EXACTAMENTE `{"pose",
"hands"}` (espejo de `api/adapters/real.py`); que un `dict` de `feature_spec` es
aceptado como `config`; y que `init()` con una clave faltante o sobrante produce
`CONFIG_MISMATCH` sin tocar disco cuando se inyecta un `FakeDetector`. Esta prueba
no requiere `.task` real.

---

## 9. Reconstrucción de secuencias (`src/lsc_data/sequence.py`)

Agrupar por `(source, signer_id, class_id, repetition_id)`. Si el corpus no
tiene repeticiones, `repetition_id=0` (se documenta en `lsc70w.yaml` y en
`docs/data/corpus-grammar-assumption.md`). Ordenar numéricamente por
`frame_index`. Cada muestra válida = exactamente 6 índices únicos. Rechazar
(a `exclusions.csv`): faltantes, duplicados ambiguos, personas/clases mezcladas.
NUNCA completar con frames de otra ejecución ni replicar frames. Un signo
estático puede tener imágenes idénticas (no invalida). `sample_id` y
`sequence_sha256` según §6.

La gramática carpeta/nombre del corpus LSC70W se parametriza con una regex
nombrada en `lsc70w.yaml` (grupos `signer_id`, `class_label`, `repetition_id`,
`frame_index`). Como no hay ejemplos reales, el supuesto se documenta como bloque
a verificar contra el corpus real; el adapter falla con mensaje accionable si la
regex no casa, nunca adivina.

---

## 10. Splits sin fuga (`src/lsc_data/split.py`)

`make_splits(manifest_path, seed=20261022) -> SplitReport`:
1. Extraer `signer_id` únicos; ordenar de forma estable.
2. `rng = numpy.random.default_rng(20261022); perm = rng.permutation(ids)`.
3. Para N personas: `n_test = ceil(0.20*N)`,
   `n_val = max(1, int(floor(0.10*N + 0.5)))` (half-up determinista, consistente
   con el `round_half_up` del muestreo temporal de §15; NO usar el `round()`
   nativo de Python, que es half-to-even), `n_train = N - n_test - n_val`. Para
   N=70: `n_test=ceil(14.0)=14`, `n_val=max(1,floor(7.0+0.5))=7`, `n_train=49`
   (49/7/14). El half-up sólo cambia el resultado frente al nativo cuando
   `0.10*N` termina exactamente en `.5` (p. ej. N=25 ⇒ `floor(2.5+0.5)=3`).
4. Persistir la asignación concreta en `splits.json`; NO recomputar al consumir.
5. Controles: intersección vacía de identidades, de `sample_ids` y ausencia de
   duplicados exactos (vía `sequence_sha256`).
6. Cobertura: todas las 10 clases presentes por conjunto; si falta alguna, emitir
   incidente con conteos y NO cambiar la semilla repetidamente.
7. Escaladores (si los hubiera en v1) NO se ajustan con val/test.

`splits.json` guarda `signer_ids` y `sample_ids` por split, `seed`, `version` y
`manifest_sha256`.

---

## 11. Extracción y asociación mano-pose (`geometry.py`, `association.py`)

Modo IMAGE, `num_hands=2`, `num_poses=2`. Si se detecta más de una persona:
rechazar `MULTIPLE_PEOPLE`. Thresholds de detección/presencia 0.5.

Pose: conservar 7 índices en orden `0, 11, 12, 13, 14, 15, 16`. Luego 21 puntos
de mano izquierda y 21 de mano derecha (anatomía de la persona, no lado de
pantalla), cada mano ordenada 0..20. Total 7+21+21 = 49 por instante.

Asociación mano-pose (`association.py`). Para cada muñeca detectada por el Hand
Landmarker, el costo frente a una muñeca de pose es la distancia en píxeles entre
ambas, normalizada por la distancia de hombros del cuadro. El lado anatómico lo
fija la pose: pose-15 = mano IZQUIERDA de la persona, pose-16 = mano DERECHA. El
`handedness` de MediaPipe se IGNORA (no es lado de pantalla).

**Precondición de hombros por cuadro (hallazgo de revisión, MEDIUM).** La
normalización del costo usa SIEMPRE la distancia de hombros DEL PROPIO CUADRO;
nunca la escala de secuencia de §12.3. Si un cuadro NO tiene hombros válidos
(distancia de hombros `< 0.08*W`, o los puntos de pose 11/12 no son finitos / no
pasan la validez de §11), el denominador de normalización es indefinido: en ese
cuadro NO se asocia ninguna mano, el cuadro contribuye `0` a `frames_with_hand`,
y NO se emite `HAND_ASSOCIATION_AMBIGUOUS`. Esta precondición se evalúa ANTES de
las reglas por número de manos de abajo. Prueba asociada `F08d` (ver §20):
`FakeDetector` devuelve dos muñecas de mano en un cuadro cuyos hombros fallan el
mínimo; se verifica que ese cuadro no suma a `frames_with_hand` ni emite la
advertencia de ambigüedad.

Reglas explícitas por número de manos detectadas en el cuadro (FIJAS; aplican
sólo a cuadros con hombros válidos; el umbral `<=0.6` se aplica DENTRO de cada
asignación candidata, descartando enlaces con costo `>0.6` ANTES de comparar
costos totales):

- **0 manos detectadas.** No hay asociación posible; no aplica la regla de gap.
  El cuadro no aporta manos (contribuye 0 a `frames_with_hand`). No emite
  `HAND_ASSOCIATION_AMBIGUOUS`.
- **1 mano detectada.** Hay a lo sumo una asignación candidata por lado
  (izq=pose-15 o der=pose-16); se calcula el costo a cada muñeca de pose válida y
  se asigna al lado de MENOR costo **sólo si ese costo `<= 0.6`**. El gap es
  INDEFINIDO con una sola mano ⇒ NUNCA ambiguo (no se emite
  `HAND_ASSOCIATION_AMBIGUOUS`). Si el costo mínimo es `> 0.6`, la mano queda sin
  asociar (ausente). Si no hay muñeca de pose válida, no se inventa asociación por
  lado de imagen (ausente).
- **2 manos detectadas.** Se enumeran las 2 asignaciones inyectivas completas
  (mano-A→pose-15 & mano-B→pose-16, y la permutación inversa). Dentro de cada
  asignación se descartan los enlaces con costo `> 0.6`; una asignación es
  "completa" si sus dos enlaces sobreviven. Se elige la de MENOR costo total. Si
  el gap entre las dos asignaciones completas `< 0.05` ⇒
  `HAND_ASSOCIATION_AMBIGUOUS` y AMBAS manos implicadas quedan ausentes (es
  advertencia si los mínimos de calidad de §13 aún se cumplen; de lo contrario
  contribuye a `quality_pass=False`). Si sólo una asignación es completa, se usa
  esa sin regla de gap.

Punto de pose válido — regla POR DEFECTO FIJA (no delegada a config no congelada):

```
pose_point_valid = isfinite(x) and isfinite(y)
                   and visibility >= 0.5
                   and (presence >= 0.5 if presence_present else True)
```

La única perilla de config es la clave booleana
`quality_rules.use_presence` en `features_v1.yaml`, con valor **por defecto
`false`**. Semántica:
- `use_presence=false` (defecto): si el campo `presence` NO está presente en el
  landmark, el punto se valida sólo con `visibility`; si `presence` SÍ está
  presente, igualmente se exige `presence >= 0.5` (la condición de arriba). Es
  decir, la ausencia de campo nunca invalida, pero su presencia sí se respeta.
- `use_presence=true`: `presence` pasa a ser OBLIGATORIO; un landmark sin ese
  campo se considera inválido (más estricto). Esta variante NO se usa en este
  lote.

La prueba `F07` corre con el valor por defecto (`use_presence=false`). La mano
asociada aporta máscara 1 a sus 21 puntos. Punto no finito → ausente.

---

## 12. Normalización y forma (`geometry.py`)

1. Convertir `x,y` normalizados a px: `x*W`, `y*H` ANTES de distancias/rotaciones.
2. Por cuadro con hombros válidos: punto medio y distancia euclídea; exigir
   distancia de hombros `>= 0.08*W`.
3. Centro de secuencia = mediana por coordenada de los puntos medios válidos;
   escala de secuencia = mediana de las distancias válidas. COMUNES a los 6
   cuadros.
4. Para cada punto observado: `x_norm=(x_px-center_x)/scale`; igual en `y`.
   `y` positivo hacia abajo.
5. Salida: 49 triples `(x, y, m)` por instante. Si `m=0`, `x=y=0` tras
   normalizar. Coordenada cero real es válida cuando `m=1`.
6. NO interpolar faltantes, NO reflejar, NO recortar valores en v1. Si algún
   valor válido tiene `|valor| > 10` -> rechazar `GEOMETRY_OUTLIER`.

**Semántica del tercer canal `m` (hallazgo 3).** `m` es la **máscara binaria de
presencia-del-punto** (`m ∈ {0.0, 1.0}`): vale `1` cuando el punto fue observado
y resultó válido tras §11 (y, para puntos de mano, cuando la mano quedó asociada),
y `0` en ausencia. Corresponde exactamente al tercer canal que la API nombra de
forma GENÉRICA "presencia" en `api/constants.FEATURE_DIM` ("49 puntos x (x, y,
presencia)"): el array `(6,147)` es opaco para la API, que sólo lo transporta, de
modo que ambos nombres refieren al mismo canal. `m` **NO** es el score `presence`
de MediaPipe (ese score, cuando el landmark lo expone, sólo interviene en la regla
`pose_point_valid` de §11 para decidir si un punto es válido; nunca se almacena
como el valor del canal). En consecuencia: un punto con `presence` alto de
MediaPipe pero descartado por asociación/validez queda con `m=0`; el canal
almacenado es la máscara, no el score.

---

## 13. Calidad (`quality.py`)

`quality_pass` requiere: 6 cuadros decodificables; a lo sumo una persona por
cuadro; hombros válidos en >= 5 cuadros; >= 1 mano asociada en >= 4 cuadros. Al
fallar, `x` conserva forma con ceros en las ausencias pero el clasificador NO
debe consumirla (`quality_pass=False`). Se registran contadores, `reason_codes`
y latencia. **Orden determinista de reason_codes** (lista fija, se emiten en este
orden si aplican):
`INVALID_FRAME_COUNT, INVALID_IMAGE, SIZE_MISMATCH, MULTIPLE_PEOPLE,
INSUFFICIENT_POSE, INSUFFICIENT_HANDS, GEOMETRY_OUTLIER,
HAND_ASSOCIATION_AMBIGUOUS, CONFIG_MISMATCH`.
`HAND_ASSOCIATION_AMBIGUOUS` es advertencia si los mínimos se cumplen; los demás
estructurales o mínimos incumplidos -> `quality_pass=False`. El extractor NO
produce texto de usuario (la app traduce códigos).

---

## 14. Augmentation `train_v1` (`augment.py`)

`augment_train(x, seed, policy) -> ndarray`: opera sobre una copia float32;
preserva forma, máscaras y etiqueta; sólo lo llama el loader de entrenamiento;
nunca modifica el archivo base. Política `train_v1`:
1. Rotación global por secuencia, prob 0.5, ángulo uniforme en `[-5°, +5°]`
   alrededor del origen normalizado; mismo ángulo a todos los puntos y cuadros
   válidos.
2. Ruido gaussiano independiente std 0.005 (unidades de hombro), prob 0.5 por
   muestra.
Sólo modifica `x,y` donde `m=1`; los ceros de ausencia siguen en cero. Orden:
rotación y luego ruido. Sin reflexión, sin inversión temporal, sin mezcla de
clases. Generación online con política pura y sembrada. Val/test/producción
SIEMPRE sin augmentation.

---

## 15. Muestreo temporal (`temporal.py`)

Con N cuadros efectivos (`N >= 6`) se toman 6 índices. `round_half_up` se define
EXACTAMENTE así (half-up determinista sobre `float`, NO el banker's rounding del
`round()` nativo de Python), con retorno `int`:

```python
def sample_indices(n: int) -> list[int]:
    assert n >= 6
    idx = [int(math.floor(k * (n - 1) / 5 + 0.5)) for k in range(6)]
    # Invariantes garantizadas por construcción:
    assert idx[0] == 0 and idx[5] == n - 1
    return idx
```

Half-up significa que un empate exacto en `.5` redondea SIEMPRE hacia arriba
(p. ej. `2.5 → 3`), a diferencia de `round(2.5) == 2` en Python. Se opera sobre
`float`; el `+0.5` seguido de `floor` fija el desempate sin depender de Decimal.
`idx_0=0` e `idx_5=N-1` siempre. Tabla esperada (anclada en `F11`):

| N  | índices (k=0..5)        |
| -- | ----------------------- |
| 6  | `0, 1, 2, 3, 4, 5`      |
| 7  | `0, 1, 2, 4, 5, 6`      |
| 10 | `0, 2, 4, 5, 7, 9`      |
| 11 | `0, 2, 4, 6, 8, 10`     |

Utilidad pura. El módulo de Ortiz NO abre webcam (es de Juan); sólo implementa el
muestreo de índices.

---

## 16. Interfaz Python pública (desde los paquetes)

- `build_manifest(raw_dir, source_meta, label_map) -> ManifestReport`.
- `make_splits(manifest_path, seed=20261022) -> SplitReport`.
- `FeatureExtractor(config, model_paths, detector=None).extract_sequence(frames_rgb) -> FeatureResult`.
- `build_features(manifest_path, splits_path, extractor, out_dir) -> FeatureBuildReport`
  (procesa en orden de `sample_id`; un fallo de muestra se registra y no elimina
  a las demás; corrupción de config o incompatibilidad de esquema detiene el
  proceso).
- `augment_train(x, seed, policy) -> ndarray`.

Ninguna de estas carga detectores ni descarga en el import; la init explícita
verifica `.task` y hashes.

---

## 17. Esquema de características y versiones (`packaging.py`)

`features.npz`: `X float32 (N,6,147)`, `y int64 (N,)`, `sample_id unicode (N,)`,
`signer_id unicode (N,)`, `split unicode (N,)`, `quality_pass bool (N,)`. SIN
object arrays ni pickle (`np.savez` con arrays tipados; `allow_pickle=False` al
leer).

`feature_spec.json`: según §4.2; `feature_spec_hash` según §6.1. Un rename o
cambio de fecha NO altera el hash semántico.

---

## 18. Errores tipados

Códigos de calidad — CONJUNTO (sin orden) espejo del `frozenset`
`api/constants.QUALITY_REASON_CODES` (9 códigos):
`{INVALID_FRAME_COUNT, INVALID_IMAGE, SIZE_MISMATCH, MULTIPLE_PEOPLE,
INSUFFICIENT_POSE, INSUFFICIENT_HANDS, HAND_ASSOCIATION_AMBIGUOUS,
GEOMETRY_OUTLIER, CONFIG_MISMATCH}`. Como el origen es un `frozenset`, el orden
en el que se escriben aquí es IRRELEVANTE; la prueba de contrato compara
conjuntos (`==` de `set`), no secuencias. El ÚNICO orden normativo de EMISIÓN es
el de §13 (lista ordenada que es la fuente única del orden); §18 no redefine un
orden. Una prueba de contrato compara el conjunto de Ortiz con el de la API
(referencia de sólo lectura) y falla si divergen.

Manejo de errores por operación:
- `acquire`: config inválido -> exit 2 (fatal, loggea razón, no descarga).
  Archivo con rutas peligrosas -> rechazo con detalle, no mueve a destino (fatal
  para esa adquisición, exit 3). Reversible: no deja estado parcial (usa temp dir).
- `build_manifest`: registro corrupto/ambiguo -> fila en `exclusions.csv`
  (recuperable, continúa). Config/label_map inválido -> fatal, exit 2.
- `make_splits`: falta de clase en un split -> incidente con conteos (no fatal,
  se persiste; no recomputa semilla). Intersección no vacía o duplicado ->
  integridad, exit 3.
- `extract_sequence`: fallos estructurales -> `quality_pass=False` con
  `reason_codes` (no excepción; es resultado válido del contrato). Entrada no
  conforme (no 6 cuadros, tamaños distintos) -> `INVALID_FRAME_COUNT`/
  `SIZE_MISMATCH` en `reason_codes`.
- `build_features`: fallo de una muestra -> loggea y continúa. Corrupción de
  config o esquema incompatible -> detiene el proceso (fatal, exit 3).
- `validate`: 0 conforme, 2 entradas/config inválidos, 3 fallo de integridad.

Validación de cada entrada externa (tipo, requerido/opcional, límites, acción al
fallar) se declara en los modelos pydantic (§4) y los JSON Schema (§5); al fallar
se rechaza con `reason_code`/incidente, nunca se asume un valor por defecto
silencioso.

---

## 19. CLI `python -m lsc_data` y códigos de salida (`cli.py`, `__main__.py`)

Subcomandos (`--config` apunta a YAML/JSON versionado; rutas explícitas):
- `acquire --config configs/data/source.yaml --archive OPCIONAL`
- `inventory --raw-dir data/raw/lsc70_v2 --out reports/data/inventory.csv`
- `manifest --config configs/data/lsc70w.yaml --out data/interim/manifest.jsonl`
- `split --manifest data/interim/manifest.jsonl --seed 20261022 --out data/interim/splits.json`
- `features --config configs/data/features_v1.yaml --out artifacts/data/v1`
  (semántica de `--resume` fijada abajo)
- `validate --bundle artifacts/data/v1`

Semántica de `--resume` de `features` (hallazgo 5). `--resume` reanuda una
construcción previa en `--out` SÓLO si el bundle existente es compatible con la
configuración actual; en concreto, antes de reanudar se recomputan y comparan
EXACTAMENTE dos hashes:
- `feature_spec_hash` (§6.1) del `feature_spec.json` que resultaría de la config
  actual vs el guardado en el bundle existente, y
- `manifest_sha256` (§6.4) del manifiesto de entrada actual vs el registrado en el
  bundle.
Si AMBOS coinciden, se reanuda saltando las muestras ya empaquetadas y procesando
sólo las faltantes (en orden de `sample_id`). Si CUALQUIERA difiere, el comando
ABORTA con **exit 3 (integridad)** y un mensaje accionable que nombra el hash
divergente (esperado vs encontrado), y NO sobrescribe ni modifica el bundle
existente. Sin `--resume`, `features` construye desde cero en un directorio
limpio. (Los hashes de `.task` quedan fuera de esta comprobación porque este lote
no ejercita el detector real; cuando DO05 se desbloquee, `detector_versions`/
`model_asset_hashes` ya forman parte del `feature_spec_hash`, así que un cambio de
`.task` cambia ese hash y, por tanto, también aborta `--resume`.)

Códigos de salida (uniformes en toda la CLI):
- `0` éxito / conforme.
- `2` entradas o configuración inválidas (p. ej. YAML mal formado, regex que no
  casa, esquema de config incompatible).
- `3` fallo de integridad (fuga detectada, hash divergente, formas incorrectas,
  cobertura de clases rota en `validate`).
Cualquier otra excepción no controlada mapea a `2` sólo si proviene de entrada
inválida; de lo contrario se propaga como error interno con traza en log (no al
usuario).

---

## 20. Mapeo de pruebas de aceptación

Cada prueba usa FakeDetector y/o geometría sintética; ninguna depende de
Google Drive ni Mendeley. Las pruebas de fallo deliberado inyectan el defecto
indicado y DEBEN fallar (rojo) si el defecto no se detecta.

### Datos D01..D12 (`tests/data/test_manifest_D.py`, `test_split_leakage.py`)
- **D01** Reconstrucción válida: 6 índices únicos ordenados -> una `ManifestRecord`
  válida; `sample_id` y `sequence_sha256` estables.
- **D02** Faltan frames: 5 índices -> excluido con `reason_code` de secuencia
  incompleta; no se completa con otra ejecución.
- **D03** Duplicado ambiguo: dos frames con mismo `frame_index` -> exclusión.
- **D04** Personas/clases mezcladas en un grupo -> exclusión.
- **D05** Signo estático con imágenes idénticas -> válido (no invalida).
- **D06** `sequence_sha256` igual entre dos `signer_id` distintos -> incidente de
  copia cruzada. *(fallo deliberado: si el control no detecta la copia, falla.)*
- **D07** `class_id` fuera de 0..9 o negativo mal etiquetado -> el loader rechaza
  `y` fuera de 0..9; negativos van a paquete propio con `y=-1`.
- **D08** Split 70 personas -> 49/7/14 exactos con `seed=20261022`.
- **D09** Fuga de identidad: mismo `signer_id` en train y test -> *(fallo
  deliberado)* el control de intersección debe marcar integridad (exit 3).
- **D10** Cobertura: falta una clase en val -> incidente con conteos, sin recomputar
  semilla.
- **D11** Determinismo: dos ejecuciones from-scratch -> mismos `sample_id`,
  etiquetas y splits.
- **D12** Seguridad de adquisición: archivo con `..`/symlink/absoluta/ejecutable
  -> rechazo; nada se mueve a destino; **exit 3 (fallo de integridad)**, alineado
  con §19. Un `source.yaml` mal formado o incompatible -> **exit 2 (config
  inválido)**. Ambos casos no dejan estado parcial (uso de directorio temporal).

### Características F01..F12 (`tests/features/test_features_F.py`)
- **F01** FakeDetector con una persona y ambas manos -> `x (6,147)`,
  `quality_pass=True`, suma de máscaras coherente.
- **F02** Forma y dtype: `x.dtype==float32`, `x.shape==(6,147)`; triples en orden
  `(x,y,m)`.
- **F03** Normalización: centro/escala medianos comunes a los 6 cuadros; punto
  conocido cae donde se espera.
- **F04** Ausencia: `m=0` -> `x=y=0`; coordenada cero real con `m=1` se conserva.
- **F05** `GEOMETRY_OUTLIER`: valor válido `|v|>10` -> rechazo.
- **F06** `MULTIPLE_PEOPLE`: dos poses -> rechazo.
- **F07** Hombros insuficientes (<5 cuadros válidos) -> `INSUFFICIENT_POSE`,
  `quality_pass=False`.
- **F08** Manos y asociación por conteo (§11):
  - **F08a** 0 manos en un cuadro: sin asociación, sin regla de gap, cuadro no
    cuenta para `frames_with_hand`.
  - **F08b** 1 mano con costo `<=0.6`: se asigna al lado de menor costo, sin
    regla de gap (nunca ambigua); con costo `>0.6`: queda ausente.
  - **F08c** Manos insuficientes (<4 cuadros con mano asociada) ->
    `INSUFFICIENT_HANDS`, `quality_pass=False`.
  - **F08d** Cuadro SIN hombros válidos con 2 muñecas de mano devueltas por el
    FakeDetector (precondición de hombros, §11): ese cuadro NO asocia ninguna
    mano, no suma a `frames_with_hand` y NO emite `HAND_ASSOCIATION_AMBIGUOUS`
    (el denominador de normalización del costo sería indefinido). *(fallo
    deliberado si el cuadro suma mano o emite la advertencia de ambigüedad.)*
- **F09** Asociación con 2 manos (§11):
  - **F09a** gap `>=0.05`: se elige la asignación de menor costo total; no
    ambigua.
  - **F09b** gap `<0.05`: `HAND_ASSOCIATION_AMBIGUOUS`, ambas manos ausentes;
    advertencia si los mínimos de §13 se cumplen.
  - **F09c** una sola asignación completa (el otro enlace `>0.6`): se usa esa sin
    regla de gap.
- **F10** Augment `train_v1`: rotación+ruido sembrados preservan forma, máscaras y
  ceros de ausencia; val/test sin augment. *(fallo deliberado si augment toca
  puntos con m=0.)*
- **F11** Muestreo temporal: `sample_indices(N)` (half-up, §15) para
  `N=6,7,10,11` -> EXACTAMENTE los índices tabulados en §15
  (`6→[0,1,2,3,4,5]`, `7→[0,1,2,4,5,6]`, `10→[0,2,4,5,7,9]`,
  `11→[0,2,4,6,8,10]`), con `idx_0=0` e `idx_5=N-1`. *(fallo deliberado si se usa
  `round()` nativo half-to-even en lugar de half-up, o si el orden se altera.)*
- **F12** Orden temporal corrupto (frames barajados) -> el resultado difiere;
  prueba que el orden importa. *(fallo deliberado: debe fallar si el pipeline es
  invariante al orden.)*

### Contratos (`tests/contracts/`)
- Hash canónico estable ante rename/fecha, cambia ante `point_order`.
- Paridad pydantic↔JSON Schema.
- Orden canónico de etiquetas (no inferido de listdir/alfabético).
- Conjunto de `reason_codes` == `frozenset` de la API (comparación de conjuntos).
- Config incompatible: `labels.json` o `feature_spec_hash` divergente -> el
  consumidor falla al INICIO (no silenciosamente). `.task` faltante -> mensaje
  accionable y NINGUNA descarga silenciosa.
- `model_paths` (hallazgo 1): claves EXACTAS `{"pose","hands"}`; `config` acepta
  `dict` de `feature_spec`; `init()` con clave faltante/sobrante ->
  `CONFIG_MISMATCH` sin tocar disco (con `FakeDetector`).
- Compatibilidad básica de versiones de `requirements-data.in` SIN asumir un
  `pyproject.toml` raíz: valida el `.in`, Python 3.11 y pydantic v2, y además
  (hallazgo 2) afirma que `tests/data`, `tests/features` y `tests/contracts`
  existen y son recolectables vía paths explícitos, SIN leer ni depender de
  `pytest.ini`.

Para F01..F12 se usa geometría sintética; en comparaciones de transformación se
SIMULA la detección (no se re-ejecuta un detector sobre una imagen transformada).

**Nota de ejecución de toda la suite (hallazgos 1 y 2).** Dado el `pytest.ini`
raíz de Juan (`testpaths=tests/app,tests/e2e`), estas pruebas D/F y de contrato se
recolectan SÓLO con el comando canónico de paths explícitos de §23
(`python -m pytest tests/data tests/features tests/contracts`); un `pytest` simple
desde la raíz no las ejecutaría. Ortiz no modifica `pytest.ini`.

---

## 21. `.gitignore` (revisado contra el archivo real)

El `.gitignore` actual YA ignora por COMPLETO los datos pesados de Ortiz; NO
hace falta añadir patrones. Verificado en el archivo real:

- `data/` ignora TODO el árbol de datos → cubre `data/raw/`, `data/interim/` y
  `data/processed/` sin reglas adicionales.
- `artifacts/*` (con la sola excepción `!artifacts/README.md`) ignora TODO
  `artifacts/` → cubre `artifacts/data/` sin reglas adicionales.
- `*.task` y `*.keras` ya están ignorados (artefactos de modelo/extractor).

Por tanto, las cuatro adiciones del borrador v1 (`data/raw/`, `data/interim/`,
`data/processed/`, `artifacts/data/`) eran REDUNDANTES y se eliminan. La única
intervención admisible sobre `.gitignore` es, opcionalmente, un bloque de
COMENTARIO que documente que esas rutas de Ortiz quedan excluidas por los
patrones amplios ya existentes (sin duplicar patrones). Si no se agrega el
comentario, no se toca `.gitignore` en absoluto.

Verificación explícita de que lo que SÍ debe versionarse NO cae bajo ninguna
exclusión (confirmado contra los patrones reales):
- `contracts/` y `contracts/fixtures/SYNTHETIC_*` → no coinciden con ningún
  patrón; se versionan.
- `src/lsc_data/`, `src/lsc_features/`, `configs/data/`, `tests/**` → código y
  config; se versionan.
- `docs/data/`, `docs/article/data.md` → se versionan.
- `reports/data/README.md` y `reports/data/.gitkeep` → el único patrón de
  `reports/` es `reports/app/*_private*`, que NO toca `reports/data/`; se
  versionan. Los CSV generados (`reports/data/inventory.csv`) son salidas; si se
  decide no versionarlos se añadirá una regla específica `reports/data/*.csv`,
  pero eso es opcional y no forma parte de las rutas versionadas mínimas.
- `deployment/requirements-data.in` → no coincide con ningún patrón; se versiona.

Regla invariante: NUNCA excluir `contracts/`, los fixtures sintéticos ni el
código.

---

## 22. Dependencias entregadas a Acosta (`deployment/requirements-data.in`)

Lista con versiones ancladas: `numpy`, `opencv-python-headless`, `mediapipe`
(Tasks), `pydantic>=2`, `jsonschema`, `PyYAML`, `pytest`.

El `pyproject.toml`/lockfile/CI raíz son responsabilidad de Acosta y HOY NO
EXISTEN en el repositorio (verificado: sólo existe `pyproject.app.toml`, de
Juan). Por tanto `requirements-data.in` se entrega como SOLICITUD para que
Acosta lo fusione cuando cree ese lockfile raíz; Ortiz no puede ni debe asumir
un `pyproject.toml` raíz presente.

En consecuencia, `tests/contracts/test_requirements_version.py` NO debe acoplar
una prueba de Ortiz a un archivo ajeno inexistente: valida SÓLO (a) que
`deployment/requirements-data.in` existe, es parseable y ancla versiones;
(b) que el entorno Python en ejecución es 3.11 y que `pydantic` es v2
(`pydantic.VERSION` empieza por `2.`); y (c) que los tres directorios de prueba
de Ortiz (`tests/data`, `tests/features`, `tests/contracts`) existen y son
recolectables vía paths explícitos (comprobación de existencia de los directorios
y de que `pytest` los descubre cuando se pasan como argumentos). Esta prueba NO
lee ni depende de `pytest.ini` ni de un `pyproject.toml` raíz; si algún día
existe el lock raíz, la verificación de fusión será tarea de Acosta.

Ortiz NO toca `pyproject.toml` (inexistente, de Acosta), `pyproject.app.toml`
(de Juan), lockfile, CI raíz, `requirements-app.in` ni `pytest.ini` (de Juan).

**Configuración de pytest existente — `pytest.ini` raíz (hallazgos 1 y 2).** El
análisis anterior sobre `pyproject.toml` es correcto, pero el repo SÍ contiene un
archivo que gobierna el descubrimiento de pruebas y que el borrador no
contemplaba: `pytest.ini` en la raíz (verificado), cuyo encabezado declara que es
"Configuración local para ejecutar las pruebas del módulo de Juan" y que fija:

```ini
[pytest]
markers =
    realmodel: requiere model.keras + .task reales (excluir en CI rápida)
    e2e: pruebas de navegador con Playwright
    slow: pruebas lentas (latencia, sesiones prolongadas)
testpaths =
    tests/app
    tests/e2e
addopts = -ra
```

Con ese `testpaths`, una invocación simple `pytest` desde la raíz recolecta SÓLO
`tests/app` y `tests/e2e`; las pruebas de Ortiz (`tests/data`, `tests/features`,
`tests/contracts`) NO se recolectarían. Además hay un conflicto de propiedad:
`pytest.ini` NO está en las rutas de Ortiz (AGENTS.md lo deja fuera) y la CI raíz
es de Acosta, así que **Ortiz no edita `pytest.ini`**.

Resolución (opción A de la revisión, sin tocar archivos ajenos): el comando
canónico de la suite de Ortiz pasa los paths EXPLÍCITAMENTE, lo que anula
`testpaths` sin modificar `pytest.ini` (pytest, al recibir rutas en la línea de
comandos, ignora `testpaths`):

```
python -m pytest tests/data tests/features tests/contracts
```

Pasar los paths en la invocación conserva las `markers` y el `addopts=-ra` del
`pytest.ini` (que siguen aplicándose), de modo que las marcas `realmodel`/`slow`
siguen reconocidas; sólo se sustituye el conjunto de rutas recolectadas. Este
comando queda fijado en §23 y en `docs/data/README.md`. La incorporación de
`tests/data`, `tests/features` y `tests/contracts` al `testpaths` compartido del
futuro `[tool.pytest.ini_options]` raíz se registra en `docs/data/blocks.md` como
SOLICITUD a Acosta (dueño de la CI raíz), análoga a la entrega de
`requirements-data.in`. Mientras tanto, nada de la suite verde de Ortiz depende de
editar `pytest.ini`.

---

## 23. Testabilidad

- **Comando canónico de la suite de Ortiz (fijado, hallazgos 1 y 2).** Como el
  `pytest.ini` raíz (de Juan) fija `testpaths=tests/app,tests/e2e` y Ortiz no lo
  edita, la suite de Ortiz se ejecuta SIEMPRE con los paths explícitos, que anulan
  `testpaths` sin modificar el archivo ajeno:

  ```
  python -m pytest tests/data tests/features tests/contracts
  ```

  Este es el comando que aparece en `docs/data/README.md` y en los comandos de
  verificación de `docs/data/progreso-DO.md`. La fusión de esos paths al
  `testpaths` compartido es una solicitud a Acosta (ver §22 y
  `docs/data/blocks.md`). La suite verde NO depende ni de editar `pytest.ini` ni
  de Google Drive/Mendeley.
- Unitario (rápido, sin red, sin `.task`): toda la geometría, asociación,
  normalización, augment, temporal, hashing, splits, manifiesto y validación con
  FakeDetector y fixtures sintéticos.
- Integración (bloque, NO en este lote): extractor con `.task` real (DO05),
  paquete sobre corpus real (DO07), paridad con la API de Juan (DO08).
- El diseño es testeable porque la matemática es pura y el detector es
  inyectable; ningún camino verde depende de artefactos reales.

---

## 24. Bloques fuera de alcance (documentar, NO implementar) — `docs/data/blocks.md`

- **DO02 Descarga del corpus real.** Desbloqueo: colocar el archivo en una ruta
  local y ejecutar `python -m lsc_data acquire --config configs/data/source.yaml
  --archive <ruta>`; verifica DOI `10.17632/9ssyn8tff5.2`, versión, tamaño y
  SHA-256 contra `source.yaml`. No se simula descarga de red.
- **DO05 Ejecución con `.task` real.** Desbloqueo: obtener los `.task` de
  MediaPipe con procedencia verificable, registrarlos en `features_v1.yaml`
  (`detector_versions`, `model_asset_hashes`) y ejecutar el `RealDetector`; init
  explícita verifica existencia+hash. Falta de `.task` -> mensaje accionable, sin
  descarga silenciosa.
- **DO07 Paquete de características sobre datos reales.** Desbloqueo: con DO02 y
  DO05 resueltos, `python -m lsc_data features --config configs/data/features_v1.yaml
  --out artifacts/data/v1` y `validate --bundle artifacts/data/v1`.
- **DO08 Paridad con la API de Juan.** Desbloqueo: unificar `FeatureResult`
  (`valid_shoulders` int vs bool), confirmar conjunto de `reason_codes` y el
  `labels.json` de Acosta vs el de Ortiz mediante `tests/contracts`. Cambios en
  `api/` los hace Juan, no Ortiz.
- **DO09 Captura externa y consentimientos.** Desbloqueo: protocolo de captura,
  `consent_ref` opacos y campos de captura propia; nunca versionar copias de
  consentimiento.
- **DO10 Artículo final con métricas de Acosta.** Desbloqueo: incorporar métricas
  reales del modelo a `docs/article/data.md` cuando Acosta las entregue; hoy la
  sección queda con estructura y sin métricas inventadas.
- **DO11 Fusión de `testpaths` en la config de pruebas raíz (hallazgos 1 y 2).**
  Hoy el `pytest.ini` raíz (de Juan) fija `testpaths=tests/app,tests/e2e` y Ortiz
  no lo edita; su suite corre con paths explícitos (§23). Desbloqueo: solicitar a
  Acosta (dueño de la CI raíz) añadir `tests/data`, `tests/features` y
  `tests/contracts` al `testpaths` del futuro `[tool.pytest.ini_options]` del
  `pyproject.toml` raíz, para que la CI compartida recolecte las pruebas de Ortiz
  sin paths explícitos. Entrega análoga a `requirements-data.in`. Ortiz NO
  modifica `pytest.ini`.

---

## 25. Reglas de trabajo de IA aplicadas

No se fabrican resultados ni validación humana inexistente. Donde faltan datos,
pesos, licencia o participantes, se procede con fixtures ROTULADOS sintéticos y
se registra el bloque concreto. Fixtures sintéticos y métricas de datos reales se
separan por nombre y ruta (`contracts/fixtures/SYNTHETIC_*` vs `artifacts/data/`).
Cada tarea DO cierra en `docs/data/progreso-DO.md` con: id, cambio, archivo de
salida, comando de verificación, conteos medidos y riesgo/bloque pendiente.
Determinismo: ejecución from-scratch produce los mismos `sample_id`, etiquetas y
splits (prueba D11).

---

## 26. Respuesta a la revisión

### 26.0 Segunda ronda (`design-review.json` v2, veredicto CHANGES_REQUESTED — 1 HIGH, 1 MEDIUM, 3 NIT)

Se resuelven los 5 hallazgos sin tocar rutas de Juan ni de Acosta. Los hechos en
que se apoya la revisión se verificaron de nuevo contra el código real:
`pytest.ini` (raíz, de Juan, `testpaths=tests/app,tests/e2e`, con `markers` y
`addopts=-ra`), `api/constants.FEATURE_DIM` (docstring "(x, y, presencia)") y
`api/adapters/*` (coacción de `x` y `valid_shoulders`).

- **Hallazgo 1 [HIGH] — `pytest.ini` raíz excluye las pruebas de Ortiz — RESUELTO.**
  §22 y §23 reconocen el `pytest.ini` raíz (de Juan), fijan el comando canónico
  con paths explícitos `python -m pytest tests/data tests/features tests/contracts`
  (que anula `testpaths` sin editar el archivo), aclaran que Ortiz NO edita
  `pytest.ini`, y registran la fusión de esos paths al `testpaths` compartido como
  solicitud a Acosta (nuevo bloque DO11 en §24; también anotado junto a
  `requirements-data.in`). §20 añade la nota de ejecución de la suite.
- **Hallazgo 2 [MEDIUM] — `test_requirements_version.py` ignoraba `pytest.ini` — RESUELTO.**
  §22 cierra el análisis de ejecución de pruebas reconociendo el `pytest.ini`
  existente; la prueba de versiones (§20) añade la comprobación (c): afirma que
  `tests/data`, `tests/features` y `tests/contracts` existen y son recolectables
  vía paths explícitos, SIN leer ni depender de `pytest.ini`.
- **Hallazgo 3 [NIT] — canal "presencia" (API) vs máscara `m` (diseño) — RESUELTO.**
  §12 aclara que `m` es la máscara binaria de presencia-del-punto (0/1),
  correspondiente al canal que `FEATURE_DIM` nombra genéricamente "presencia", y
  NO el score `presence` de MediaPipe (que sólo decide validez del punto en §11).
- **Hallazgo 4 [NIT] — invariante de forma/dtype de `FeatureResult.x` — RESUELTO.**
  §8.1 declara como invariante de SALIDA que el extractor construye `x` con
  `x.shape==(6,147)` y `x.dtype==np.float32` antes de devolver `FeatureResult`
  (también cuando `quality_pass=False`), verificado por `F02`.
- **Hallazgo 5 [NIT] — `--resume` subdeterminado — RESUELTO.** §19 especifica que
  `--resume` reanuda sólo si `feature_spec_hash` y `manifest_sha256` del bundle
  coinciden con los recomputados; si alguno difiere, aborta con exit 3 (integridad)
  y mensaje accionable, sin sobrescribir.

### 26.1 Primera ronda (`design-review.json` v1, veredicto CHANGES_REQUESTED)

Se resuelven los 8 hallazgos. Ninguno requirió tocar rutas de Juan ni de Acosta;
los tres hechos en que se apoya la revisión se verificaron contra el código real
(`api/adapters/real.py`, `api/adapters/types.py`, `.gitignore`, ausencia de
`pyproject.toml` raíz).

- **Hallazgo 1 [HIGH] — firma de `model_paths` — RESUELTO.** §8.3 ahora fija la
  firma exacta `FeatureExtractor(config: Mapping, model_paths: Mapping[str, str|Path],
  detector=None)`, declara que `config` acepta un `dict` de `feature_spec`
  (como hace `api/adapters/real.py`), que `model_paths` tiene claves obligatorias
  EXACTAS `{"pose","hands"}`, y que `init()` valida existencia+hash sólo de esas
  dos entradas. Se añadió `tests/contracts/test_model_paths_contract.py`
  (§3, §20) que afirma las claves y el rechazo `CONFIG_MISMATCH`.

- **Hallazgo 2 [MEDIUM] — `.gitignore` redundante — RESUELTO.** §21 reescrito:
  reconoce que `data/` y `artifacts/*` (más `*.task`/`*.keras`) YA están
  totalmente ignorados; se eliminan las 4 adiciones redundantes (a lo sumo un
  comentario) y se verifica explícitamente que `contracts/`, fixtures sintéticos,
  código, `docs/data/`, `reports/data/README.md` y `.gitkeep` NO caen bajo
  exclusión. §3 actualizado.

- **Hallazgo 3 [MEDIUM] — `pyproject.toml` raíz inexistente — RESUELTO.** §2 y
  §22 ahora indican que el `pyproject.toml`/lockfile/CI raíz son de Acosta y HOY
  NO existen; `requirements-data.in` se entrega como solicitud y
  `test_requirements_version.py` valida sólo el `.in`, Python 3.11 y pydantic v2,
  sin asumir un `pyproject.toml` raíz.

- **Hallazgo 4 [MEDIUM] — `round_half_up` indefinido — RESUELTO.** §15 define
  `idx_k = int(math.floor(k*(n-1)/5 + 0.5))` para `k=0..5` (half-up determinista,
  NO el `round()` banker's de Python), retorno `int`, `idx_0=0`, `idx_5=N-1`, con
  tabla esperada para `N=6,7,10,11`. `F11` ancla esa tabla.

- **Hallazgo 5 [MEDIUM] — asociación 0/1/2 manos — RESUELTO.** §11 añade las tres
  ramas: 0 manos (sin asociación ni gap); 1 mano (menor costo si `<=0.6`, sin
  gap, nunca ambigua); 2 manos (enumerar asignaciones inyectivas, descartar
  enlaces `>0.6` antes de comparar totales, elegir menor costo, gap `<0.05` ->
  `HAND_ASSOCIATION_AMBIGUOUS` + ambas ausentes). `F08`/`F09` se desglosan en
  casos 0/1/2 manos.

- **Hallazgo 6 [MEDIUM] — `presence` de pose delegado a config — RESUELTO.** §11
  fija la regla por defecto `pose_point_valid = finite AND visibility>=0.5 AND
  (presence>=0.5 si presence presente else True)` y nombra la única perilla
  `quality_rules.use_presence` (defecto `false`); `F07` corre con el defecto. §8.2
  actualizado.

- **Hallazgo 7 [NIT] — orden de `reason_codes` entre §13 y §18 — RESUELTO.** §18
  aclara que su listado es el CONJUNTO (espejo del `frozenset`, orden
  irrelevante) y que el único orden normativo de emisión es el de §13.

- **Hallazgo 8 [NIT] — exit code de `acquire` con rutas peligrosas — RESUELTO.**
  `D12` (y §7) fijan exit 3 (integridad) para rutas peligrosas y exit 2 para
  `source.yaml` inválido, alineado con §19.

Decisiones donde había alternativas (opencv-headless, pydantic v2 + jsonschema,
`FeatureResult` como dataclass compatible con la API) siguen elegidas y
justificadas en §2 y §4. El único punto de fricción con el contrato existente
(`valid_shoulders` int en Ortiz vs `bool` en `api/adapters/types.py`, coaccionado
con `bool(count)` en `api/adapters/real.py`) se mantiene como bloque DO08 sin
modificar `api/`.

Entregable: `c:\Users\danie\Documents2\LSC/.agents/tasks/design.md`.
