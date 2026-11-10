# Plan de ejecución — Lote 2 del módulo de datos (Ortiz): pipeline REAL

Objetivo: DESBLOQUEAR y EJECUTAR el pipeline real completo (DO02→DO03→DO04→
DO05-real+DO07→DO08→DO10) con el corpus y los `.task` YA en disco, SIN reescribir
el núcleo verde (contracts/, src/lsc_data, src/lsc_features, suite D01-D12/F01-F12).
Se activa lo que hoy está bloqueado y se ajusta la config de gramática al corpus
real; no se tocan `app/`, `api/`, `pytest.ini`, `pyproject.app.toml`.

## Hechos verificados en la exploración (fuente de las decisiones)

- Corpus real en disco: raíz `Speak in Colombian Sign Language A Dynamic LSC70
  Database\LSC70\LSC70\LSC70W`. Estructura `Per##/CLASE/Per##_CLASE_K.jpg`
  (carpetas de clase en MAYÚSCULAS; nombre de archivo repite `Per##_CLASE_`),
  39 personas (Per01..Per39), 388 ejecuciones, 2323 .jpg. Anomalías: Per39 sin
  TARDES ni YO; Per39/NOMBRE con 1 frame (K=0) en vez de 6.
- Clases de carpeta observadas: ANNOS, BUENAS, DIAS, GUSTAR, HOLA, LICOR, NOCHES,
  NOMBRE, TARDES, YO. Mapa canónico (contracts/labels.json): HOLA=0, BUENAS=1,
  DIAS=2, TARDES=3, NOCHES=4, YO=5, NOMBRE=6, ANNOS=7, GUSTAR=8, LICOR=9.
- `.task` reales en `artifacts/models/` (`pose_landmarker.task`,
  `hand_landmarker.task`), git-ignorados. Es la MISMA ruta que usa
  `api/adapters/real.py` como `artifacts_dir`, salvo que la API espera los
  `.task` en el mismo dir que `feature_spec.json`.
- Entorno: Python 3.13.2; `cv2` 5.0.0 importa y `cvtColor` funciona; `mediapipe`
  importa y expone `vision.PoseLandmarker`/`HandLandmarker`. Los resultados
  exponen `.pose_landmarks` (list[list[NormalizedLandmark]]) y `.hand_landmarks`
  + `.handedness`; `NormalizedLandmark` tiene `.x,.y,.z,.visibility,.presence`.
- Invocación de la CLI: `python -m lsc_data` NO resuelve (no está instalado el
  paquete). SÍ resuelven **`python -m src.lsc_data ...`** y
  **`$env:PYTHONPATH="src"; python -m lsc_data ...`**. Se adopta
  `python -m src.lsc_data` como forma canónica (evita exportar variables).
- Suite canónica hoy VERDE: `python -m pytest tests/data tests/features
  tests/contracts -q` → 77 passed, 1 skipped.

## Gaps reales que hay que activar (no reescribir)

1. `src/lsc_data/features_build.py::run_features` LANZA `ConfigError` siempre
   (bloque DO07): hay que cablear `build_features` + extractor real + escritura
   de bundle con `packaging.write_bundle` (incluye columna `split` y
   `feature_spec.json`, que `build_features` por sí solo NO escribe).
2. `src/lsc_features/detector.py::RealDetector.detect_pose/detect_hands` LANZAN
   `NotImplementedError`: hay que mapear los resultados de MediaPipe Tasks a
   `PoseResult`/`HandsResult`.
3. `configs/data/lsc70w.yaml`: la regex `^(?P<signer_id>S\d{3})/.../frame_...`
   y el `label_map` en minúsculas NO casan el corpus real. Hay que reescribir la
   regex a la gramática `Per##/CLASE/Per##_CLASE_K.jpg` y el `label_map` a las
   claves MAYÚSCULAS observadas.
4. `configs/data/features_v1.yaml` y `contracts/feature_spec.json`:
   `detector_versions`/`model_asset_hashes` en null (bloque DO05). Registrar la
   procedencia real de los `.task` cambia el `feature_spec_hash` (§6.1/§19); se
   maneja con cuidado para no romper `tests/contracts/test_feature_spec_hash.py`.
5. DO08: `api/adapters/types.FeatureResult.valid_shoulders` es `bool`; Ortiz emite
   `int`. Falta una prueba en `tests/contracts/` que fije la compatibilidad
   estructural (la API coacciona `bool(count)`), resuelta DESDE el lado de Ortiz.
6. `source.yaml`: `expected_sha256`/`expected_size_bytes` en null → `acquire
   --archive` rechaza (exit 2). Para DO02 real hay que registrar procedencia o
   documentar el modo "verificar sin archivo" (exit 0).

## Decisiones de diseño (una o dos frases cada una)

- **Gramática del corpus (DO02/DO03).** Se usa el adapter regex existente
  (`adapter_lsc70w.py`), sólo se cambia el patrón en `lsc70w.yaml`. Regex con
  retro-referencias nombradas (Python `re` las soporta):
  `^(?P<signer_id>Per\d{2})/(?P<class_label>[A-Z]+)/(?P=signer_id)_(?P=class_label)_(?P<frame_index>\d+)\.jpg$`.
  Razón: el adapter sólo exige los grupos `signer_id`, `class_label`,
  `frame_index`; `repetition_id` queda por defecto 0 (corpus sin repeticiones).
- **label_map MAYÚSCULAS.** `label_map` mapea las claves de carpeta reales a los
  `class_id` canónicos. `raw_label` lo deriva `manifest._raw_label_by_class` de
  la clave de carpeta; es aceptable que `raw_label` sea la etiqueta de carpeta.
  Razón: no se infiere orden de `os.listdir`; el `class_id` lo fija el mapa.
- **Base del corpus (`--raw-dir`).** Los comandos apuntan a la raíz real
  `LSC70W`. Como la ruta tiene espacios y está fuera de `data/raw/`, se copia o
  se referencia directamente; se referencia directamente con comillas (no se
  mueve el corpus). `frame_paths` del manifiesto quedan relativos a esa raíz; el
  loader de `build_features` debe resolverlos contra la misma raíz.
- **Resolución de rutas en el loader (DO07).** `build.py::_default_frame_loader`
  hoy lee `record["frame_paths"]` tal cual (relativos). Se inyecta un loader que
  antepone la `--raw-dir` para abrir el archivo real; NO se modifica la firma
  pública de `build_features` (acepta `frame_loader`). La orquestación vive en
  `features_build.py` (ruta de Ortiz), no en `build.py`.
- **RealDetector (DO05).** Se completa el mapeo MediaPipe→dataclasses de Ortiz en
  `detector.py`: `mp.Image(SRGB, data=rgb)`, `.detect(image)`, y traducción de
  `result.pose_landmarks`/`result.hand_landmarks` a `PoseResult`/`HandsResult`
  conservando `x,y,visibility,presence`. Opciones ya fijadas en el `__init__`
  existente (IMAGE, num_poses=2, num_hands=2, thresholds 0.5). Razón: el núcleo
  matemático no cambia; sólo se rellena la frontera con MediaPipe.
- **Escritura de bundle (DO07).** `run_features` usa `build_features` para la
  extracción pero persiste con `packaging.write_bundle` para incluir `split` y
  `feature_spec.json` (lo que `validate` exige). Razón: `validate.py` requiere
  `feature_spec.json` con `feature_spec_hash` y columna `split`; el npz de
  `build.py` por sí solo no los tiene.
- **feature_spec_hash y .task (DO05).** Registrar `detector_versions`/
  `model_asset_hashes` reales en `contracts/feature_spec.json` CAMBIA el hash
  canónico; se recomputa y se actualiza el campo `feature_spec_hash` del JSON con
  `contracts.canonical.feature_spec_hash`, y se verifica que
  `test_feature_spec_hash.py` siga verde (compara estabilidad ante rename/fecha,
  no un valor fijo). `features_v1.yaml` refleja los mismos valores.
- **DO08 parity int↔bool.** Nueva prueba `tests/contracts/test_feature_result_parity.py`
  que importa `api.adapters.types.FeatureResult` (solo lectura) y
  `src.lsc_features.types.FeatureResult`, afirma campos compatibles y que
  `bool(valid_shoulders_int)` es la coacción que hace la API. No se toca `api/`.
- **DO02 sin congelar procedencia.** Como no hay SHA-256/size oficiales del
  archivo, DO02 se ejecuta en modo "verificar sin archivo" (`acquire --config`
  sin `--archive` → exit 0, registra DOI/URL), que es el camino real soportado
  sin red; el corpus ya está extraído en disco. Se documenta en progreso-DO.

## Pasos (ordenados por dependencia)

- [ ] 1. (DO02/DO03-config) Reescribir la gramática real en
      `configs/data/lsc70w.yaml`: `path_regex` a
      `^(?P<signer_id>Per\d{2})/(?P<class_label>[A-Z]+)/(?P=signer_id)_(?P=class_label)_(?P<frame_index>\d+)\.jpg$`
      y `label_map` con claves MAYÚSCULAS (HOLA:0, BUENAS:1, DIAS:2, TARDES:3,
      NOCHES:4, YO:5, NOMBRE:6, ANNOS:7, GUSTAR:8, LICOR:9). Mantener
      `default_repetition_id: 0`.
      Archivos: `c:\Users\danie\Documents2\LSC\configs\data\lsc70w.yaml`.
      Verificar: `python -c "from src.lsc_data.adapter_lsc70w import load_grammar_config, Lsc70wAdapter; g=load_grammar_config('configs/data/lsc70w.yaml'); a=Lsc70wAdapter(g); p=a.parse_frame('Per01/HOLA/Per01_HOLA_0.jpg'); print(p.signer_id,p.class_label,p.class_id,p.frame_index)"`
      → imprime `Per01 HOLA 0 0`. La suite sigue verde:
      `python -m pytest tests/data tests/features tests/contracts -q`.

- [ ] 2. (DO02) Ejecutar `acquire` en modo verificación sin red y registrar
      procedencia. No se congela SHA-256/size (no hay valores oficiales del
      archivo descargado); se corre sin `--archive`.
      Comando: `python -m src.lsc_data acquire --config configs/data/source.yaml`
      Artefacto: log con DOI `10.17632/9ssyn8tff5.2`, URL, versión; exit 0.
      Verificar: el comando termina con código 0 e informa el bloque DO02; el
      corpus ya extraído se usa directamente en el paso 3.

- [ ] 3. (DO03) Inventario y manifiesto reales sobre la raíz del corpus.
      Comandos (raíz con comillas; se usa directamente, no se copia a data/raw):
      `python -m src.lsc_data inventory --raw-dir "Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W" --out reports/data/inventory.csv`
      `python -m src.lsc_data manifest --config configs/data/lsc70w.yaml --raw-dir "Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W" --out data/interim/manifest.jsonl --exclusions data/interim/exclusions.csv`
      Artefactos: `reports/data/inventory.csv` (39 grupos Per##),
      `data/interim/manifest.jsonl`, `data/interim/exclusions.csv`.
      Nota: `manifest.build_manifest` lee tamaños con cv2 (disponible) y hace
      `adapter.parse_frame` sobre CADA archivo; si alguna ruta no casa la
      gramática FALLA con exit 2. Las secuencias incompletas de Per39
      (NOMBRE 1 frame; TARDES/YO ausentes) caen a `exclusions.csv`
      (`INVALID_FRAME_COUNT`), no rompen el manifiesto.
      Verificar: `python -c "import json; recs=[json.loads(l) for l in open('data/interim/manifest.jsonl',encoding='utf-8') if l.strip()]; print('válidas',len(recs)); import collections; print('por_clase',dict(collections.Counter(r['class_id'] for r in recs)))"`
      → nº de válidas = 388 − (ejecuciones de Per39 incompletas). Exclusión de
      `Per39/NOMBRE` presente en `exclusions.csv`. Conteo esperado: 386 válidas
      (388 ejecuciones menos NOMBRE incompleto de Per39; TARDES/YO de Per39 no
      existen como carpetas, no generan ejecución). Registrar el conteo REAL
      medido, sin forzar.

- [ ] 4. (DO04) Split determinista con seed 20261022 sobre el manifiesto real.
      Comando: `python -m src.lsc_data split --manifest data/interim/manifest.jsonl --seed 20261022 --out data/interim/splits.json`
      Artefacto: `data/interim/splits.json` con `signer_ids`/`sample_ids` por
      split, `manifest_sha256`, `class_coverage`, `incidents`.
      Nota: N=39 personas (no 70). `split_counts(39)` → n_test=ceil(7.8)=8,
      n_val=max(1,floor(3.9+0.5))=4, n_train=27 (27/4/8). Registrar el conteo
      REAL. Posibles incidentes de cobertura por Per39 (sin TARDES/YO) si cae en
      val/test: se persisten como incidente, NO se recomputa la semilla.
      Verificar: `python -c "import json; d=json.load(open('data/interim/splits.json',encoding='utf-8')); print('seed',d['seed']); print('train/val/test',len(d['signer_ids']['train']),len(d['signer_ids']['val']),len(d['signer_ids']['test'])); print('incidents',d['incidents'])"`
      → seed=20261022, suma de personas = 39, sin fuga (el comando lanza exit 3
      si hubiera intersección).

- [ ] 5. (DO05) Completar `RealDetector.detect_pose/detect_hands` en
      `src/lsc_features/detector.py` para mapear MediaPipe Tasks → dataclasses.
      En `detect_pose`: construir `mp.Image(image_format=mp.ImageFormat.SRGB,
      data=image_rgb)`, llamar `self._pose.detect(image)`, y traducir
      `result.pose_landmarks` (list[list[NormalizedLandmark]]) a
      `PoseResult(pose_landmarks=[[PoseLandmark(x,y,visibility,presence) ...]])`.
      En `detect_hands`: igual con `self._hands.detect(image)` →
      `HandsResult(hand_landmarks=[[HandLandmark(x,y)...]], handedness=[...])`.
      Imports de mediapipe PEREZOSOS (ya están en `__init__`; añadir `import
      mediapipe as mp` perezoso en los métodos). No tocar el núcleo matemático.
      Archivos: `c:\Users\danie\Documents2\LSC\src\lsc_features\detector.py`.
      Verificar: `python -c "from src.lsc_features.detector import RealDetector; import numpy as np, cv2; d=RealDetector('artifacts/models/pose_landmarker.task','artifacts/models/hand_landmarker.task'); img=cv2.cvtColor(cv2.imread('Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W/Per01/HOLA/Per01_HOLA_0.jpg'),cv2.COLOR_BGR2RGB); pr=d.detect_pose(img); hr=d.detect_hands(img); print('poses',len(pr.pose_landmarks),'hands',len(hr.hand_landmarks)); d.close()"`
      → imprime conteos sin excepción (≥0). La suite sigue verde (RealDetector no
      se importa en los tests sintéticos).

- [ ] 6. (DO05) Registrar procedencia de los `.task` en
      `contracts/feature_spec.json` y `configs/data/features_v1.yaml`:
      `detector_versions` = versión declarada (string, p. ej. "mediapipe-tasks
      pose/hand v... " o la que figure en el paquete) y `model_asset_hashes` =
      SHA-256 reales de `artifacts/models/*.task`
      (pose `4EAA5EB7...C4AD`, hands `FBC2A300...CDE1`, en minúsculas). Recomputar
      el campo `feature_spec_hash` del JSON con
      `contracts.canonical.feature_spec_hash` sobre el spec actualizado.
      Archivos: `contracts/feature_spec.json`, `configs/data/features_v1.yaml`.
      Verificar: `python -c "import json; from contracts.canonical import feature_spec_hash; s=json.load(open('contracts/feature_spec.json',encoding='utf-8')); print('hash coincide', feature_spec_hash(s)==s['feature_spec_hash'])"`
      → `True`. Y `python -m pytest tests/contracts/test_feature_spec_hash.py -q`
      → verde (la prueba verifica estabilidad ante rename/fecha, no un valor
      literal).

- [ ] 7. (DO07) Cablear la construcción real del bundle en
      `src/lsc_data/features_build.py::run_features`: en lugar de lanzar
      `ConfigError`, cargar el `feature_spec` (de `configs/data/features_v1.yaml`
      o `contracts/feature_spec.json`), construir
      `FeatureExtractor(spec, {"pose":..., "hands":...}, detector=RealDetector(...))`
      con los `.task` de `artifacts/models/`, llamar
      `src.lsc_features.build.build_features(manifest, splits, extractor, out_dir,
      frame_loader=<loader que antepone --raw-dir>)`, y escribir el bundle final
      con `packaging.write_bundle` (X, y, sample_id, signer_id, **split** desde
      splits.json, quality_pass, feature_spec, exclusions). Mantener la semántica
      de `--resume` (ya implementada en `check_resume_compatibility`). Añadir el
      parámetro de raíz del corpus a la CLI `features` (`--raw-dir`) en `cli.py`
      (ruta de Ortiz) para resolver `frame_paths`.
      Archivos: `src/lsc_data/features_build.py`, `src/lsc_data/cli.py`.
      Comando: `python -m src.lsc_data features --config configs/data/features_v1.yaml --manifest data/interim/manifest.jsonl --splits data/interim/splits.json --raw-dir "Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W" --out artifacts/data/v1`
      Artefactos: `artifacts/data/v1/features.npz`, `.../feature_spec.json`,
      `.../exclusions.csv`.
      Verificar: `python -m src.lsc_data validate --bundle artifacts/data/v1`
      → exit 0. Y `python -c "from src.lsc_data.packaging import read_features_npz; b=read_features_npz('artifacts/data/v1/features.npz'); print('N',b.X.shape[0],'shape',b.X.shape[1:],'dtype',b.X.dtype); import collections; print('splits',dict(collections.Counter(b.split.tolist()))); print('quality_pass',int(b.quality_pass.sum()))"`
      → N == nº de válidas del manifiesto, shape (6,147), dtype float32, splits no
      vacíos, quality_pass registrado.

- [ ] 8. (DO07) Añadir una prueba de humo de orquestación en `tests/data/` que
      ejercite `run_features` con un `RealDetector` sustituido por `FakeDetector`
      y un `frame_loader` sintético (sin corpus ni .task reales), verificando que
      escribe un bundle válido (`validate` → 0) con columna `split`. Esto mantiene
      el camino de orquestación cubierto en la suite verde sin depender de
      artefactos reales.
      Archivos: `tests/data/test_features_build_orchestration.py`.
      Verificar: `python -m pytest tests/data/test_features_build_orchestration.py -q`
      → verde. Suite completa sigue verde.

- [ ] 9. (DO08) Nueva prueba de paridad con la API en `tests/contracts/`:
      importar `api.adapters.types.FeatureResult` (solo lectura) y
      `src.lsc_features.types.FeatureResult`; afirmar que los nombres de campo
      coinciden, que Ortiz emite `valid_shoulders: int` y que `bool(int)` es la
      coacción que la API aplica (`api/adapters/real.py` hace
      `valid_shoulders=bool(getattr(result,'valid_shoulders',True))`). Documentar
      que la divergencia int↔bool se resuelve desde el lado de Ortiz sin tocar
      `api/`. No modificar `api/`.
      Archivos: `tests/contracts/test_feature_result_parity.py`.
      Verificar: `python -m pytest tests/contracts/test_feature_result_parity.py -q`
      → verde.

- [ ] 10. (DO10) Actualizar `docs/article/data.md` con los CONTEOS REALES medidos
      (personas=39, ejecuciones=388, .jpg=2323, válidas del manifiesto,
      excluidas con su reason_code, split 27/4/8 o el real, cobertura de clases,
      incidentes). Sin métricas del modelo (de Acosta; se dejan como pendiente).
      Actualizar `docs/data/progreso-DO.md` con una entrada por DO02..DO08 y DO10
      (id, cambio, archivos, comando de verificación, conteos, riesgo/bloque).
      Archivos: `docs/article/data.md`, `docs/data/progreso-DO.md`,
      `reports/data/do-batch2-report.md` (nuevo).
      Verificar: los conteos del artículo coinciden con los de `inventory.csv`,
      `manifest.jsonl`, `splits.json` y el bundle; no hay métricas de modelo
      inventadas. Suite final verde:
      `python -m pytest tests/data tests/features tests/contracts -q`.

## Propiedad respetada

Sólo se editan rutas de Ortiz: `contracts/`, `src/lsc_data`, `src/lsc_features`,
`configs/data`, `tests/{data,features,contracts}`, `reports/data`, `docs/data`,
`docs/article/data.md`. NO se tocan `app/`, `api/`, `pytest.ini`,
`pyproject.app.toml`, lockfile ni CI raíz. El comando de suite usa paths
explícitos (anula `testpaths` de Juan sin editar `pytest.ini`).

## Supuestos y lagunas (asunción razonable, no bloqueo)

- DO02 real sin SHA-256/size oficiales: se usa el modo verificación sin
  `--archive` (exit 0) porque el corpus ya está extraído; congelar procedencia
  del archivo es tarea futura cuando exista un ZIP oficial con hash publicado.
- `detector_versions`: si el paquete mediapipe no expone una versión por `.task`,
  se registra la versión del paquete mediapipe + nombre del asset como string de
  procedencia; lo esencial es `model_asset_hashes` (SHA-256 reales ya medidos).
- Conteos (386 válidas, split 27/4/8): son estimaciones derivadas del inventario;
  el ejecutor registra los valores REALES observados por cada comando.
