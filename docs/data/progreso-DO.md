# Bitácora de cierre por DO (módulo de datos, Ortiz)

Registro por tarea DO del lote 1: id, cambio, archivo(s) de salida, comando de
verificación, conteos medidos y riesgo/bloque pendiente (design §25). Los
conteos de pruebas provienen del comando canónico
`python -m pytest tests/data tests/features tests/contracts`.

## DO01 — Configuración versionada, dependencias, reports y docs (FEAT-005)

- **Cambio.** Configuración del extractor v1, solicitud de dependencias a Acosta,
  carpeta de reportes y documentación del módulo; verificación de `.gitignore`.
- **Archivos de salida.** `configs/data/features_v1.yaml` (además de
  `source.yaml` y `lsc70w.yaml` ya existentes), `deployment/requirements-data.in`,
  `reports/data/.gitkeep`, `reports/data/README.md`,
  `reports/data/do-batch1-report.md`, `docs/data/README.md`,
  `docs/data/progreso-DO.md`, `docs/data/corpus-grammar-assumption.md`,
  `docs/data/blocks.md`, `docs/article/data.md`.
- **Comando de verificación.**
  ```
  python -c "import yaml,glob; cfgs={__import__('os').path.basename(p): yaml.safe_load(open(p,encoding='utf-8')) for p in glob.glob('configs/data/*.yaml')}; assert cfgs['features_v1.yaml']['quality_rules']['use_presence'] is False; print('configs ok')"
  git check-ignore contracts contracts/fixtures/SYNTHETIC_valid_sample.jsonl src/lsc_data deployment/requirements-data.in
  python -m pytest tests/data tests/features tests/contracts
  ```
- **Conteos medidos.** 3 YAML en `configs/data/` cargan con `yaml.safe_load`;
  `features_v1.yaml.quality_rules.use_presence == False`. `requirements-data.in`
  lista 7 dependencias ancladas. Suite: 77 passed, 1 skipped (skip por entorno
  Python 3.13 != 3.11 anclado; no rompe el verde).
- **Riesgo/bloque.** CERRADO. El patrón `data/` del `.gitignore` raíz (sin
  anclar) ignoraba CUALQUIER carpeta `data` a cualquier profundidad, incluidas
  `configs/data/`, `reports/data/`, `docs/data/` y `tests/data/`, impidiendo
  versionar los archivos nuevos. El diseño §21 asumía —incorrectamente— que
  `data/` sólo cubría el árbol de datos pesados. **Corrección acordada y
  autorizada por el usuario** (dueño de la decisión de propiedad): el usuario
  ancló el patrón a la raíz `data/`→`/data/` en el `.gitignore` raíz. Verificado
  con `git check-ignore --no-index`: `configs/data/*`, `reports/data/*`,
  `docs/data/*` y `tests/data/*` quedan versionables, mientras `data/raw`,
  `data/interim`, `data/processed` y `artifacts/data` siguen ignorados. Ortiz NO
  reclama propiedad de `.gitignore` (archivo raíz, fuera de sus rutas); el cambio
  lo hizo el usuario, no este módulo.

## DO03/DO04 — Extractor y suite (contexto)

Implementados en features previas (FEAT-003/FEAT-004). Verificación cubierta por
el mismo comando canónico de pruebas. Esta bitácora se amplía conforme se cierren
nuevas tareas DO.

## Cierre del lote 1 DO01 (task-do-batch1-ortiz)

- **Id.** `task-do-batch1-ortiz` — estado `completed`. Revisión
  `.agents/tasks/do-batch1-review.json` con verdict **APPROVED** (2 NIT sin
  acción requerida: ancla `.gitignore` como decisión del dueño y skip previsto
  por entorno Python 3.13 != 3.11).
- **Cambio.** Cierre del lote 1 del módulo de datos y extracción de Ortiz
  (FEAT-001..FEAT-005): contratos, `src/lsc_data`, `src/lsc_features`,
  `configs/data`, suite D01..D12 / F01..F12 + contratos, `requirements-data.in`
  y documentación. Sin red, sin corpus real, sin `.task` reales (detector
  inyectable y fixtures sintéticos).
- **Archivos del cierre.** `.agents/tasks/task-do-batch1-ortiz/task.json`
  (status→`completed`), `docs/data/progreso-DO.md` (esta bitácora),
  `docs/data/blocks.md` (nota de cierre del `.gitignore`),
  `reports/data/do-batch1-report.md`, junto con los entregables ya versionados
  en `configs/data/`, `reports/data/`, `docs/data/` y `docs/article/data.md`.
- **Comando de verificación.**
  ```
  python -m pytest tests/data tests/features tests/contracts
  ```
- **Conteos de pruebas.** Suite canónica **VERDE**: 78 ítems recolectados,
  **77 passed, 1 skipped** (el skip es por diseño: entorno Python 3.13 != 3.11
  anclado; se activará al fusionar el lock raíz de Acosta, bloque DO11). Sin
  fallos.
- **Bloques abiertos (fuera del lote 1, documentados en `docs/data/blocks.md`).**
  - **DO02** — Descarga del corpus real. Desbloquea al colocar el archivo local
    y correr `acquire` (verifica DOI/versión/tamaño/SHA-256); nunca descarga por
    red.
  - **DO05** — Ejecución con `.task` reales de MediaPipe. Desbloquea al obtener
    los `.task` con procedencia verificable y registrar versiones/hashes en
    `features_v1.yaml`.
  - **DO07** — Paquete de características sobre datos reales. Depende de DO02 y
    DO05.
  - **DO08** — Paridad con la API de Juan (unificar `FeatureResult`,
    `reason_codes` y `labels.json`). Los cambios en `api/` los hace Juan.
  - **DO09** — Captura externa y consentimientos. Requiere protocolo de captura;
    nunca se versionan copias de consentimiento.
  - **DO10** — Artículo final con métricas de Acosta. Las métricas reales entran
    cuando Acosta las entregue; hoy sin métricas inventadas.
  - **DO11** — Fusión de `testpaths` en la config de pruebas raíz. Se solicita a
    Acosta (dueño de la CI raíz); Ortiz no modifica `pytest.ini`.
