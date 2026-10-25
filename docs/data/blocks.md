# Bloques fuera de alcance del lote 1 (documentar, NO implementar)

Estas tareas DO quedan fuera del primer lote porque dependen del corpus real, de
los `.task` reales de MediaPipe, de artefactos de Acosta o de la CI raíz. Cada
bloque indica su condición de desbloqueo EXACTA (design §24). No se simula ni se
fabrica ninguno de estos resultados (design §25).

## DO02 — Descarga del corpus real

- **Por qué está bloqueado.** `acquire` NUNCA descarga por red; el corpus real
  no está en el repositorio.
- **Desbloqueo.** Colocar el archivo del corpus en una ruta local y ejecutar:

  ```
  python -m lsc_data acquire --config configs/data/source.yaml --archive <ruta>
  ```

  Verifica DOI `10.17632/9ssyn8tff5.2`, versión, tamaño y SHA-256 contra
  `configs/data/source.yaml` (hoy `expected_sha256`/`expected_size_bytes` son
  `null`; se completan al congelar la procedencia). Rutas peligrosas en la
  extracción (`..`, absolutas, symlinks que escapan, extensiones no permitidas)
  -> rechazo, exit 3, sin mover nada a destino. No se simula descarga.

## DO05 — Ejecución con `.task` real de MediaPipe

- **Por qué está bloqueado.** El extractor es matemática pura con detector
  inyectable; el camino real (`RealDetector`) no se ejercita y no hay `.task`
  reales con procedencia verificable.
- **Desbloqueo.** Obtener los `.task` (`pose_landmarker.task`,
  `hand_landmarker.task`) con procedencia verificable, registrar sus versiones y
  hashes en `configs/data/features_v1.yaml` (`detector_versions`,
  `model_asset_hashes`) y ejecutar con el `RealDetector`. La `init()` explícita
  verifica existencia+hash de EXACTAMENTE las claves `{"pose","hands"}` de
  `model_paths`. Falta de `.task` -> mensaje accionable, SIN descarga silenciosa.

## DO07 — Paquete de características sobre datos reales

- **Por qué está bloqueado.** Depende de DO02 (corpus) y DO05 (`.task`).
- **Desbloqueo.** Con DO02 y DO05 resueltos:

  ```
  python -m lsc_data features --config configs/data/features_v1.yaml \
      --manifest data/interim/manifest.jsonl --splits data/interim/splits.json \
      --out artifacts/data/v1
  python -m lsc_data validate --bundle artifacts/data/v1
  ```

## DO08 — Paridad con la API de Juan

- **Por qué está bloqueado.** Unificación del contrato compartido entre módulos.
- **Desbloqueo.** Unificar `FeatureResult` (`valid_shoulders` es `int` en Ortiz
  y `bool` en `api/adapters/types.py`; la API lo coacciona con `bool(count)`),
  confirmar que el conjunto de `reason_codes` coincide con
  `api/constants.QUALITY_REASON_CODES` y comparar el `labels.json` de Acosta
  contra `contracts/labels.json` mediante `tests/contracts`. Los cambios en
  `api/` los hace Juan, no Ortiz.

## DO09 — Captura externa y consentimientos

- **Por qué está bloqueado.** No hay protocolo de captura ni participantes en
  este lote.
- **Desbloqueo.** Definir protocolo de captura, usar `consent_ref` como id opaco
  y poblar los campos opcionales de captura propia del `ManifestRecord`. NUNCA se
  versionan copias de consentimiento (design §7/§25).

## DO10 — Artículo final con métricas de Acosta

- **Por qué está bloqueado.** Las métricas del modelo son de Acosta y aún no
  existen.
- **Desbloqueo.** Incorporar las métricas reales a `docs/article/data.md` cuando
  Acosta las entregue. Hoy esa sección queda con estructura y SIN métricas
  inventadas (design §25).

## DO11 — Fusión de `testpaths` en la config de pruebas raíz

- **Por qué está bloqueado.** El `pytest.ini` raíz (de Juan) fija
  `testpaths=tests/app,tests/e2e` y Ortiz NO lo edita; la suite de Ortiz corre
  con paths explícitos (design §22/§23).
- **Desbloqueo.** Solicitar a Acosta (dueño de la CI raíz) añadir `tests/data`,
  `tests/features` y `tests/contracts` al `testpaths` del futuro
  `[tool.pytest.ini_options]` del `pyproject.toml` raíz, para que la CI
  compartida recolecte las pruebas de Ortiz sin paths explícitos. Entrega análoga
  a `deployment/requirements-data.in`. Ortiz NO modifica `pytest.ini`.

## Nota: corrección del `.gitignore` raíz (resuelta, NO es un bloque abierto)

Durante FEAT-005 se detectó que el patrón `data/` del `.gitignore` raíz no estaba
anclado y, por tanto, ignoraba CUALQUIER carpeta `data` a cualquier profundidad
(`configs/data/`, `reports/data/`, `docs/data/`, `tests/data/`), impidiendo
versionar los entregables de Ortiz. El diseño §21 asumía —incorrectamente— que
`data/` sólo cubría el árbol de datos pesados. **El usuario (dueño de la decisión
de propiedad) resolvió el conflicto anclando el patrón a la raíz
`data/`→`/data/`** en el `.gitignore` raíz. Queda como corrección acordada y
autorizada, no como un bloqueo pendiente ni como un cambio reclamado por Ortiz
(el `.gitignore` raíz no está entre sus rutas). `data/raw`, `data/interim`,
`data/processed` y `artifacts/data` siguen ignorados.
