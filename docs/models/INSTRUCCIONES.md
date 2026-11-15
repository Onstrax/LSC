# Cómo ejecutar el proyecto y comprobar que funciona (módulo de Acosta)

Todos los comandos se ejecutan **desde la raíz del repo**. Python 3.11 es el objetivo del diseño
(3.12 también fue probado). Disco: ~2 GB (TensorFlow).

## 0. Entorno (una sola vez)

PowerShell (Windows):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[features,train,app,dev]"
```
bash / macOS / Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[features,train,app,dev]"
```

## 1. ¿Funciona el código? (NO necesita datos reales, ~2 min)

```bash
python -m src.lsc_models verify
```
Debe terminar con `RESULTADO: la tubería funciona` y código de salida 0. Hace: importa dependencias,
valida las 12 ejecuciones configuradas, verifica los contratos de Ortiz, genera datos **sintéticos**,
corre las 12 ejecuciones con presupuesto reducido, crea tablas/selección/gráficas, la sensibilidad,
comprueba que el test sigue bloqueado, que el Predictor exportado coincide con el modelo en memoria
y que la API de Juan carga tu modelo. Verá `[SKIP] 4. features.npz REAL`: es normal sin datos reales.

Con las pruebas unitarias incluidas: `python -m src.lsc_models verify --with-tests`.
Si algo falla: `python -m src.lsc_models verify --trace`.

> Que esto pase demuestra que el **código** funciona. No dice nada sobre qué tan bien se reconoce LSC:
> eso solo lo dicen los pasos 3–4 con datos reales.

## 2. Suite completa del repositorio

```bash
python -m pytest tests/contracts tests/data tests/features tests/models tests/inference tests/app -q
```
Esperado: todo en verde; 1 omitida si usa Python distinto de 3.11 (comprobación de versión de Ortiz).
(No use `pytest` a secas: `pytest.ini` de Juan solo apunta a `tests/app` y `tests/e2e`.)

## 3. Generar `features.npz` REAL

Necesita dos cosas que no vienen en el repo:

**a) Modelos MediaPipe (`.task`) — se descargan solos.**
```bash
python -m src.lsc_models download-models
```
Baja los `.task` oficiales de MediaPipe y **solo conserva los que coinciden con el SHA-256 del
contrato** de `feature_spec.json` (se desconoce qué variante de pose usó Ortiz, así que prueba
heavy/full/lite y se queda con la que coincida). Necesita internet. Si ninguna coincide o no hay red,
explica qué pasó; sin internet, descargue los dos `.task` a mano desde la página de MediaPipe y cópielos
a `artifacts/models/`. Si usted descargó variantes distintas, `--accept-mismatch` las acepta, pero los
resultados NO tendrán la procedencia del contrato y debe declararse. `prepare-data` ejecuta este paso
automáticamente si faltan (desactivar con `--no-download`).

**b) El corpus LSC70W — descarga manual (una vez).** Abra
https://data.mendeley.com/datasets/9ssyn8tff5/2 (DOI 10.17632/9ssyn8tff5.2, CC BY 4.0), descargue el
conjunto y descomprímalo en la raíz del repo para que exista
`Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W`
(otra ubicación: `--raw-dir "<ruta>"`).

Luego:
```bash
python -m src.lsc_models prepare-data
```
Orquesta el pipeline de Ortiz (inventario → manifiesto → particiones por persona → características →
validación) y escribe `artifacts/data/v1/features.npz` (varios minutos). Si falta algo, se detiene
diciendo exactamente qué; no inventa datos.

Solo para ver la tubería con datos de mentira: `python -m src.lsc_models prepare-data --synthetic`
(escribe en `artifacts/data/synthetic/`, nunca en `v1`).

## 4. Las 12 ejecuciones y el análisis (A1–A3)

```bash
python -m src.lsc_models run-all          # 12 ejecuciones; reanudable (puede interrumpir y repetir)
python -m src.lsc_models aggregate        # tablas, selección, gráficas
python -m src.lsc_models sensitivity      # sensibilidad a fallos del extractor (solo validación)
```
Salidas en `reports/models/` (ver su README). Para leerlas:

| Archivo | Qué responde |
|---|---|
| `ablation_summary.csv` | ¿Qué configuración es mejor? (media ± desv. de 3 semillas) |
| `per_seed.csv` | Resultado de cada una de las 12 ejecuciones y del baseline mayoritario |
| `per_class_by_seed.csv` | ¿Qué señas se confunden? |
| `selection.json` | Configuración elegida, semilla 17 desplegada, metas vs. validación |
| `figures/` | Curvas de aprendizaje, matrices de confusión |
| `sensitivity_hand_dropout.csv` | ¿Cuánto cae el desempeño si el extractor pierde manos? |

Señales de que **no** hay evidencia: `evidence_valid: false` en `selection.json` o en un `metrics.json`
(datos sintéticos). Sin datos reales, `run-all` sin `--fast` simplemente no arranca.

## 5. Congelar (18 nov.)

```bash
python -m src.lsc_models freeze
```
Crea `reports/models/freeze.json` y habilita **una sola** lectura del test (A4). Después de esto no se
cambian configuración ni modelos; rechaza ejecuciones sintéticas.

## 6. Probar la aplicación completa

La app = API (FastAPI) + interfaz web. El navegador usa **su propia cámara**; el servidor solo recibe imágenes.

### 6a. Modo simulado (sin modelo ni datos) — funciona ya
```bash
python -m src.lsc_models check-app --spawn mock   # prueba automática por HTTP (10 comprobaciones)
python -m src.lsc_models run-app --mode mock --open
```
Se abre http://localhost:8000. Debe verse el rótulo **"Modo de prueba · respuestas simuladas"**. Permita la
cámara, siga la captura guiada (preparación 2 s + captura 1 s) y verá una sugerencia o "repita". Las
respuestas son simuladas: sirve para probar la interfaz, no el reconocimiento.
Revise a mano: denegar la cámara (mensaje comprensible), repetir un intento, borrar el historial, teclado.

### 6b. Modo real con un modelo de DEMO sintético (prueba de integración de punta a punta)
Cámara → MediaPipe → modelo → interfaz, sin esperar el corpus. **Las respuestas no significan nada**:
el modelo se entrenó con datos de mentira. Paquete marcado `SYNTHETIC-DEMO`.
```bash
python -m src.lsc_models download-models                 # .task oficiales (necesita internet)
python -m src.lsc_models prepare-data --synthetic
python -m src.lsc_models run-all --bundle artifacts/data/synthetic/features.npz --out reports/models_synthetic --fast --only bilstm_noaug__seed17
python -m src.lsc_models export-artifacts --reports reports/models_synthetic --bundle artifacts/data/synthetic/features.npz --run bilstm_noaug__seed17 --allow-synthetic --force
python -m src.lsc_models run-app --mode real --open
python -m src.lsc_models check-app                         # en otra terminal, con la app encendida
```
En `check-app` real, las imágenes de prueba (sin persona) deben dar `retry` con razones de calidad
(p. ej. `INSUFFICIENT_POSE`): eso demuestra que el extractor real está activo.
Con una persona frente a la cámara debería llegar a `accepted` o `retry` por baja confianza.

### 6c. Modo real con el modelo verdadero (cuando existan los datos reales)
```bash
python -m src.lsc_models prepare-data            # sección 3
python -m src.lsc_models run-all && python -m src.lsc_models aggregate
python -m src.lsc_models export-artifacts --force   # exporta la ejecución desplegable (semilla 17)
python -m src.lsc_models run-app --mode real --open
```
`export-artifacts` verifica que el modelo exportado reproduce las predicciones guardadas (1e-5), ajusta un
umbral **provisional** con validación (cobertura ≥ 80 %) y deja `requires_confirmation: true`: hasta que
A4 mida el rechazo de negativos, la interfaz debe presentar la salida como **sugerencia a confirmar**.
Rechaza datos sintéticos salvo `--allow-synthetic`.

### Contenedor (Juan) 
`docker compose -f deployment/compose.yaml up --build -d` y `python deployment/check_ready.py --url http://127.0.0.1:8000`
(ver `docs/app/operator-guide.md`). No lo probé aquí.

Si el servidor real no está listo, `/api/v1/health` responde 503 con la causa (faltan archivos, hashes
incompatibles); **nunca cae solo a modo simulado**.

## Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| `No existe artifacts/data/v1/features.npz` | Falta el paso 3. Use `verify` mientras tanto. |
| `--fast ... solo se permite con datos sintéticos` | Intentó acortar el entrenamiento con datos reales; no está permitido. |
| `TestAccessError` | Intentó leer el test antes de `freeze`; es intencional. |
| `ModuleNotFoundError: tensorflow` | Falta `pip install -e ".[train]"` (o entorno sin activar). |
| `pip` no resuelve dependencias | Ver `docs/models/compat.md` (pines de Ortiz incompatibles con TF). |
| `run-app --mode real` dice que faltan archivos | Ejecute `download-models` y `export-artifacts` (6b/6c). |
| `check-app` real da 503 | Falta `.task` en `artifacts/` o hash incompatible: lea el detalle de `/api/v1/health`. |
| `exportación rechazada: ... sintéticos` | Intenta exportar resultados de mentira como modelo real; es intencional. |
