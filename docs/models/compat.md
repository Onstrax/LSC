# Compatibilidad e integración (Acosta, integrador)

Estado: **todos los problemas de integración detectados están corregidos en el repo.** Este archivo
documenta qué se encontró, qué se cambió y qué queda sin verificar.

## Combinación probada

| Paquete | Probado en | Versión |
|---|---|---|
| tensorflow-cpu / keras | Windows py3.12 (Acosta) y Linux py3.12 | 2.21.0 / 3.15.1 |
| numpy | idem | 2.5.x |
| mediapipe | Linux py3.12 / resolución para py3.11 | 0.10.33 / 0.10.35 |
| scikit-learn / matplotlib | idem | 1.9.1 / 3.11.2 |

Para **Python 3.11** (imagen Docker) solo se verificó que el conjunto de dependencias **se resuelve**
(`pip install --dry-run --python-version 3.11`); la suite completa se ejecutó en 3.12.

## Correcciones aplicadas

1. **Hash de `hand_landmarker.task` en el contrato (Ortiz).** Tenía 65 caracteres: una "e" repetida
   (`…044cce[e]e322…`). El valor real (observado al descargar el `.task` oficial, 64 caracteres) es
   `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`. Se corrigió en
   `contracts/feature_spec.json` y `configs/data/features_v1.yaml`, y se **recomputó** `feature_spec_hash`:
   `08779533e99c…` → `0fdfe6b2b954…`. `tests/models/test_migrate.py` demuestra que el hash viejo se
   reproduce exactamente con el valor erróneo, es decir, que el cambio no oculta otra diferencia.
   - Los `.task` y por tanto las características **no cambian**: no hay que reprocesar. Para alinear un
     bundle ya generado: `python -m src.lsc_models sync-bundle-spec` (guarda respaldo y se niega a tocar
     nada que no sea exactamente la diferencia conocida).
   - Los reportes históricos (`reports/data/`, `docs/data/`, `.agents/`) conservan el valor antiguo como registro.
2. **Pines irresolubles de `requirements-data.in` (Ortiz).** `mediapipe==0.10.21` exigía `protobuf<5` y
   `numpy<2`, incompatibles con TensorFlow (`protobuf>=6.31`) y con su propio `numpy==2.2.3`. Ahora son rangos.
3. **`labels.json` (Juan).** `api/config.py::load_labels` solo aceptaba textos; el contrato de Ortiz guarda
   objetos y la API habría mostrado objetos como `display_text`. Ahora acepta ambos, ordena por `class_id`
   y **exige el orden canónico** (antes un orden distinto etiquetaba mal sin avisar).
4. **Imagen Docker (Juan/Acosta).** No instalaba TensorFlow ni MediaPipe ni copiaba `src/`/`contracts/`, pero
   arrancaba en modo `real`: no habría podido servir el modelo. Reescrita; se instala `.[serve]`.
5. **`.dockerignore`/`.gitignore`.** No excluían `LSC70/`: el contexto de build y `git add .` habrían incluido
   las imágenes del corpus.
6. **`pytest.ini`** unificado: `python -m pytest` ejecuta todas las suites.
7. **Reparto** en `AGENTS.md` y `README.md`: 30/30/40 → 33/33/34 (acuerdo vigente).
8. **`origin_guard` de la API (Juan) — la interfaz no cargaba fuera del puerto 8000.** Chromium envía `Origin`
   al cargar `main.js` (script de módulo) y la lista fija de orígenes (`:8000`) devolvía **403**: sin JavaScript no hay
   interfaz. Las pruebas e2e (puerto aleatorio) fallaban por esto. Ahora se acepta el `Origin` si está en la lista **o es
   el propio servicio** (`Origin` = autoridad de la cabecera `Host`); `Host` se sigue validando y orígenes de otros
   sitios o puertos siguen bloqueados (`tests/app/test_origin_guard.py`).
9. **`tests/e2e/conftest.py`**: si falta el binario de Chromium se omite con un mensaje, en lugar de fallar.

## Verificado vs. no verificado

Verificado: suite completa (incluidas las 7 pruebas e2e en **Chromium real con cámara simulada**); servicio en modo real
con modelo exportado y extractor simulado; cierre de importaciones usando solo las carpetas que copia la imagen;
construcción del paquete (`pip wheel`).

**No verificado** (no hay Docker, cámara física ni los `.task` en el entorno de desarrollo del asistente):
- `docker compose build/up` (la imagen no se ha construido nunca);
- la cámara física y la prueba de lados (espejo) con una persona real; la cámara simulada no las certifica;
- el servicio en modo real con **MediaPipe real** (necesita los `.task`; lo prueba el equipo con `check-app`).

## Pendiente de decisión (no bloquea)

`feature_spec.json` declara `detector_versions: mediapipe-tasks:1.2.2`, que no corresponde a una versión
de paquete verificable. Es una etiqueta descriptiva; no se cambió porque entra al `feature_spec_hash`.
Si se decide corregirla: editar, recomputar el hash y repetir `sync-bundle-spec` (ajustando `migrate.py`).
