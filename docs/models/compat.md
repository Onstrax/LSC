# Prueba de compatibilidad inicial de dependencias (Acosta)

Entorno de la prueba: Python 3.12.3, Linux, CPU. **Pendiente repetir en Python 3.11**
(objetivo del diseño) y en el equipo de exposición antes de congelar el lock.

## Combinación que funciona (verificada: `pip check` limpio y suites en verde)

| Paquete | Versión probada |
|---|---|
| tensorflow-cpu / keras | 2.21.0 / 3.15.1 |
| numpy | 2.5.4 |
| mediapipe | 0.10.33 |
| scikit-learn / matplotlib | 1.9.1 / 3.11.2 |
| pydantic / PyYAML | 2.14.3 / 6.0.3 |

Suites ejecutadas ahí: `tests/contracts tests/data tests/features tests/app` →
128 passed, 1 skipped (el skip es la comprobación de Python 3.11);
`tests/models tests/inference` → 52 passed.

## Conflictos encontrados (resolver por issue, no unilateralmente)

`deployment/requirements-data.in` (módulo de Ortiz) fija `numpy==2.2.3` y
`mediapipe==0.10.21`. Con TensorFlow esa combinación **no se puede resolver**
(`pip install --dry-run`):

1. `mediapipe==0.10.21` exige `protobuf<5`; `tensorflow-cpu 2.21.0` exige `protobuf>=6.31.1`.
2. `mediapipe==0.10.21` exige `numpy<2`, lo que contradice el propio `numpy==2.2.3`.

Además `contracts/feature_spec.json` registra `mediapipe-tasks:1.2.2` como versión del
detector, que no coincide con ninguna de las versiones anteriores. Esa cadena entra en el
`feature_spec_hash`, así que **no se toca** sin que Ortiz confirme si cambia o no la semántica.

**Solicitud a Ortiz:** confirmar con qué versión de mediapipe se generó `features.npz` y si
`0.10.33` produce los mismos `x` (prueba de paridad sobre algunas muestras). Mientras tanto
el pyproject raíz declara rangos (`mediapipe>=0.10.21`) en vez de pines.

## Para cerrar el lock (antes del ensayo final, 27 nov.)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install pip-tools
pip-compile --extra features --extra train --extra app --extra dev -o requirements-lock.txt pyproject.toml
pip install -r requirements-lock.txt && python -m pytest -q
```

## Defecto del contrato de Ortiz: hash de `hand_landmarker.task` de 65 caracteres

`contracts/feature_spec.json` (y `configs/data/features_v1.yaml`) registran
`model_asset_hashes.hands = fbc2a30080c3…3607cde1` con **65** caracteres hexadecimales; un SHA-256
tiene 64. El reporte de Ortiz cita `fbc2a300…607cde`, que coincide con los primeros 64, así que el
"1" final parece sobrante. El extractor no verifica ese hash en ejecución, por eso nada falla hoy.

No se corrige desde este módulo: el hash entra al `feature_spec_hash`, y cambiarlo invalida el
contrato y todo lo que dependa de él. **Solicitud a Ortiz:** confirmar con `Get-FileHash` (o
`sha256sum`) el hash real del `.task`, corregir ambos archivos y recomputar `feature_spec_hash`.
`prepare-data` y `verify` lo señalan como AVISO y no bloquean.

## Incompatibilidad Ortiz ↔ Juan: formato de `labels.json`

`contracts/labels.json` (Ortiz) guarda `labels` como **lista de objetos**
(`{class_id, code, display_text, raw_label}`), pero `api/config.py::load_labels` (Juan) espera una
**lista de textos**. Con el archivo del contrato la API en modo real devolvería objetos como
`display_text`. Comprobado con `LSC_MODE=real` y el archivo copiado tal cual.

Solución aplicada dentro del módulo de Acosta (sin tocar archivos ajenos): `export-artifacts` escribe
`artifacts/labels.json` con `labels` = textos (para la API) y `entries` = los objetos del contrato.

**Solicitud a Juan (issue):** que `load_labels` acepte también objetos
(`l["display_text"] if isinstance(l, dict) else l`) y verifique el orden por `class_id`.
