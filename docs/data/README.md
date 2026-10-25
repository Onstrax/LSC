# Módulo de datos `src/lsc_data` (Ortiz)

Paquete de datos sin red: reconstrucción de secuencias, manifiesto, inventario,
splits deterministas sin fuga, adquisición segura, empaquetado tipado sin pickle,
validación y CLI. No importa `cv2`/`mediapipe`/`PyYAML`/`numpy` en el import del
paquete; esas dependencias se cargan de forma perezosa sólo cuando una función
las necesita.

## Layout `src/` y cómo invocar la CLI

El paquete vive bajo `src/` (layout `src/`). Por eso hay dos formas equivalentes
de invocar la CLI desde la raíz del repositorio:

```
# Opción A: ejecutar el paquete por su ruta bajo src/
python -m src.lsc_data --help

# Opción B: poner src/ en PYTHONPATH y usar el nombre del paquete
#   PowerShell:  $env:PYTHONPATH="src"; python -m lsc_data --help
#   bash:        PYTHONPATH=src python -m lsc_data --help
```

El import de librería funciona como `import src.lsc_data` desde la raíz; con
`PYTHONPATH=src` también funciona `import lsc_data`.

## Subcomandos (códigos de salida uniformes 0/2/3)

- `acquire --config configs/data/source.yaml [--archive <ruta>] [--dest <dir>]`
  — sin `--archive` NO descarga (registra el bloque DO02) y sale 0; con
  `--archive` valida procedencia (SHA-256, tamaño) y extrae de forma segura.
- `inventory --raw-dir <dir> --out reports/data/inventory.csv`
- `manifest --config configs/data/lsc70w.yaml --raw-dir <dir> --out <manifest.jsonl>`
- `split --manifest <manifest.jsonl> [--seed 20261022] [--out splits.json]`
- `features --config configs/data/features_v1.yaml --manifest <...> --splits <...> --out <dir> [--resume]`
  (construcción sobre corpus real: bloque DO07; `--resume` aborta con exit 3 si
  `feature_spec_hash` o `manifest_sha256` del bundle no coinciden)
- `validate --bundle <dir>`

Códigos: `0` éxito/conforme, `2` entrada/config inválida, `3` fallo de
integridad (fuga, rutas peligrosas, hash divergente, formas/cobertura rotas).

## Suite de pruebas

Las pruebas de Ortiz (`tests/data`, `tests/features`, `tests/contracts`, de
FEAT-004) se recolectan con paths explícitos, que anulan el `testpaths` del
`pytest.ini` raíz (de Juan) sin editarlo:

```
python -m pytest tests/data tests/features tests/contracts
```
