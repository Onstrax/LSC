# Fixtures sintéticos — contracts/fixtures

**TODO lo que hay en este directorio es SINTÉTICO.** No proviene del corpus real
de Mendeley (LSC70W/LSC70AN/LSC70ANH) ni de capturas de personas. No contiene
imágenes, nombres personales ni copias de consentimiento. Los `sample_id`,
`signer_id`, hashes de frame y rutas son valores fabricados para ejercitar los
contratos y las pruebas sin red, sin corpus y sin archivos `.task`.

Convención de nombres: todo fixture sintético lleva el prefijo `SYNTHETIC_`
para separarlo por nombre y por ruta de cualquier artefacto de datos reales
(que viven bajo `artifacts/data/`, git-ignorado).

## Archivos

- `SYNTHETIC_valid_sample.jsonl` — una `ManifestRecord` válida sintética.
- `SYNTHETIC_invalid_samples.jsonl` — casos inválidos rotulados por `reason_code`
  (un objeto JSON por línea: `{ "reason_code": ..., "record": {...} }`).
- `SYNTHETIC_splits.json` — un `SplitDoc` sintético mínimo.
- `SYNTHETIC_feature_result_6x147.npz` — `x` float32 `(6, 147)`, consumible sin
  corpus. Generado con `np.savez` (sin pickle); se lee con
  `np.load(..., allow_pickle=False)`.

## Regenerar `SYNTHETIC_feature_result_6x147.npz`

El `.npz` se regenera de forma determinista con numpy (sin pickle, sin object
arrays). Ejecutar desde la raíz del repo:

```python
import numpy as np

rng = np.random.default_rng(20261022)
# 49 puntos x (x, y, m) = 147 por cuadro; 6 cuadros.
x = np.zeros((6, 147), dtype=np.float32)
coords = rng.uniform(-1.0, 1.0, size=(6, 49, 2)).astype(np.float32)
x[:, 0::3] = coords[:, :, 0]       # canal x
x[:, 1::3] = coords[:, :, 1]       # canal y
x[:, 2::3] = np.float32(1.0)       # canal m (máscara de presencia = 1)

np.savez(
    "contracts/fixtures/SYNTHETIC_feature_result_6x147.npz",
    x=x,
)
```

El array resultante cumple `x.shape == (6, 147)` y `x.dtype == np.float32`.
