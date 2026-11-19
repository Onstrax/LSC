# Guía A1–A3: cómo reproducir las 12 ejecuciones

Prerrequisito: `artifacts/data/v1/features.npz` entregado por Ortiz (no se versiona).
**Sin ese archivo el código se niega a correr**; no inventa datos.

```bash
pip install -e ".[features,train,dev]"

# 1) Las 12 ejecuciones (4 configuraciones x semillas 17/29/43). Reanudable.
python -m src.lsc_models run-all

# 2) Tablas, selección y gráficas desde lo guardado
python -m src.lsc_models aggregate

# 3) Sensibilidad a fallos del extractor (solo validación)
python -m src.lsc_models sensitivity

# 4) 18 nov.: congelar decisiones (habilita el test único de A4)
python -m src.lsc_models freeze
```

Para probar la tubería SIN el corpus (solo interfaces, nunca evidencia):

```bash
python -m src.lsc_models synthetic-fixture --out /tmp/syn.npz
python -m src.lsc_models run-all --bundle /tmp/syn.npz --out /tmp/syn_out
```

## Reglas que el código hace cumplir

- Test reservado: `Bundle.load_split("test")` falla (`TestAccessError`) salvo `allow_test=True`
  **y** `reports/models/freeze.json` existente. Ninguna ejecución de A3 lo lee (`split_used: [train, val]`).
- Entrenamiento solo con muestras procesables (`quality_pass`); la validación se informa de dos
  modos: procesables e integral (un rechazo de calidad cuenta como no acertado).
- Mismo presupuesto para ambas arquitecturas (`training:` único en `experiments.yaml`,
  ~77 mil parámetros cada una; el recuento queda en `metrics.json`).
- Aumento `train_v1` (Ortiz) solo sobre train, por época, con semilla derivada de (semilla, época, índice).
- Reanudación segura: una ejecución solo se reutiliza si su `config_hash` coincide.
- Resultados sobre datos sintéticos llevan `evidence_valid: false`, no se comparan con las metas
  y `freeze` los rechaza.

## Qué NO hace A3 (queda en A4)

Test único, umbral de rechazo, negativos, `thresholds.json`, `preprocessing.json`,
`model_card.md`, `manifest.json` y el paquete del modelo.

## Límites a tener presentes al leer los resultados

- La validación son 7 personas (70 secuencias, 63 procesables): una secuencia equivale a ~0,016 de
  exactitud, y 0,01 de F1 macro es menos que una secuencia. La regla de desempate por latencia se activará
  con frecuencia y las diferencias pequeñas entre configuraciones son ruido. Decirlo así en la discusión.
  Con solo 3 semillas, reporte siempre media ± desviación y los valores por semilla.
- `p95_latency_ms` es solo de inferencia del modelo (lote 1). La meta de p95 < 2 s es de
  extremo a extremo y la mide Juan (J4).
