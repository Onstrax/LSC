# Reportes de modelos (Acosta)

Salidas reproducibles de A1–A3. **Nada de aquí se escribe a mano**: todo lo genera
`python -m src.lsc_models ...` a partir de lo guardado por cada ejecución.

```
reports/models/
  runs/<config>__seed<N>/     config.json, metrics.json, history.csv,
                              predictions_val.npz, confusion_val.csv,
                              per_class_val.csv, model.keras (git-ignorado)
  ablation_summary.csv        media ± desv. (3 semillas) por configuración
  per_seed.csv                una fila por ejecución (12)
  per_class_by_seed.csv       precisión/recall/F1 por clase y semilla
  selection.json              regla de selección aplicada + semilla desplegada
  sensitivity_hand_dropout.csv, quality_rejections_val.csv
  figures/                    curvas de aprendizaje, confusión, F1 por semilla
  freeze.json                 congelación (habilita el test único de A4)
```

**Resultados sintéticos** (`synthetic_contract_fixture`) llevan `evidence_valid: false`
y no deben copiarse al artículo. No se versionan `runs/*/model.keras` (`.gitignore`).
