# Metodología de modelado y resultados (Acosta) — BORRADOR ESTRUCTURAL

> Estado: A3 listo para ejecutarse. **Este documento no contiene ninguna métrica todavía.**
> Cada cifra se pega desde `reports/models/*.csv` una vez corridas las 12 ejecuciones
> sobre `features.npz` real. No se escribe una cifra sin su archivo fuente.

## 1. Método (ya fijado; descrito tal como se ejecuta)

- **Entrada:** secuencias de 6 instantes × 49 puntos × (x, y, presencia) = (6, 147), float32,
  normalizadas por hombros (extractor de Ortiz, `features_v1`).
- **Particiones:** por persona (27/4/8 según el reporte de datos); sin fuga entre conjuntos.
- **Modelos:** BiLSTM (proyección densa 64 → BiLSTM 64 → softmax) y Transformer compacto
  (proyección 64, embedding posicional, 2 bloques de 4 cabezas, FF 128, promedio temporal →
  softmax). Aproximadamente 77 mil parámetros cada uno.
- **Presupuesto común:** Adam, lr 1e-3, lote 32, máximo 80 épocas, paciencia 12 sobre F1 macro
  de validación, restauración de los mejores pesos.
- **Aumento (`train_v1`):** rotación global ±5° y ruido gaussiano σ=0,005, solo en entrenamiento.
- **Diseño experimental:** 2 arquitecturas × {sin, con} aumento × semillas 17, 29 y 43 = 12 ejecuciones.
- **Referencia:** clasificador de clase mayoritaria.
- **Selección:** F1 macro medio de validación; diferencia ≤ 0,01 → menor p95 de inferencia;
  se despliega la semilla 17. Regla fijada antes de abrir el test.

## 2. Resultados — PENDIENTE (fuente: `reports/models/`)

| Tabla/figura | Fuente |
|---|---|
| Comparación de las 4 configuraciones (media ± desv.) | `ablation_summary.csv` |
| Resultados por semilla | `per_seed.csv` |
| Resultados por clase y semilla | `per_class_by_seed.csv` |
| Curvas de aprendizaje, confusión | `figures/` |
| Sensibilidad a pérdida de manos y rechazos de calidad | `sensitivity_hand_dropout.csv`, `quality_rejections_val.csv` |
| Cumplimiento de metas (acc ≥ 85 %, F1 ≥ 0,80) | `selection.json` (`goals_vs_validation`) |

Mantener separados: exactitud sobre muestras procesables, desempeño integral (con rechazos de
calidad) y rechazo de desconocidos (A4). Las metas no son resultados.

## 3. Discusión — PENDIENTE

Incluir obligatoriamente: errores por clase, resultados negativos, metas incumplidas, el tamaño
reducido de validación (4 personas) y su efecto sobre la regla de desempate, y la brecha
corpus → cámara cuando A4 aporte la prueba externa.
