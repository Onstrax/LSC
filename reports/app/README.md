# Evidencias de la aplicación (latencia y usabilidad)

Este directorio contiene las evidencias técnicas y de rendimiento del módulo de
Juan. **No contiene imágenes personales ni identidades.** Los registros de
participantes del piloto sólo se conservan bajo su consentimiento y con un
identificador opaco.

## Archivos

| Archivo | Contenido |
| --- | --- |
| `latency_attempts.template.csv` | Encabezado del registro por intento (unidades en las columnas). |
| `usability_tasks.template.csv` | Encabezado del registro de tareas de usabilidad. |
| `analyze_latency.py` | Calcula `latency_summary.json` con percentiles por rango más próximo. |
| `latency_attempts.csv` | (Generado en el ensayo/piloto) registros reales por intento. |
| `latency_summary.json` | (Generado) métricas con N, p50, p95, media, errores, timeouts, rechazos. |
| `usability_tasks.csv` | (Generado en el piloto) tareas, ayuda, éxito, duración, incidencias. |

## Definiciones (SDD §16)

- **Tiempo integral**: desde el fin de la ventana de captura (antes de
  seleccionar y codificar JPEG) hasta que el resultado queda presentado en el
  navegador (marca tras el siguiente ciclo de pintura). Se mide con
  `performance.now()` en ambos extremos del mismo navegador.
- **Percentil**: rango más próximo. Ordenar y tomar la posición `ceil(p·N)`,
  con posiciones desde 1.
- **Timeouts**: se reportan **censurados** con su plazo; nunca como 0 ms ni como
  observaciones completas.
- **Metas**: `p95 < 2 s` (de `accepted`, fin de captura a sugerencia visible) y
  al menos 95 % de intentos válidos sin fallo técnico. `accepted` y `retry` por
  calidad o confianza son respuestas funcionales; `error` técnico y `timeout`
  son fallos. La exactitud y la cobertura se calculan aparte con Acosta.
- **Protocolo de medición**: al menos 100 intentos válidos tras 5 calentamientos
  separados. No repetir hasta conservar sólo tiempos rápidos.

## Uso

La interfaz exporta `latency_attempts.csv` con el botón "Exportar latencias".
Luego:

```bash
python reports/app/analyze_latency.py reports/app/latency_attempts.csv
```

El arranque en frío se mide por separado, desde iniciar el servicio hasta
`readiness` real.
