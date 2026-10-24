# Guion de demostración · Módulo de aplicación (Juan)

Exposición grupal: 2 de diciembre de 2026, 17:30, diez minutos + cinco de
preguntas. La parte de aplicación ocupa **cuatro minutos** dentro de los diez
del grupo. Ensayo grupal de máximo 9 min 30 s. El equipo sube el PPT antes de la
exposición.

> Regla: el **modo real debe ser visible** durante la demo; el respaldo grabado
> se identifica como tal. Una IA no afirma haber enviado el PPT sin evidencia.

## Estructura (4 minutos)

| Tiempo | Contenido | Nota |
| --- | --- | --- |
| 0:00–0:30 | Problema y alcance: una seña aislada, 10 clases, uso local | Mostrar vocabulario en pantalla |
| 0:30–1:00 | Arquitectura: navegador → FastAPI → extractor (Ortiz) + modelo (Acosta) | Diagrama de una lámina |
| 1:00–2:30 | Demo en vivo (modo real): activar cámara, capturar, sugerencia, corregir, repetir | Verificar `check_ready.py` antes |
| 2:30–3:15 | Estados: mostrar un `retry` (p. ej. sin manos) y la recuperación de error | Guion de recuperación |
| 3:15–4:00 | Latencia y límites: p95 medido, metas, qué no demuestra el piloto | Cifras desde `reports/app/` |

## Puntos a enfatizar

- La sugerencia **no** es una verificación de significado; el vocabulario está
  limitado a diez señas.
- El sistema separa sugerencia y corrección; no reentrena con el historial.
- La medición de latencia es de extremo a extremo (fin de captura a texto
  visible), no solo el tiempo de red.

## Lista de verificación previa (equipo de exposición)

- [ ] Artefactos montados en `./artifacts` (modelo + `.task`).
- [ ] `docker compose -f deployment/compose.yaml up --build -d` ejecutado.
- [ ] `python deployment/check_ready.py` devuelve LISTO en modo real.
- [ ] Cámara del navegador funcionando en `http://localhost:8000`.
- [ ] Sin dependencia de Internet para la inferencia.
- [ ] Video de respaldo disponible y rotulado como grabado.
- [ ] PPT subido (con evidencia del envío).

## Preguntas probables y respuestas breves

- **¿Reconoce cualquier seña?** No; solo las diez de LSC70W. Fuera de ese
  conjunto, la sugerencia no es fiable.
- **¿Guarda las imágenes?** No en operación ordinaria; se procesan de forma
  temporal.
- **¿Funciona con nuevos usuarios?** La evaluación por señantes no vistos y el
  piloto externo lo documentan; es una evidencia exploratoria.
