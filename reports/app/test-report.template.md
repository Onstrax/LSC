# Informe de pruebas · plantilla

Rellenar con **ejecuciones reales**. Cada fila apunta a un comando y su fecha.

## Pruebas automáticas

| Suite | Comando | Fecha | Resultado | Incidencias |
| --- | --- | --- | --- | --- |
| API y contrato (A01–A14) | `python -m pytest tests/app -q` | _fecha_ | _N passed_ | _ref_ |
| Latencia (percentiles) | `python -m pytest tests/app/test_latency.py -q` | _fecha_ | _N passed_ | _ref_ |
| Accesibilidad estática | `python -m pytest tests/app/test_accessibility_static.py -q` | _fecha_ | _N passed_ | _ref_ |
| Lógica JS (Node) | `node --test tests/e2e/js_unit.mjs` | _fecha_ | _N passed_ | _ref_ |
| Interfaz e2e (Playwright) | `python -m pytest tests/e2e -q` | _fecha_ | _requiere navegador_ | _ref_ |

## Pruebas manuales (requieren persona)

| Caso | Descripción | Fecha | Resultado | Notas |
| --- | --- | --- | --- | --- |
| Prueba física de lados | Marcador izquierda/derecha: bytes sin espejo | _fecha_ | _pendiente_ | obligatoria |
| Permiso real de cámara | `getUserMedia` con permiso del navegador | _fecha_ | _pendiente_ | — |
| Recorrido con teclado | Flujo completo sin ratón, foco visible | _fecha_ | _pendiente_ | — |
| Zoom 200 % y 360 px | Acciones esenciales visibles | _fecha_ | _pendiente_ | — |
| Sesión prolongada (20 intentos) | Memoria estable, sin acumulación | _fecha_ | _pendiente_ | — |

## Incidencias abiertas

- _Listar aquí los problemas conocidos y su estado._

> Una suite sintética en verde no certifica reconocimiento ni usabilidad humana.
