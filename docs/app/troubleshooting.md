# Resolución de fallos

Guía técnica de diagnóstico para el operador y el desarrollador. Para el usuario
final, ver [`user-guide.md`](user-guide.md).

## El servicio no arranca o `check_ready.py` devuelve 503

1. Confirme que `./artifacts` contiene todos los archivos (ver
   [`../../artifacts/README.md`](../../artifacts/README.md)).
2. Revise el diagnóstico en los logs del contenedor:
   ```bash
   docker compose -f deployment/compose.yaml logs --tail 50
   ```
3. Causas típicas:
   - **Falta `model.keras`, `labels.json`, un `.task`, etc.** → el servicio
     queda no disponible (no cambia a mock).
   - **`feature_spec_hash` incompatible entre extractor y modelo** → readiness
     503 con `CONFIG_INCOMPATIBLE`. Compare versiones con Acosta y Ortiz.
   - **`thresholds.json` sin `policy_version`** → la política no está versionada;
     solicite la versión a Acosta.

## La cámara no se activa

- **Permiso denegado**: revise los permisos del sitio en el navegador y vuelva a
  pulsar activar.
- **Contexto no seguro**: use `http://localhost:8000` servido por la aplicación.
  No abra `index.html` directamente desde el disco.
- **Cámara en uso**: cierre otras aplicaciones que usen la cámara.
- **“Navegador no compatible”**: falta `requestVideoFrameCallback`. Use el
  navegador certificado.

## Respuestas inesperadas del endpoint

| HTTP | Diagnóstico |
| --- | --- |
| 400 | `metadata` no es JSON válido o el multipart está mal formado. |
| 413 | El cuerpo o un JPEG excede el límite (6 MiB / 512 KiB). |
| 415 | Algún archivo no es JPEG. |
| 422 | La metadata viola el contrato (cantidad, dimensiones, índices, timestamps, `mirrored`). |
| 429 | Hay una inferencia activa. Normal bajo concurrencia. |
| 503 | Servicio cargando/degradado/incompatible. |
| 504 | El trabajo superó el plazo del servidor (10 s por defecto). |
| 500 | Salida del modelo inválida o excepción inesperada. |

## Captura: menos de seis cuadros

Si la cámara entrega pocos cuadros por segundo o la ventana es muy corta, la
captura puede quedar con `N < 6`. La interfaz pide repetir **sin rellenar** con
cuadros replicados. Mejore la iluminación, acerque la cámara o verifique la
tasa de cuadros real (el navegador no garantiza 30 FPS).

## Latencia alta

- Mida con el protocolo de [`../../reports/app/README.md`](../../reports/app/README.md):
  al menos 100 intentos válidos tras 5 calentamientos.
- Revise `server_total` y las etapas (`decode`, `extraction`, `inference`,
  `decision`) en `timings_ms` para localizar el cuello de botella.
- Recuerde: la meta `p95 < 2 s` es independiente de los plazos de recuperación
  (10 s de servidor, 12 s de UI).

## Memoria creciente en sesiones largas

La interfaz libera buffers, blobs y URLs tras cada envío, cancelación o error, y
limita el historial a 20 entradas sin imágenes. Si observa crecimiento
sostenido, verifique que no haya pestañas duplicadas capturando a la vez y
revise la consola del navegador.

## Diagnóstico desde la terminal

```bash
# Salud y versiones
curl -s http://127.0.0.1:8000/api/v1/health | python -m json.tool

# Configuración pública para la UI
curl -s http://127.0.0.1:8000/api/v1/config | python -m json.tool
```

> Las peticiones CLI locales sin `Origin` se permiten para pruebas; el acceso
> desde un navegador exige un `Origin` del conjunto local permitido.
