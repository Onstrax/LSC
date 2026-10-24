# Guía de recuperación durante la demostración

Plan para incidencias en vivo. El objetivo es no perder los cuatro minutos de
aplicación. El respaldo grabado **siempre** se identifica como grabado.

## Antes de empezar

1. Dejar el servicio arrancado y verificado con `check_ready.py` (modo real).
2. Tener abierta una pestaña en `http://localhost:8000` ya con la cámara
   activada y encuadre probado.
3. Tener el video de respaldo listo en una segunda pestaña o archivo local.

## Árbol de decisión

| Síntoma en vivo | Acción inmediata | Si no se resuelve en ~15 s |
| --- | --- | --- |
| La cámara no activa | Revisar permiso del navegador y reintentar | Pasar al video de respaldo (rotulado) |
| Respuesta `retry` repetida | Mejorar encuadre (manos visibles), repetir | Mostrar un `retry` como ejemplo de robustez y continuar |
| Error 503 | Verificar `check_ready.py` en una terminal | Reiniciar contenedor; mientras tanto, usar respaldo |
| Error 429 | Esperar (hay inferencia activa) | Reintentar una vez; si persiste, respaldo |
| Error 504 | El trabajo superó el plazo | Reiniciar contenedor; usar respaldo |

## Reinicio rápido del servicio

```bash
docker compose -f deployment/compose.yaml restart
python deployment/check_ready.py
```

## Narración del respaldo

Si se usa el video grabado, decir explícitamente: “Esto es una grabación
autorizada de una sesión previa en modo real”. No presentar el respaldo como
ejecución en vivo.

## Qué no hacer

- No cambiar a modo simulado para “arreglar” la demo: el modo real debe ser
  visible.
- No ajustar umbrales ni capturas para forzar una sugerencia.
- No afirmar métricas que no provengan de `reports/app/`.
