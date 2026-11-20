@AGENTS.md

# Contexto para Claude Code — Proyecto LSC (reconocedor de 10 señas aisladas, LSC70W)

Usuario: Daniel Alejandro Acosta Avila (integrador). Es dueño de `src/lsc_models/`, `src/lsc_inference/`,
`configs/train/`, `reports/models/`, `pyproject.toml`, CI raíz, y **autoriza corregir cualquier módulo** si
bloquea la aplicación final (Ortiz y Juan no harán más cambios). Fechas: exposición 2 dic. 17:30;
artículo 4 dic. Plan completo: `docs/models/ROADMAP_FINAL.md`. Guía de comandos: `docs/models/INSTRUCCIONES.md`.
Responde en español.

## Comandos (siempre desde la raíz; Windows PowerShell)
- Verificación rápida del código (sin datos reales): `python -m src.lsc_models verify`
- Suite completa: `python -m pytest` (≈1–2 min; e2e se omite sin Playwright)
- Usa siempre `.\.venv\Scripts\python.exe para ejecutar comandos.`
- Flujo con datos reales: `prepare-data` → `sync-bundle-spec` → `run-all` → `aggregate` → `sensitivity` →
  `freeze` → `export-artifacts` → `final-eval` (todos con `python -m src.lsc_models <comando>`)
- App: `python -m src.lsc_models run-app --mode mock|real --open`; comprobar: `check-app`

## Reglas que NO se negocian
1. **El split `test` se abre UNA sola vez**, con `final-eval`, después de `freeze`. Nunca ajustar nada con él.
   Si algo cambia las predicciones después de `freeze`, hay que declararlo (prueba intacta invalidada).
2. **No inventar datos, métricas, pesos, participantes ni consentimientos.** Si falta un archivo, decirlo.
   Los datos `synthetic_contract_fixture` nunca son evidencia y nunca van al artículo.
3. Cada cifra del artículo debe salir de un archivo de `reports/models/` (reproducible).
4. Separar siempre: exactitud sobre procesables, desempeño integral (rechazos cuentan como error) y rechazo
   de negativos. Las metas (acc ≥ 0,85; F1 macro ≥ 0,80; p95 < 2 s) no son resultados; informar el resultado real.
5. La probabilidad softmax no es certeza: la interfaz muestra "sugerencia" y el usuario confirma.
6. Nunca subir a Git el corpus (`LSC70/`), `artifacts/data`, `*.keras` ni imágenes de personas.
7. Antes de dar algo por terminado: correr `python -m pytest` y mostrar el resultado real.

## Estado conocido
- Corregido: hash del contrato (`feature_spec_hash` = 0fdfe6b2b954…), `labels.json`, `origin_guard`, Dockerfile,
  pines de dependencias. Ver `docs/models/compat.md`.
- **Modelo por defecto de la app = v2** (`artifacts_v2/`, `docs/models/v2_robusto.md`), desde el 20 nov. por decisión del integrador y
  antes de medir su exactitud etiquetada en webcam. v1 sigue congelado en `artifacts/` (cifra de test: exactitud 0,870, abierta UNA vez).
  v2 tiene el test SELLADO: no hay cifra de test y NO se ejecuta `final-eval`/`freeze` con v2. `export-artifacts` y `final-eval`
  siguen apuntando a v1 por defecto.
- Verificado (nov. 2026, sesión del usuario): cámara física y MediaPipe real, fuera del contenedor.
- Sin verificar nunca: `docker compose build/up`; MediaPipe real dentro del contenedor.
