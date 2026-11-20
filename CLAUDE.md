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
   **Excepción declarada (20 nov. 2026):** el integrador autorizó un SEGUNDO uso del test, para v2
   (`reports/models_v2/test_authorization.json`). Ya se usó; **no hay un tercer uso**: no ejecutar `final-eval` otra vez,
   ni con v1 ni con v2, y no ajustar nada con el test. El artículo debe declarar los dos usos.
2. **No inventar datos, métricas, pesos, participantes ni consentimientos.** Si falta un archivo, decirlo.
   Los datos `synthetic_contract_fixture` nunca son evidencia y nunca van al artículo.
3. Cada cifra del artículo debe salir de un archivo de `reports/models/` o `reports/models_v2/` (reproducible).
4. Separar siempre: exactitud sobre procesables, desempeño integral (rechazos cuentan como error) y rechazo
   de negativos. Las metas (acc ≥ 0,85; F1 macro ≥ 0,80; p95 < 2 s) no son resultados; informar el resultado real.
5. La probabilidad softmax no es certeza: la interfaz muestra "sugerencia" y el usuario confirma.
6. Nunca subir a Git el corpus (`LSC70/`), `artifacts/data`, `*.keras` ni imágenes de personas.
7. Antes de dar algo por terminado: correr `python -m pytest` y mostrar el resultado real.

## Estado conocido
- Corregido: hash del contrato (`feature_spec_hash` = 0fdfe6b2b954…), `labels.json`, `origin_guard`, Dockerfile,
  pines de dependencias. Ver `docs/models/compat.md`.
- **Modelo OFICIAL y por defecto de la app = v2** (`artifacts_v2/`, `reports/models_v2/`, `docs/models/v2_robusto.md`), desde el 20 nov.
  por decisión del integrador. Test (131 procesables): v2 exactitud 0,931 / F1 0,931 (3 semillas: 0,921 ± 0,009); v1 0,870 / 0,872
  (3 semillas: 0,878 ± 0,008). Comparación pareada: p = 0,021 (`reports/models_v2/paired_test_comparison.json`). v1 queda como línea base
  congelada en `artifacts/`. `export-artifacts` y `final-eval` siguen apuntando a v1 por defecto (no pisar v1 por accidente).
- Pendiente de medir: exactitud etiquetada en webcam de v2 (falta el orden real de la sesión), rechazo de negativos, personas ajenas al corpus.
- Verificado (nov. 2026, sesión del usuario): cámara física y MediaPipe real, fuera del contenedor.
- Verificado (20 nov. 2026): `docker compose build/up` con el modelo v2 montado; MediaPipe real dentro del contenedor; `check_ready.py` LISTO y
  `check-app` 8/8 contra el contenedor (predicción ≈ 0,2 s en régimen). Para eso hubo que añadir `libgles2`/`libegl1` al Dockerfile y corregir
  el `.dockerignore` (una prueba lo vigila).
- Verificado (20 nov. 2026): **instalación limpia en Python 3.11.18** (contenedor python:3.11-slim, Linux): `pip install -e ".[features,train,app,dev]"`
  sin conflictos (`pip check`), 305 pruebas aprobadas (29 omitidas por falta de Node en el contenedor) y `verify` 10/10. Lock: `deployment/requirements-lock-py311.txt`.
  La imagen Docker usa mediapipe 0.10.35 y el desarrollo 1.2.3; se midió que las características coinciden (|Δ| máx 0,0005) y las predicciones también.
  Sigue sin verificarse: Python 3.11 nativo en Windows (este equipo tiene una entrada huérfana de 3.11 en `C:\Python311`, que no existe) y la cámara física a través del contenedor.
