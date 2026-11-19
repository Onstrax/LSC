# Hoja de ruta hasta la entrega (Acosta)

Hoy: jueves 19 nov. Exposición: **miércoles 2 dic., 17:30** (10 min + 5 de preguntas). Artículo: **4 dic.**
(hora en Canvas). Marque cada paso solo cuando haya visto la salida real.

## Fase 1 — Cerrar A3 y congelar (hoy; A3 venció el 16, el congelamiento era el 18)
```powershell
python -m src.lsc_models sync-bundle-spec
python -m src.lsc_models run-all
python -m src.lsc_models aggregate
python -m src.lsc_models sensitivity
python -m src.lsc_models freeze
```
Criterio: 12 ejecuciones en `reports/models/runs/`, `selection.json` con `evidence_valid: true`,
`freeze.json` creado. Si congela después del 18 nov., decláralo en el artículo.

## Fase 2 — A4 (19–21 nov.): paquete, test único y umbral
```powershell
python -m src.lsc_models export-artifacts --force      # paquete + umbral provisional (validación)
python -m src.lsc_models final-eval                    # UNA sola vez
```
`final-eval` rechaza cualquier cambio posterior a `freeze` y no se puede repetir. Guarda
`test_evaluation.json` (3 semillas + desplegado + baseline + metas vs. resultado) y marca el paquete.
**Lo que NO se puede automatizar (requiere personas):**
- *Calibración con webcam de desarrollo* y *rechazo de negativos*: necesita capturas reales con participantes
  que consientan. Sin ellas, el umbral queda **provisional** y la interfaz presenta **sugerencias a confirmar**;
  decláralo en el artículo, no lo inventes.
- *Prueba externa* con personas distintas a las del corpus: igual. Si no es viable, repórtalo como limitación
  (el plan prevé este caso: "informar la brecha", "no atribuir validación inexistente").

## Fase 3 — A5 (22–27 nov.): integración y versión estable
1. Probar la app real con tu cámara: `export-artifacts` → `run-app --mode real --open` → `check-app`.
2. Construir el contenedor (nunca probado): `docker compose -f deployment/compose.yaml build` y luego `up`;
   `python deployment/check_ready.py --url http://127.0.0.1:8000`.
3. Instalación limpia: clonar el repo en otra carpeta, entorno nuevo, `pip install -e ".[features,train,app,dev]"`,
   `python -m pytest`, `verify`.
4. Congelar dependencias (lock) en Python 3.11 (`docs/models/compat.md`) y etiquetar: `git tag v1.0-entrega`.
5. Grabar un video breve de respaldo de la demostración (identificado como respaldo).
6. Medir latencia integral (p95 < 2 s) con la app real: la mide `reports/app/` (Juan) — ahora te toca a ti.

## Fase 4 — A6 (22 nov.–4 dic.): artículo y presentación
- Escribe `docs/article/modeling.md` pegando cifras **solo** desde `reports/models/*.csv|json`.
- Ensambla el artículo (máx. 10 págs.): resumen 200–300 palabras, 3–5 palabras clave, método, resultados,
  discusión (errores, resultados negativos, metas incumplidas, tamaño de validación), conclusiones, referencias.
  Verifica correos y afiliaciones; declara el uso de IA generativa.
- Presentación: tus tramos son 3 min (método y resultados) y 1 min (límites y conclusión). Ensayos ≤ 9 min 30 s.
- 2 dic.: PPT en el padlet antes de las 17:30. 4 dic.: artículo por Canvas y padlet.

---

# Cómo usar Claude Code en este proyecto

Claude Code corre en TU equipo: puede ejecutar los comandos anteriores, leer los errores reales (Docker, cámara,
entrenamiento) y corregir el código. Es lo que un asistente sin acceso a tu máquina no puede hacer.

## Instalación (Windows, PowerShell)
```powershell
irm https://claude.ai/install.ps1 | iex       # instalador nativo; abra una PowerShell NUEVA después
cd D:\LSC
claude                                         # inicia sesión la primera vez
```
(Git for Windows es opcional: sin él Claude Code usa PowerShell.) El repo ya trae `CLAUDE.md`, que le da el
contexto y las reglas del proyecto; no necesita `/init`.

## Hábitos que evitan problemas
- Trabaja en una rama: `git checkout -b integracion-final`. Revisa los cambios con `git diff` antes de aceptar.
- Pídele pasos pequeños y que **muestre la salida real** de `python -m pytest` al terminar.
- Dile siempre las reglas críticas (ya están en `CLAUDE.md`): no abrir el test, no inventar datos ni cifras.
- No lo dejes tocar `LSC70/` ni subir datos a Git.

## Prompts listos (pega uno a la vez)
**1. Verificar el entorno**
> Lee CLAUDE.md y docs/models/ROADMAP_FINAL.md. Ejecuta `python -m src.lsc_models verify --with-tests` y
> `python -m pytest`. Muéstrame el resultado real y, si algo falla, diagnostica la causa antes de cambiar nada.

**2. Ejecutar A3 con datos reales**
> Ejecuta la Fase 1 del ROADMAP en orden (sync-bundle-spec, run-all, aggregate, sensitivity). Si una ejecución
> falla, repórtalo; no cambies hiperparámetros ni configs/train/experiments.yaml. Al terminar, resume
> reports/models/ablation_summary.csv y selection.json sin interpretar más allá de los datos.

**3. Congelar, exportar y evaluar el test (una sola vez)**
> Confirma que existen 12 ejecuciones y selection.json con evidence_valid=true. Muéstrame la configuración
> elegida y espera mi confirmación explícita. Si digo "confirmo", ejecuta freeze, export-artifacts --force y
> final-eval UNA sola vez. Si final-eval falla, NO lo repitas ni toques los datos: explícame el error.

**4. Probar la app real y el contenedor**
> Ejecuta export-artifacts si falta, luego `run-app --mode real` en una terminal y `check-app` en otra.
> Después construye el contenedor con docker compose (deployment/compose.yaml); si falla el build, corrige
> el Dockerfile/requirements y vuelve a probar hasta que check_ready.py pase. Muéstrame cada salida.

**5. Redactar el artículo con evidencia**
> Lee reports/models/*.csv|json y docs/article/modeling.md. Redacta metodología, resultados y discusión usando
> SOLO cifras de esos archivos, citando el archivo fuente de cada tabla. Incluye metas incumplidas, errores por
> clase y limitaciones. No inventes referencias: si falta una, déjala marcada como [PENDIENTE].

**6. Instalación limpia y versión estable**
> Clona el repo en una carpeta temporal, crea un entorno nuevo, instala, ejecuta `python -m pytest` y `verify`.
> Si todo pasa, crea el tag v1.0-entrega y lista los comandos para reproducir las tablas del artículo.
