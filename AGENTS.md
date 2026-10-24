# Acuerdo de propiedad y reparto — Proyecto LSC

Reparto aprobado **30 / 30 / 40** (ver propuesta Fase 1). Este archivo fija qué
rutas pertenecen a cada responsable para evitar colisiones y dobles contratos.

## Responsables

| Persona | Código de estudiante | Parte | Entregable principal |
| --- | --- | --- | --- |
| Juan Esteban Álvarez García | 202212030 | 30 % | Aplicación web local (UI + API + despliegue + uso) |
| Daniel Felipe Ortiz Vallejo | 202221234 | 30 % | Extractor de características, contratos de datos, consentimiento, revisión lingüística |
| Daniel Alejandro Acosta Avila | 202011236 | 40 % | Modelo, probabilidades, política de aceptación, versiones, CI raíz |

## Propiedad de rutas

| Ruta | Propietario | Regla |
| --- | --- | --- |
| `app/`, `api/`, `deployment/` | Juan | Interfaz, servicio, contenedor |
| `tests/app/`, `tests/e2e/` | Juan | Pruebas de API, UI y recorrido integral |
| `docs/app/`, `docs/article/deployment.md`, `docs/presentation/` | Juan | Documentación de uso, sección del artículo, demo |
| `reports/app/` | Juan | Evidencias de latencia y usabilidad (sin imágenes personales) |
| `api/schemas/`, `api/fixtures/` | Juan | Propuesta del contrato HTTP y respuestas sintéticas |
| `contracts/`, `tests/contracts/` | **Ortiz** | Contrato de datos y su verificación |
| `src/lsc_features/` | **Ortiz** | Matemática del extractor |
| `src/lsc_inference/`, artefactos del modelo | **Acosta** | Predictor y pesos |
| `pyproject.toml`, lockfile, CI raíz | **Acosta** | Dependencias y CI del repositorio |

## Reglas de colaboración

1. **Contrato HTTP único.** Juan propone el esquema desde `api/schemas/`; Ortiz
   incorpora la versión acordada en `contracts/`, con una prueba que detecte
   divergencias. No se mantienen dos contratos activos distintos.
2. **Dependencias.** Juan entrega `deployment/requirements-app.in` y los
   comandos necesarios; los cambios a `pyproject.toml`, lockfile y CI raíz se
   solicitan a Acosta. No se introduce un segundo entorno de Python incompatible.
3. **Sin fallback silencioso.** El modo por defecto es `real`. Si faltan pesos o
   archivos `.task` en modo real, el servicio permanece no disponible (503);
   nunca cambia automáticamente a `mock`.
4. **Sin modificar módulos ajenos.** Juan no cambia normalización, pesos, clases
   ni umbrales para mejorar una demostración.
5. **Orden canónico de etiquetas** (fuente única = `labels.json` de Acosta,
   expuesto por `GET /api/v1/config`):
   `0 Hola · 1 Buenos · 2 Días · 3 Tardes · 4 Noches · 5 Yo · 6 Nombre ·
   7 Años · 8 Gustar · 9 Licor`.

## Interfaces de integración (resumen)

```python
# Extractor (Ortiz) — src/lsc_features
extractor = FeatureExtractor(config, model_paths)
result = extractor.extract_sequence(frames_rgb)   # 6 x RGB uint8, mismo tamaño, orden temporal, sin espejo
# result: x float32 (6,147), quality_pass, reason_codes, valid_shoulders,
#         frames_with_hand, feature_spec_hash, elapsed_ms

# Predictor (Acosta) — src/lsc_inference
probabilities = Predictor.predict(x)              # x float32 (B,6,147) -> (B,10) en [0,1], suma ~1

# Decisión (Acosta, adaptada por Juan)
decision = DecisionAdapter.decide(probabilities)  # -> accepted, label_id, confidence, reason_codes
```

Artefactos esperados de Acosta: `model.keras`, `labels.json`,
`feature_spec.json`, `preprocessing.json`, `thresholds.json`, `model_card.md`,
`manifest.json`. Artefactos esperados de Ortiz: configuración y modelos `.task`
del extractor con procedencia verificable.

## Fechas clave (Bogotá, 2026)

| Hito | Fecha |
| --- | --- |
| Intercambio mínimo de ejemplos entre componentes | 24 oct. |
| Congelación de la configuración de la prueba externa | 18 nov. |
| Versión estable | 27 nov. |
| Exposición (17:30, 10 min + 5 de preguntas) | 2 dic. |
| Artículo | 4 dic. |
