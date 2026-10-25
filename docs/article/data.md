# Sección de datos del artículo (estructura, DO10)

Esta sección documenta el corpus y el pipeline de datos del proyecto LSC. En
este lote NO contiene métricas del modelo ni conteos del corpus real: esos
valores se incorporan cuando Acosta entregue sus métricas (bloque DO10) y cuando
el corpus real se procese (bloques DO02/DO07). No se inventan cifras
(design §24/§25).

## 1. Corpus

- **Nombre / procedencia.** LSC70W (Lengua de Señas Colombiana). DOI
  `10.17632/9ssyn8tff5.2`, versión 2 (Mendeley Data). Los subconjuntos LSC70AN y
  LSC70ANH se mantienen separados de LSC70W; sólo se procesa el subset declarado
  en `configs/data/source.yaml`.
- **Clases.** 10 clases en orden canónico fijo (fuente: `contracts/labels.json`):
  `0 Hola, 1 Buenos, 2 Días, 3 Tardes, 4 Noches, 5 Yo, 6 Nombre, 7 Años,
  8 Gustar, 9 Licor`.
- **Conteos (pendiente DO02/DO07).** Número de personas, muestras por clase y
  totales se reportan desde `reports/data/` tras procesar el corpus real.

## 2. Representación de características

- Cada muestra es una secuencia de 6 instantes; por instante se conservan 49
  puntos (7 de pose + 21 mano izquierda + 21 mano derecha), cada uno como triple
  `(x, y, m)`. Forma final `(6, 147)`, `dtype=float32`.
- `m` es una máscara binaria de presencia-del-punto (`0`/`1`); los puntos
  ausentes quedan en `(0, 0, 0)`.
- Normalización por secuencia: centro = mediana de los puntos medios de hombros;
  escala = mediana de las distancias de hombros (comunes a los 6 cuadros).
- El contrato exacto (orden de puntos, versiones, hash) vive en
  `contracts/feature_spec.json`; el `feature_spec_hash` identifica la versión.

## 3. Reconstrucción, splits y calidad

- **Secuencias.** Agrupación por `(source, signer_id, class_id, repetition_id)`,
  orden numérico por `frame_index`, exactamente 6 índices únicos por muestra. La
  gramática del corpus es un supuesto a verificar (ver
  `docs/data/corpus-grammar-assumption.md`).
- **Splits sin fuga.** División por `signer_id` con semilla `20261022`; ninguna
  persona aparece en más de un conjunto. Para 70 personas: 49/7/14
  (train/val/test). Controles de intersección y de copias cruzadas por
  `sequence_sha256`.
- **Calidad.** Una muestra pasa si tiene 6 cuadros decodificables, a lo sumo una
  persona por cuadro, hombros válidos en ≥5 cuadros y ≥1 mano asociada en ≥4
  cuadros; en caso contrario se marca `quality_pass=False` con `reason_codes`.

## 4. Augmentation

- Política `train_v1` sólo en entrenamiento: rotación global por secuencia
  (±5°, prob 0.5) y ruido gaussiano (std 0.005, prob 0.5), aplicados sólo donde
  `m=1`. Validación, prueba y producción NUNCA aplican augmentation.

## 5. Reproducibilidad y ética

- Ejecución determinista: desde cero produce los mismos `sample_id`, etiquetas y
  splits. Empaquetado tipado sin pickle.
- No se versionan copias de consentimiento; `consent_ref` es un id opaco. Las
  rutas del manifiesto no contienen nombres personales (design §7/§25).

## 6. Pendientes (bloques)

- **DO02/DO07.** Conteos del corpus real y paquete de características.
- **DO10.** Métricas del modelo (Acosta) para completar resultados.

Ver `docs/data/blocks.md` para las condiciones de desbloqueo exactas.
