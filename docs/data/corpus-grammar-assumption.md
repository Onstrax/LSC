# Supuesto de gramática del corpus LSC70W (a verificar)

El adapter `src/lsc_data/adapter_lsc70w.py` reconstruye secuencias a partir de la
estructura de carpetas/nombres del corpus LSC70W usando una **regex nombrada**
declarada en `configs/data/lsc70w.yaml`. Como en este lote NO se dispone del
corpus real (bloque DO02), el patrón es un **SUPUESTO** que debe verificarse
contra el corpus real antes de empaquetar (design §9).

## Supuesto actual

Regex en `configs/data/lsc70w.yaml`:

```
^(?P<signer_id>S\d{3})/(?P<class_label>[a-z]+)(?:/(?P<repetition_id>\d+))?/frame_(?P<frame_index>\d+)\.jpg$
```

Interpretación supuesta:

- `signer_id` — identificador de la persona, formato `S` + 3 dígitos (p. ej. `S001`).
- `class_label` — etiqueta de carpeta en minúsculas sin tildes, mapeada a
  `class_id` 0..9 por `label_map` (fuente del texto: `contracts/labels.json`).
- `repetition_id` — OPCIONAL. El corpus LSC70W **no** declara repeticiones, así
  que por defecto es `0` (`default_repetition_id: 0`).
- `frame_index` — índice entero del cuadro dentro de la secuencia; se ordena
  numéricamente y cada muestra válida tiene exactamente 6 índices únicos (design §9).

## Cómo se verifica contra el corpus real (desbloqueo)

1. Resolver DO02: colocar el archivo del corpus en una ruta local y ejecutar
   `python -m lsc_data acquire --config configs/data/source.yaml --archive <ruta>`.
2. Ejecutar `inventory --raw-dir <dir> --out reports/data/inventory.csv` y revisar
   las rutas reales.
3. Ajustar la regex de `lsc70w.yaml` si la estructura real difiere (extensión,
   separadores, formato de `signer_id`, presencia de repeticiones).
4. El adapter FALLA con un mensaje accionable si la regex no casa una ruta
   candidata; **nunca adivina** (design §9). Un fallo indica que el supuesto debe
   corregirse, no que se deba relajar el control.

## Riesgos del supuesto

- Si el corpus usa otra extensión (p. ej. `.png`) o una jerarquía distinta, la
  regex no casará y habrá que actualizarla.
- Si existieran repeticiones reales, deberán reflejarse como grupo
  `repetition_id` para no agrupar cuadros de ejecuciones distintas.
- El orden canónico de `class_id` NO se infiere de `os.listdir` ni
  alfabéticamente; proviene de `label_map` y de `contracts/labels.json`
  (design §4.1).
