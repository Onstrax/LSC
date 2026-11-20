# Referencias visuales de las señas

La interfaz muestra, al elegir una seña del vocabulario, el archivo que `manifest.json`
indica para ella. Mientras una entrada tenga `"file": null`, la interfaz dice que la referencia
aún no está disponible; no se inventa ninguna.

## Cómo agregar una referencia
1. Copie el archivo a esta carpeta. Nombre simple, sin espacios ni rutas: `hola.png`, `licor.mp4`.
   Formatos: png, jpg, webp, gif, svg, mp4, webm. Prefiera video corto o GIF: las señas tienen movimiento.
2. En `manifest.json`, complete la entrada de la seña (la clave es el texto exacto de la etiqueta):
   - `file`: nombre del archivo.
   - `alt`: descripción corta para lectores de pantalla (qué hace la mano, no solo «imagen de Hola»).
   - `tip` (opcional): una frase de ayuda.
   - `credit`: autoría y licencia. Obligatorio si la licencia lo exige.
3. Recargue la página.

## Reglas del proyecto (no negociables)
- **No use fotos ni cuadros del corpus LSC70** (`LSC70/`): son personas reales y nunca van a Git.
- **No suba imágenes de personas sin su consentimiento escrito** para este uso.
- Use ilustraciones propias, o material con licencia que permita reutilizarlo, y cite la fuente en `credit`.
- Las referencias deben ser **correctas en LSC**: pídale a una persona usuaria de LSC que las revise.
  No se generan ni se dibujan de memoria.
- Archivos pequeños (idealmente < 500 KB cada uno) para que la app cargue rápido en local.
