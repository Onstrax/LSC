# Reportes y salidas de datos (Ortiz)

Esta carpeta recibe las salidas que generan los subcomandos de la CLI
`python -m lsc_data` (design §3, §19). El `.gitkeep` mantiene la carpeta en el
repositorio aunque sus salidas no se versionen.

## Qué se genera aquí

- `inventory.csv` — inventario del corpus crudo
  (`inventory --raw-dir <dir> --out reports/data/inventory.csv`).
- Reportes de manifiesto/splits/features cuando se construyan sobre el corpus
  real (bloques DO02/DO07).

## Qué se versiona y qué no

- Se versiona: este `README.md` y el `.gitkeep`. El único patrón de `.gitignore`
  bajo `reports/` es `reports/app/*_private*`, que NO toca `reports/data/`
  (design §21).
- Las salidas de datos (`reports/data/*.csv` y demás reportes generados) son
  derivadas del corpus y no forman parte de las rutas versionadas mínimas; si se
  decide no versionarlas se añadiría una regla específica `reports/data/*.csv`,
  pero eso es opcional (design §21).

No se guardan imágenes ni evidencias con identidades personales en esta carpeta
(las evidencias con posibles identidades viven fuera del repo, design §7/§25).
