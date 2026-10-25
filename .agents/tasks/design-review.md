# Revisión de diseño — Módulo de datos y características (Ortiz, 30 %)

Documento revisado: `.agents/tasks/design.md` (v3, incorpora dos rondas previas
de revisión). Revisión hecha en frío contra el código real del repositorio.

Alcance de la revisión: ambigüedad, suposiciones no verificadas, detalles
faltantes, viabilidad, scope creep, conflictos entre secciones, manejo de
errores no especificado y reglas de validación de entrada. El veredicto es
mecánico: cualquier hallazgo HIGH o MEDIUM ⇒ `CHANGES_REQUESTED`.

---

## Resumen de verificación

He leído y comprobado contra las fuentes reales las afirmaciones del diseño que
dependen de código ajeno (`api/`, `pytest.ini`, `.gitignore`, ausencia de
`pyproject.toml` raíz), la matemática del muestreo temporal (§15) y la partición
de splits para N=70 (§10). Casi todas resultaron correctas; el detalle está en
las secciones "Suposiciones verificadas" y "Suposiciones no verificadas / erróneas".

El diseño es inusualmente completo: cubre el layout `(6,147)`, el orden de 49
puntos, el triple `(x,y,m)`, el hashing canónico byte a byte, los splits
deterministas, el `FakeDetector` inyectable, el mapeo D01–D12 / F01–F12, las
pruebas de fallo deliberado, la prueba de config incompatible y los bloques
fuera de alcance DO02/DO05/DO07/DO08/DO09/DO10 con instrucciones de desbloqueo.
Respeta la propiedad de rutas de `AGENTS.md`.

Quedan, sin embargo, dos vacíos que bloquean implementación sin tomar una
decisión nueva (uno MEDIUM, uno NIT que no bloquea) más algunas aclaraciones.

---

## Hallazgos

### 1. [MEDIUM] Normalización del costo de asociación indefinida cuando el cuadro no tiene hombros válidos

Ubicación: §11 ("Asociación mano-pose"), en relación con §12.2 y §13.

§11 define el costo de enlace muñeca-de-mano ↔ muñeca-de-pose como "la distancia
en píxeles entre ambas, **normalizada por la distancia de hombros del cuadro**",
y aplica el umbral `<=0.6` sobre ese costo normalizado. Pero §12.2 sólo calcula
distancia de hombros "por cuadro **con hombros válidos**" y §13 admite muestras
con hombros válidos en sólo ≥5 de 6 cuadros. En el (o los) cuadro(s) sin hombros
válidos, el denominador de normalización es indefinido (o cero), de modo que el
costo de asociación no está definido. El diseño no dice qué ocurre con la
asociación de manos en un cuadro sin hombros válidos: ¿las manos quedan
automáticamente ausentes?, ¿se usa la escala mediana de secuencia (§12.3) como
denominador de respaldo?, ¿se usa un denominador por cuadro previo?

Esto es ambiguo con interpretaciones válidas distintas que cambian
`frames_with_hand`, `HAND_ASSOCIATION_AMBIGUOUS` y por tanto `quality_pass`.

Fix concreto (elegir uno y fijarlo en §11): "Si un cuadro no tiene hombros
válidos (distancia de hombros < `0.08*W` o pose-11/pose-12 no finitos), NO se
realiza asociación de manos en ese cuadro; ese cuadro contribuye 0 a
`frames_with_hand` y no emite `HAND_ASSOCIATION_AMBIGUOUS`. La normalización del
costo usa SIEMPRE la distancia de hombros del propio cuadro; nunca la escala de
secuencia de §12.3." Añadir un caso de prueba (p. ej. `F08d`/`F09d`) con
FakeDetector que devuelva muñecas de mano en un cuadro sin hombros válidos y
verifique que ese cuadro no suma manos.

### 2. [NIT] Desempate de `round()` nativo en `n_val` de los splits no especificado

Ubicación: §10, paso 3: `n_val = max(1, round(0.10*N))`.

§15 advierte explícitamente que NO debe usarse el `round()` nativo de Python
(banker's/half-to-even) para el muestreo temporal, y lo reemplaza por un half-up
determinista. Sin embargo §10 usa el `round()` nativo para `n_val` sin fijar la
regla de desempate. Para el único caso que exige el prompt (N=70 ⇒ `round(7.0)=7`)
no hay empate, así que no afecta la aceptación; pero para cualquier N donde
`0.10*N` termine en `.5` (p. ej. N=25 ⇒ 2.5) el resultado dependería del modo de
redondeo del intérprete, y un lector podría asumir half-up por analogía con §15.

Fix concreto: fijar la semántica en §10, p. ej. "`n_val = max(1, int(floor(0.10*N + 0.5)))`
(half-up, consistente con §15)" o declarar explícitamente que se usa
redondeo bancario de Python y que el efecto sólo aplica a tamaños `.5` exactos.
No bloquea porque N=70 es inequívoco.

---

## Suposiciones verificadas (contra el código real)

1. `api/constants.py`: `SEQUENCE_SHAPE == (6, 147)`, `FEATURE_DIM == 147` con
   docstring "49 puntos x (x, y, presencia)", `NUM_CLASSES == 10`. ✔ Coincide con
   §1, §8.1, §12 (el canal `m` descrito en §12 como máscara de presencia-del-punto
   corresponde al canal genérico "presencia" de `FEATURE_DIM`).
2. `api/constants.QUALITY_REASON_CODES` es un `frozenset` con EXACTAMENTE los 9
   códigos listados en §13/§18. ✔ La afirmación de §18 (comparación de conjuntos,
   orden irrelevante; único orden de emisión = §13) es correcta.
   `NON_BLOCKING_WARNINGS == {"HAND_ASSOCIATION_AMBIGUOUS"}` ✔ consistente con §13.
3. `api/adapters/types.py`: `FeatureResult` es `@dataclass(frozen=True)` con
   campos `x, quality_pass, reason_codes, valid_shoulders, frames_with_hand,
   feature_spec_hash, elapsed_ms`, y `valid_shoulders: bool = True`. ✔ Confirma la
   nota de §1/§8.1/DO08: el diseño pide `valid_shoulders: int` y la API lo coacciona
   con `bool(...)`; la fricción es real y está correctamente marcada como bloque
   DO08 sin tocar `api/`.
4. `api/adapters/real.py`: construye `FeatureExtractor(feature_config or
   self._feature_spec, model_paths)` con `model_paths = {"pose": ..., "hands": ...}`
   y pasa `feature_spec.json` cargado como `dict`. ✔ Confirma la firma exacta y las
   reglas de claves `{"pose","hands"}` de §8.3. También hace
   `np.asarray(result.x, dtype=np.float32)` y `bool(getattr(result,
   "valid_shoulders", True))`. ✔ Coincide con §8.1 y la nota DO08.
5. `pytest.ini` raíz existe y declara `testpaths = tests/app, tests/e2e`,
   `markers = realmodel/e2e/slow`, `addopts = -ra`, con encabezado "módulo de
   Juan". ✔ Confirma §20/§22/§23: un `pytest` simple no recoge las pruebas de
   Ortiz; el comando canónico con paths explícitos las recoge y conserva
   markers/addopts. La decisión de no editar `pytest.ini` respeta `AGENTS.md`.
6. `.gitignore` real ignora `data/`, `artifacts/*` (con `!artifacts/README.md`),
   `*.task`, `*.keras`, y el único patrón de `reports/` es `reports/app/*_private*`.
   ✔ Confirma §21: las 4 adiciones del borrador eran redundantes; `contracts/`,
   fixtures sintéticos, código, `docs/data/`, `reports/data/README.md` y `.gitkeep`
   NO caen bajo exclusión.
7. No existe `pyproject.toml` raíz; sólo `pyproject.app.toml`. ✔ Confirma §2/§22:
   `requirements-data.in` se entrega como solicitud a Acosta y
   `test_requirements_version.py` no debe acoplarse a un archivo inexistente.
8. Muestreo temporal §15: verificado numéricamente
   `6→[0,1,2,3,4,5]`, `7→[0,1,2,4,5,6]`, `10→[0,2,4,5,7,9]`, `11→[0,2,4,6,8,10]`,
   con `idx_0=0`, `idx_5=N-1`. ✔ La tabla de §15 y el anclaje de `F11` son correctos.
9. Partición de splits §10 para N=70: verificado `n_test=ceil(0.20*70)=14`,
   `n_val=max(1,round(7.0))=7`, `n_train=49`. ✔ Coincide con la expectativa 49/7/14.
10. No existe código previo en `src/lsc_data/` ni `src/lsc_features/` (implementación
    desde cero, sin colisión con módulos ajenos). ✔
11. Propiedad de rutas: el árbol de §3 toca sólo `contracts/`, `src/lsc_data/`,
    `src/lsc_features/`, `configs/data/`, `tests/{data,features,contracts}/`,
    `reports/data/`, `docs/data/`, `docs/article/data.md`,
    `deployment/requirements-data.in` y a lo sumo un comentario en `.gitignore`.
    ✔ Respeta `AGENTS.md`; `api/schemas` y `api/adapters/types.py` se usan sólo como
    referencia de lectura.

## Suposiciones no verificadas / erróneas

- Ninguna suposición del diseño resultó FALSA contra el código real. Todas las
  afirmaciones sobre `api/`, `pytest.ini`, `.gitignore` y la ausencia del
  `pyproject.toml` raíz se confirmaron.
- No verificable por diseño (correctamente marcado como bloque, no es hallazgo):
  la gramática regex del corpus LSC70W (§9) y la procedencia/hashes reales del
  corpus (DO02) y de los `.task` (DO05) no pueden comprobarse sin los artefactos
  reales; el diseño los documenta como supuestos a validar con mensaje accionable
  y sin adivinar, lo cual es la conducta correcta.

---

## Cobertura del checklist del prompt

- Propiedad de rutas: ✔ (verificado, hallazgo 11 arriba).
- JSON canónico byte a byte por cada SHA-256 (sample_id, sequence_sha256,
  feature_spec_hash reproducibles) y `feature_spec_hash` excluye timestamps: ✔
  §6, §6.1 (excluye `build_date`), §6.2, §6.3.
- Layout `(6,147)`, orden de 49 puntos (7 pose 0,11,12,13,14,15,16; 21 izq; 21
  der) y orden del triple `(x,y,m)`: ✔ §4.2, §11, §12.5, F02.
- Regla de asociación de mínimo costo con enlace `<=0.6` y gap `<0.05`, y
  `GEOMETRY_OUTLIER |valor|>10`: ✔ §11, §12.6 — con la salvedad del hallazgo 1
  (cuadro sin hombros válidos).
- Splits sin fuga con `default_rng(20261022)`, partición
  `ceil(0.20N)/max(1,round(0.10N))/resto` y 49/7/14 para N=70: ✔ §10 (ver NIT
  hallazgo 2 sobre el desempate de `round`).
- `FakeDetector` inyectable y carga real `.task` como bloque no ejercitado: ✔
  §8.2, §8.3, DO05.
- Mapeo de D01–D12, F01–F12, pruebas de fallo deliberado y config incompatible: ✔
  §20 (D06/D09/F10/F11/F12 marcadas como fallo deliberado;
  `test_incompatible_config.py`).
- DO02/DO05/DO07/DO08/DO09/DO10 como bloques fuera de alcance con instrucciones de
  desbloqueo exactas: ✔ §24 (y DO11 añadido para `testpaths`).

---

## Veredicto

Hallazgos HIGH: 0. Hallazgos MEDIUM: 1 (hallazgo 1). Hallazgos NIT: 1.

Como hay 1 hallazgo MEDIUM, el veredicto es **CHANGES_REQUESTED**. El único
bloqueante es la normalización del costo de asociación en cuadros sin hombros
válidos (§11); es una decisión acotada y de resolución rápida.
