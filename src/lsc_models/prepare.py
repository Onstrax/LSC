"""Generación de ``features.npz`` (A0: prerrequisito de A1–A3).

Dos caminos, NUNCA mezclados:

REAL (por defecto). Orquesta el pipeline de Ortiz (``src.lsc_data``):
    inventory -> manifest -> split -> features -> validate
  Requiere (1) el corpus LSC70W crudo y (2) los modelos MediaPipe ``.task``.
  Si falta cualquiera, se detiene con un mensaje exacto de qué falta y NO fabrica
  datos. Salida: ``artifacts/data/v1/features.npz``.

SINTÉTICO (``--synthetic``, solo si el usuario lo pide). Escribe un bundle
  ``synthetic_contract_fixture`` en ``artifacts/data/synthetic/`` (carpeta distinta
  de ``v1``). Sirve para verificar que la tubería corre; sus métricas no son evidencia.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

DEFAULT_RAW = Path("Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W")
DEFAULT_MODELS = Path("artifacts/models")
DEFAULT_OUT = Path("artifacts/data/v1")
SYNTHETIC_OUT = Path("artifacts/data/synthetic")
INTERIM = Path("data/interim")
SPEC = Path(__file__).resolve().parents[2] / "contracts" / "feature_spec.json"  # independiente del cwd
TASK_FILES = {"pose": "pose_landmarker.task", "hands": "hand_landmarker.task"}


_BASE = "https://storage.googleapis.com/mediapipe-models"
# URLs oficiales de MediaPipe Tasks (se desconoce con cuál variante generó Ortiz el contrato:
# por eso se prueban todas y SOLO se conserva la que coincide con el SHA-256 del contrato).
POSE_CANDIDATES = [f"{_BASE}/pose_landmarker/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task" for v in ("heavy", "full", "lite")]
HANDS_CANDIDATES = [f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"]
PREFIX_LEN = 12  # 48 bits: una coincidencia accidental es despreciable
CANDIDATES = {"pose": POSE_CANDIDATES, "hands": HANDS_CANDIDATES}


class PrepareError(RuntimeError):
    """Falta un prerrequisito real. El mensaje dice exactamente cuál."""


class AmbiguousRawDir(PrepareError):
    """Hay más de una carpeta LSC70W candidata."""


_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "artifacts", "reports", "__pycache__", "site-packages"}


def _has_signers(d: Path) -> bool:
    return d.is_dir() and any(c.is_dir() and c.name.startswith("Per") for c in d.iterdir())


def find_lsc70w(base: Path, max_depth: int = 5) -> list[Path]:
    """Carpetas llamadas LSC70W (con Per##) bajo ``base``. LSC70AN/LSC70ANH NO se usan (subset LSC70W)."""
    found: list[Path] = []
    base = base.resolve()
    for root, dirs, _ in os.walk(base):
        depth = len(Path(root).relative_to(base).parts)
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")] if depth < max_depth else []
        if Path(root).name == "LSC70W" and _has_signers(Path(root)):
            found.append(Path(root))
    return sorted(found)


def resolve_raw_dir(raw: Path | None, search_from: Path | None = None) -> Path:
    """Localiza LSC70W.

    - Si ``raw`` es la carpeta con Per##, se usa. Si es una carpeta superior (p. ej. LSC70/LSC70 o la
      carpeta "Speak in…"), se baja hasta LSC70W.
    - Sin ``raw``: se prueba la ruta por defecto y, si no existe, se busca LSC70W desde el directorio actual.
    Devuelve ``raw`` sin cambios si no encuentra nada (el chequeo de prerrequisitos explica qué falta).
    """
    if raw is not None:
        if _has_signers(raw):
            return raw
        if raw.is_dir():
            found = find_lsc70w(raw)
            if len(found) == 1:
                return found[0]
            if len(found) > 1:
                raise AmbiguousRawDir("varias carpetas LSC70W bajo %s: %s. Indique una con --raw-dir." % (raw, ", ".join(map(str, found))))
        return raw
    if _has_signers(DEFAULT_RAW):
        return DEFAULT_RAW
    found = find_lsc70w(search_from or Path.cwd())
    if len(found) == 1:
        print(f"[prepare] LSC70W encontrada automáticamente en: {found[0]}")
        return found[0]
    if len(found) > 1:
        raise AmbiguousRawDir("varias carpetas LSC70W encontradas: " + ", ".join(map(str, found)) + ". Indique una con --raw-dir.")
    return DEFAULT_RAW


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_sha256(h: str) -> bool:
    return len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def contract_warnings(spec: dict | None = None) -> list[str]:
    """Defectos del contrato de Ortiz que NO se corrigen aquí (cambiarían el feature_spec_hash)."""
    if spec is None:
        spec = json.loads(SPEC.read_text(encoding="utf-8")) if SPEC.is_file() else {}
    out = []
    for key, h in spec.get("model_asset_hashes", {}).items():
        if not _is_sha256(str(h)):
            out.append(
                f"contracts/feature_spec.json: model_asset_hashes.{key} tiene {len(str(h))} caracteres "
                "(un SHA-256 válido tiene 64; valor alterado). Ortiz debe corregirlo y recomputar "
                "feature_spec_hash; este módulo NO lo modifica."
            )
    return out


def check_prerequisites(raw_dir: Path, models_dir: Path, warnings: list[str] | None = None) -> list[str]:
    """Devuelve la lista de problemas BLOQUEANTES (vacía = todo listo).

    Los defectos del contrato (hash mal formado) van a ``warnings`` y no bloquean:
    el extractor de Ortiz no verifica el hash en ejecución.
    """
    problems: list[str] = []
    warnings = warnings if warnings is not None else []
    if not raw_dir.is_dir():
        problems.append(
            f"No existe el corpus crudo en: {raw_dir}\n"
            "    -> Descargue 'Speak in Colombian Sign Language: A Dynamic LSC70 Database' "
            "(Mendeley Data, DOI 10.17632/9ssyn8tff5.2, CC BY 4.0),\n"
            "       descomprímalo dentro del repo (se busca sola una carpeta llamada LSC70W con Per##/), "
            "o indique esa carpeta con --raw-dir. Solo se usa LSC70W; LSC70AN y LSC70ANH se ignoran."
        )
    elif not any(raw_dir.glob("Per*")):
        problems.append(f"{raw_dir} existe pero no contiene carpetas 'Per##'. ¿Es la carpeta LSC70W correcta?")
    spec = json.loads(SPEC.read_text(encoding="utf-8")) if SPEC.is_file() else {}
    warnings.extend(contract_warnings(spec))
    hashes = spec.get("model_asset_hashes", {})
    for key, fname in TASK_FILES.items():
        p = models_dir / fname
        expected = str(hashes.get(key, ""))
        if not p.is_file():
            problems.append(
                f"Falta {p}\n    -> Pida a Ortiz el archivo {fname} (MediaPipe Tasks, {key}); "
                f"su SHA-256 debe empezar por {expected[:12]}…"
            )
            continue
        actual = _sha256(p)
        if _is_sha256(expected):
            if actual != expected:
                problems.append(
                    f"{p} no coincide con el SHA-256 esperado de contracts/feature_spec.json "
                    f"({expected[:12]}…). Usar otro .task cambiaría el feature_spec_hash."
                )
        elif actual not in (expected[:64], expected[-64:]) and not actual.startswith(expected[:PREFIX_LEN]):
            warnings.append(f"{p}: SHA-256 real {actual[:12]}… NO coincide ni por prefijo con el hash (mal formado) del contrato.")
    return problems


def _fetch(url: str, timeout: int = 180) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (URLs fijas y oficiales)
        return r.read()


def _accepted_hashes(expected: str) -> set[str]:
    """Hash(es) aceptables. Si el del contrato está mal formado (65 car.), se aceptan sus
    dos recortes de 64 (el defecto documentado en docs/models/compat.md)."""
    return {expected} if _is_sha256(expected) else {expected[:64], expected[-64:]}


def download_models(models_dir: Path = DEFAULT_MODELS, *, fetch=_fetch, spec: dict | None = None,
                    accept_mismatch: bool = False) -> dict[str, str]:
    """Descarga los .task oficiales y conserva SOLO los que coinciden con el contrato.

    Devuelve {clave: estado}. Lanza PrepareError si alguno no se pudo obtener/verificar,
    explicando los hashes que SÍ se obtuvieron. ``accept_mismatch`` guarda la primera
    descarga aunque no coincida (rompe la procedencia del contrato; explícito y ruidoso).
    """
    spec = spec if spec is not None else json.loads(SPEC.read_text(encoding="utf-8"))
    hashes = spec["model_asset_hashes"]
    models_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    failures: list[str] = []
    observed: dict[str, dict] = {}
    for key, fname in TASK_FILES.items():
        dest = models_dir / fname
        expected = str(hashes[key])
        ok_hashes = _accepted_hashes(expected)
        prefix = None if _is_sha256(expected) else expected[:PREFIX_LEN]

        def matches(d: str) -> str | None:
            if d in ok_hashes:
                return "exacto"
            if prefix and d.startswith(prefix):
                return "prefijo"
            return None

        if dest.is_file() and matches(_sha256(dest)):
            status[key] = "ya existe y coincide"
            continue
        seen: list[tuple[str, str]] = []
        first: tuple[str, bytes] | None = None
        for url in CANDIDATES[key]:
            try:
                blob = fetch(url)
            except Exception as exc:  # red caída, 404, DNS...
                seen.append((url, f"no descargable ({type(exc).__name__}: {exc})"))
                continue
            digest = hashlib.sha256(blob).hexdigest()
            how = matches(digest)
            if how:
                dest.write_bytes(blob)
                observed[key] = {"file": fname, "url": url, "sha256": digest, "match": how, "contract_hash": expected}
                status[key] = f"descargado y verificado ({url.rsplit('/', 1)[-1]})" + (
                    f" POR PREFIJO de {PREFIX_LEN} caracteres (el hash del contrato está mal formado); SHA-256 real: {digest}"
                    if how == "prefijo" else "")
                break
            seen.append((url, f"sha256 {digest[:12]}… (no coincide)"))
            first = first or (url, blob)
        else:
            if accept_mismatch and first is not None:
                dest.write_bytes(first[1])
                status[key] = f"GUARDADO SIN VERIFICAR ({first[0].rsplit('/', 1)[-1]}): la procedencia del contrato NO se cumple"
                continue
            detail = "; ".join(f"{u.rsplit('/', 1)[-1]} -> {m}" for u, m in seen)
            failures.append(f"{key}: ninguna variante coincide con el contrato ({sorted(ok_hashes)[0][:12]}…). Intentos: {detail}")
    for k, v in status.items():
        print(f"[download-models] {k}: {v}")
    if observed:  # evidencia para que Ortiz corrija el contrato
        obs_path = models_dir / "hashes_observed.json"
        prev = json.loads(obs_path.read_text(encoding="utf-8")) if obs_path.is_file() else {}
        obs_path.write_text(json.dumps({**prev, **observed}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if failures:
        raise PrepareError(
            "No se pudieron obtener/verificar los .task:\n  - " + "\n  - ".join(failures)
            + "\n  Sin internet: descargue manualmente desde https://ai.google.dev/edge/mediapipe/solutions/vision/"
              "pose_landmarker y .../hand_landmarker y copie los .task a " + str(models_dir)
            + ".\n  Si descargó variantes distintas al contrato, repita con --accept-mismatch (los resultados NO tendrán la "
              "procedencia de feature_spec.json y debe declararse)."
        )
    return status


def _run(step: str, argv: list[str]) -> None:
    from src.lsc_data.cli import main as data_main

    print(f"\n[prepare] {step}: lsc_data {' '.join(argv)}", flush=True)
    rc = data_main(argv)
    if rc != 0:
        raise PrepareError(f"el paso '{step}' terminó con código {rc} (2=config inválida, 3=fallo de integridad)")


def prepare_real(raw_dir: Path | None = None, models_dir: Path = DEFAULT_MODELS, out_dir: Path = DEFAULT_OUT, seed: int = 20261022,
                 *, download: bool = True, accept_mismatch: bool = False) -> Path:
    if download and any(not (models_dir / f).is_file() for f in TASK_FILES.values()):
        print("[prepare] faltan .task: intentando descargarlos de las URLs oficiales de MediaPipe…")
        try:
            download_models(models_dir, accept_mismatch=accept_mismatch)
        except PrepareError as exc:
            print(f"[prepare] AVISO: {exc}", file=sys.stderr)  # se reporta de nuevo abajo como prerrequisito faltante
    raw_dir = resolve_raw_dir(raw_dir)
    warns: list[str] = []
    problems = check_prerequisites(raw_dir, models_dir, warns)
    for w in warns:
        print(f"[prepare] AVISO: {w}", file=sys.stderr)
    if problems:
        raise PrepareError("No se puede generar el features.npz REAL todavía:\n  - " + "\n  - ".join(problems))
    INTERIM.mkdir(parents=True, exist_ok=True)
    Path("reports/data").mkdir(parents=True, exist_ok=True)
    raw = str(raw_dir)
    _run("inventario", ["inventory", "--raw-dir", raw, "--out", "reports/data/inventory.csv"])
    _run("manifiesto", ["manifest", "--config", "configs/data/lsc70w.yaml", "--raw-dir", raw,
                        "--out", str(INTERIM / "manifest.jsonl"), "--exclusions", str(INTERIM / "exclusions.csv")])
    _run("particiones por persona", ["split", "--manifest", str(INTERIM / "manifest.jsonl"), "--seed", str(seed),
                                      "--out", str(INTERIM / "splits.json")])
    _run("características (MediaPipe; tarda varios minutos)", ["features", "--config", "configs/data/features_v1.yaml",
                        "--manifest", str(INTERIM / "manifest.jsonl"), "--splits", str(INTERIM / "splits.json"),
                        "--raw-dir", raw, "--out", str(out_dir)])
    _run("validación del bundle", ["validate", "--bundle", str(out_dir)])
    npz = out_dir / "features.npz"
    if not npz.is_file():
        raise PrepareError(f"el pipeline terminó pero no existe {npz}")
    return npz


def prepare_synthetic(out_dir: Path = SYNTHETIC_OUT) -> Path:
    from .data import make_synthetic_contract_fixture

    if out_dir.resolve() == DEFAULT_OUT.resolve():
        raise PrepareError("el bundle sintético no puede escribirse en artifacts/data/v1 (reservado a datos reales)")
    path = make_synthetic_contract_fixture(out_dir / "features.npz", n_signers=20)
    (out_dir / "README_SYNTHETIC.txt").write_text(
        "DATOS SINTÉTICOS (synthetic_contract_fixture). Solo para verificar la tubería. "
        "No son evidencia de exactitud y no deben usarse en el artículo.\n",
        encoding="utf-8",
    )
    return path


def main_prepare(raw_dir: str | None, models_dir: str | None, out: str | None, synthetic: bool,
                 *, download: bool = True, accept_mismatch: bool = False) -> int:
    try:
        if synthetic:
            path = prepare_synthetic(Path(out) if out else SYNTHETIC_OUT)
            print(f"\n[prepare] SINTÉTICO generado: {path}\n          (NO es evidencia; use --bundle {path} con run-all --fast)")
        else:
            path = prepare_real(Path(raw_dir) if raw_dir else None, Path(models_dir) if models_dir else DEFAULT_MODELS,
                                Path(out) if out else DEFAULT_OUT, download=download, accept_mismatch=accept_mismatch)
            print(f"\n[prepare] REAL generado: {path}")
        from .data import Bundle

        print("[prepare] resumen (solo conteos):", json.dumps(Bundle(path).summary(), indent=2))
        return 0
    except PrepareError as exc:
        print(f"\n{exc}\n\nSi solo quiere comprobar que el proyecto corre SIN datos reales:\n"
              "  python -m src.lsc_models verify", file=sys.stderr)
        return 3


def main_download(models_dir: str | None, accept_mismatch: bool) -> int:
    try:
        download_models(Path(models_dir) if models_dir else DEFAULT_MODELS, accept_mismatch=accept_mismatch)
        return 0
    except PrepareError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 3
