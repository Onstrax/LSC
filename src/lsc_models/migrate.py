"""Sincroniza ``artifacts/data/v1/feature_spec.json`` con el contrato corregido.

Contexto: el contrato original tenía mal el SHA-256 de ``hand_landmarker.task`` (una "e" de más) y por
eso un ``feature_spec_hash`` calculado sobre un valor erróneo. Los ``.task`` usados para extraer las
características son los mismos; las características (``features.npz``) NO cambian. Solo cambia el
registro de procedencia, así que se migra el archivo de especificación del bundle sin reprocesar.

Garantías (si alguna falla, NO se escribe nada):
- el bundle viejo debe corresponder EXACTAMENTE al contrato viejo conocido (hash antiguo + valor erróneo);
- salvo ``model_asset_hashes.hands`` y ``feature_spec_hash``, el resto del spec debe ser idéntico al contrato nuevo;
- el ``feature_spec_hash`` del contrato nuevo se recomputa y debe coincidir;
- se guarda copia del archivo anterior (``feature_spec.json.bak_pre_hash_fix``).
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "contracts" / "feature_spec.json"
DEFAULT_BUNDLE = Path("artifacts/data/v1")

OLD_SPEC_HASH = "08779533e99cc466db5845498789e13b11bcab02144d41135915a1e24961c004"
OLD_HANDS = "fbc2a30080c3c557093b5ddfc334698132eb341044cceee322ccf8bcf3607cde1"
NEW_HANDS = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


class MigrationError(RuntimeError):
    pass


def _strip(spec: dict) -> dict:
    s = copy.deepcopy(spec)
    s.get("model_asset_hashes", {}).pop("hands", None)
    s.pop("feature_spec_hash", None)
    return s


def sync_bundle_spec(bundle_dir: Path = DEFAULT_BUNDLE, contract: Path = CONTRACT) -> str:
    """Devuelve 'migrado' o 'ya sincronizado'. Lanza MigrationError si algo no cuadra."""
    from contracts.canonical import feature_spec_hash

    target = Path(bundle_dir) / "feature_spec.json"
    if not target.is_file():
        raise MigrationError(f"no existe {target}: genere el bundle con `prepare-data` primero")
    new = json.loads(Path(contract).read_text(encoding="utf-8"))
    old = json.loads(target.read_text(encoding="utf-8"))

    if feature_spec_hash(new) != new.get("feature_spec_hash"):
        raise MigrationError("el contrato nuevo es inconsistente: su feature_spec_hash no coincide con su contenido")
    if new["model_asset_hashes"]["hands"] != NEW_HANDS:
        raise MigrationError("el contrato no tiene el hash de manos corregido esperado; no se migra")
    if old == new:
        return "ya sincronizado"
    if old.get("feature_spec_hash") != OLD_SPEC_HASH or old["model_asset_hashes"].get("hands") != OLD_HANDS:
        raise MigrationError("el bundle no corresponde al contrato anterior conocido; no se modifica (revise a mano)")
    if _strip(old) != _strip(new):
        raise MigrationError("el bundle difiere del contrato en campos distintos del hash de manos; no se modifica")

    shutil.copy2(target, target.with_name("feature_spec.json.bak_pre_hash_fix"))
    target.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return "migrado"


def main_sync(bundle_dir: str | None) -> int:
    d = Path(bundle_dir) if bundle_dir else DEFAULT_BUNDLE
    try:
        res = sync_bundle_spec(d)
    except MigrationError as exc:
        print(f"migración rechazada: {exc}", file=sys.stderr)
        return 2
    print(f"[sync-bundle-spec] {d}/feature_spec.json: {res}")
    from src.lsc_data.cli import main as data_main

    print("[sync-bundle-spec] validando el bundle con lsc_data…")
    rc = data_main(["validate", "--bundle", str(d)])
    print("[sync-bundle-spec] validación:", "OK" if rc == 0 else f"FALLÓ (código {rc})")
    return rc
