#!/usr/bin/env python3
"""Genera release_app.json y valida el paquete de entrega (SDD §14, §15, §21).

release_app.json registra el commit, el entorno, la versión de la imagen y los
hashes de los artefactos de modelo, extractor y protocolo de captura. El script
valida que el paquete incluya manual, configuración y evidencia.

    python deployment/make_release.py --image lsc-app:local --out release_app.json
    python deployment/make_release.py --artifacts ./artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(artifacts_dir: Path) -> dict:
    names = [
        "model.keras",
        "labels.json",
        "feature_spec.json",
        "preprocessing.json",
        "thresholds.json",
        "manifest.json",
        "pose_landmarker.task",
        "hand_landmarker.task",
    ]
    result = {}
    for name in names:
        result[name] = sha256_file(artifacts_dir / name)
    return result


def read_versions(artifacts_dir: Path) -> dict:
    out = {"model_version": None, "feature_spec_hash": None, "policy_version": None}
    manifest = artifacts_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            out["model_version"] = data.get("model_version")
            out["feature_spec_hash"] = data.get("feature_spec_hash")
        except Exception:
            pass
    thresholds = artifacts_dir / "thresholds.json"
    if thresholds.exists():
        try:
            data = json.loads(thresholds.read_text(encoding="utf-8"))
            out["policy_version"] = data.get("policy_version")
        except Exception:
            pass
    return out


def validate_package() -> list[str]:
    """Comprueba que existan los entregables de documentación y configuración."""
    required = [
        "README.md",
        "AGENTS.md",
        "api/schemas/openapi.json",
        "docs/app/user-guide.md",
        "docs/app/operator-guide.md",
        "docs/app/http-contract.md",
        "docs/app/troubleshooting.md",
        "docs/article/deployment.md",
        "deployment/compose.yaml",
        "deployment/Dockerfile",
        "deployment/requirements-app.txt",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    return missing


def main(argv: list[str] | None = None) -> int:
    from api import __version__ as API_VERSION
    from api.constants import CAPTURE_PROTOCOL_VERSION

    parser = argparse.ArgumentParser(description="Genera release_app.json")
    parser.add_argument("--image", default="lsc-app:local")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="release_app.json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="falla si faltan entregables de documentación",
    )
    args = parser.parse_args(argv)

    artifacts_dir = (ROOT / args.artifacts).resolve()
    missing = validate_package()

    release = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_version": API_VERSION,
        "capture_protocol_version": CAPTURE_PROTOCOL_VERSION,
        "image": args.image,
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "artifacts_dir": str(artifacts_dir),
        "artifact_versions": read_versions(artifacts_dir),
        "artifact_sha256": artifact_hashes(artifacts_dir),
        "package_missing": missing,
    }

    out_path = (ROOT / args.out).resolve()
    out_path.write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"release_app.json escrito en {out_path}")

    if missing:
        print("ADVERTENCIA: faltan entregables:\n  - " + "\n  - ".join(missing), file=sys.stderr)
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
