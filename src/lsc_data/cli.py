"""CLI ``python -m lsc_data`` (design §19).

Subcomandos: ``acquire``, ``inventory``, ``manifest``, ``split``, ``features``,
``validate``. Mapeo UNIFORME de códigos de salida:

- ``0`` éxito / conforme,
- ``2`` entradas o configuración inválidas (``ConfigError``),
- ``3`` fallo de integridad (``IntegrityError``).

Construir el parser y pedir ``--help`` NO importa PyYAML, numpy, cv2 ni
mediapipe: los módulos que los necesitan se importan dentro de cada handler
(import perezoso), de modo que ``python -m lsc_data --help`` funciona sin esas
dependencias.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from .errors import ConfigError, IntegrityError

__all__ = ["build_parser", "main"]

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_INTEGRITY = 3

SEED_DEFAULT = 20261022


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser con los 6 subcomandos (sin imports pesados)."""
    parser = argparse.ArgumentParser(
        prog="lsc_data",
        description="Paquete de datos LSC sin red (adquisición, manifiesto, "
        "splits, empaquetado, validación).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_acquire = sub.add_parser(
        "acquire", help="verifica procedencia o extrae un archivo local (sin red)"
    )
    p_acquire.add_argument("--config", required=True)
    p_acquire.add_argument("--archive", default=None)
    p_acquire.add_argument("--dest", default=None)

    p_inventory = sub.add_parser("inventory", help="censa un directorio crudo a CSV")
    p_inventory.add_argument("--raw-dir", required=True)
    p_inventory.add_argument("--out", required=True)

    p_manifest = sub.add_parser("manifest", help="construye el manifiesto JSONL")
    p_manifest.add_argument("--config", required=True)
    p_manifest.add_argument("--raw-dir", required=True)
    p_manifest.add_argument("--out", required=True)
    p_manifest.add_argument("--exclusions", default=None)

    p_split = sub.add_parser("split", help="genera splits deterministas sin fuga")
    p_split.add_argument("--manifest", required=True)
    p_split.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p_split.add_argument("--out", default=None)

    p_features = sub.add_parser("features", help="empaqueta features.npz (bloque DO07)")
    p_features.add_argument("--config", required=True)
    p_features.add_argument("--manifest", required=True)
    p_features.add_argument("--splits", required=True)
    p_features.add_argument("--out", required=True)
    p_features.add_argument("--resume", action="store_true")

    p_validate = sub.add_parser("validate", help="valida un bundle de características")
    p_validate.add_argument("--bundle", required=True)

    return parser


def _handle_acquire(args: argparse.Namespace) -> int:
    from .acquisition import acquire

    moved = acquire(args.config, archive=args.archive, dest_dir=args.dest)
    logging.getLogger("lsc_data").info("acquire: %d archivos movidos", len(moved))
    return EXIT_OK


def _handle_inventory(args: argparse.Namespace) -> int:
    from .inventory import inventory

    rows = inventory(args.raw_dir, args.out)
    logging.getLogger("lsc_data").info("inventory: %d grupos", len(rows))
    return EXIT_OK


def _handle_manifest(args: argparse.Namespace) -> int:
    from .adapter_lsc70w import load_grammar_config
    from .manifest import build_manifest

    grammar = load_grammar_config(args.config)
    exclusions = args.exclusions or str(_sibling(args.out, "exclusions.csv"))
    report = build_manifest(
        args.raw_dir,
        grammar,
        manifest_path=args.out,
        exclusions_path=exclusions,
    )
    logging.getLogger("lsc_data").info(
        "manifest: %d válidas, %d excluidas", report.n_valid, report.n_excluded
    )
    return EXIT_OK


def _handle_split(args: argparse.Namespace) -> int:
    from .split import make_splits

    report = make_splits(args.manifest, seed=args.seed, out_path=args.out)
    logging.getLogger("lsc_data").info(
        "split: %d/%d/%d (train/val/test), incidentes=%d",
        report.n_train,
        report.n_val,
        report.n_test,
        len(report.incidents),
    )
    return EXIT_OK


def _handle_features(args: argparse.Namespace) -> int:
    # DO07: la construcción real sobre corpus requiere el extractor (FEAT-003) y
    # datos reales. Aquí se ejerce la semántica de --resume (§19): si el bundle
    # existente es incompatible con la config/manifiesto actuales, se ABORTA con
    # integridad (exit 3) sin sobrescribir.
    from .features_build import run_features

    return run_features(
        config_path=args.config,
        manifest_path=args.manifest,
        splits_path=args.splits,
        out_dir=args.out,
        resume=args.resume,
    )


def _handle_validate(args: argparse.Namespace) -> int:
    from .validate import validate

    return validate(args.bundle)


def _sibling(path: str, name: str) -> str:
    from pathlib import Path

    return str(Path(path).with_name(name))


_HANDLERS = {
    "acquire": _handle_acquire,
    "inventory": _handle_inventory,
    "manifest": _handle_manifest,
    "split": _handle_split,
    "features": _handle_features,
    "validate": _handle_validate,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada; devuelve el código de salida uniforme 0/2/3."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.command]
    try:
        return handler(args)
    except ConfigError as exc:
        logging.getLogger("lsc_data").error("config inválida: %s", exc)
        return EXIT_CONFIG
    except IntegrityError as exc:
        logging.getLogger("lsc_data").error("fallo de integridad: %s", exc)
        return EXIT_INTEGRITY


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
