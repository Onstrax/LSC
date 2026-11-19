"""Carga y validación de ``configs/train/experiments.yaml``.

La matriz de experimentos es 2 arquitecturas x {sin, con} aumento x 3 semillas
(17, 29, 43) = 12 ejecuciones. Ambas arquitecturas comparten particiones,
preprocesamiento y presupuesto de entrenamiento (sección ``training``); el
único grado de libertad entre configuraciones es ``model`` y ``augment``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/train/experiments.yaml")
REQUIRED_SEEDS = (17, 29, 43)
# Presupuesto reducido SOLO para verificar la tubería con datos sintéticos.
# run-all/sensitivity rechazan --fast si el bundle no es sintético.
FAST_OVERRIDES = {
    "training": {"max_epochs": 3, "patience": 2, "min_epochs": 1},
    "latency": {"warmup": 1, "repeats": 5},
    "sensitivity": {"hand_dropout_fractions": [0.0, 0.33, 0.67], "repetitions": 2},
}
REQUIRED_CONFIGS = ("bilstm_noaug", "bilstm_aug", "transformer_noaug", "transformer_aug")


class ConfigError(ValueError):
    """Configuración de entrenamiento inválida."""


@dataclass(frozen=True)
class RunSpec:
    """Una ejecución = (configuración, semilla) con todo resuelto."""

    name: str  # p. ej. "bilstm_aug"
    seed: int
    model: str  # "bilstm" | "transformer"
    augment: bool
    model_params: dict
    training: dict
    augmentation: dict
    latency: dict

    @property
    def run_id(self) -> str:
        return f"{self.name}__seed{self.seed}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "model": self.model,
            "augment": self.augment,
            "model_params": self.model_params,
            "training": self.training,
            "augmentation": self.augmentation,
            "latency": self.latency,
        }

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: dict | None = None) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    for key in ("data", "seeds", "training", "augmentation", "models", "configs", "selection", "goals"):
        if key not in cfg:
            raise ConfigError(f"falta la sección '{key}'")
    if tuple(cfg["seeds"]) != REQUIRED_SEEDS:
        raise ConfigError(f"las semillas deben ser {REQUIRED_SEEDS}; hay {tuple(cfg['seeds'])}")
    if tuple(cfg["configs"].keys()) != REQUIRED_CONFIGS:
        raise ConfigError(f"las configuraciones deben ser {REQUIRED_CONFIGS}")
    if cfg.get("deployed_seed") != 17:
        raise ConfigError("deployed_seed debe ser 17 (regla fijada antes del test)")
    for name, entry in cfg["configs"].items():
        if entry.get("model") not in cfg["models"]:
            raise ConfigError(f"{name}: modelo '{entry.get('model')}' no definido en 'models'")
        if not isinstance(entry.get("augment"), bool):
            raise ConfigError(f"{name}: 'augment' debe ser booleano")
    if "test" in json.dumps(cfg["training"]).lower():
        raise ConfigError("'training' no puede referirse al split de test")


def expand_runs(cfg: dict) -> list[RunSpec]:
    """Devuelve las 12 ejecuciones en orden estable (configuración, semilla)."""
    runs: list[RunSpec] = []
    for name, entry in cfg["configs"].items():
        for seed in cfg["seeds"]:
            runs.append(
                RunSpec(
                    name=name,
                    seed=int(seed),
                    model=entry["model"],
                    augment=bool(entry["augment"]),
                    model_params=copy.deepcopy(cfg["models"][entry["model"]]),
                    training=copy.deepcopy(cfg["training"]),
                    augmentation=copy.deepcopy(cfg["augmentation"]),
                    latency=copy.deepcopy(cfg.get("latency", {})),
                )
            )
    return runs
