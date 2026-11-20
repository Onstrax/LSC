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
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/train/experiments.yaml")
REQUIRED_SEEDS = (17, 29, 43)
KNOWN_POLICIES = ("train_v1", "train_v2_robust")
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
    # Solo v2 (p. ej. robust_val, test_sealed). Vacío en v1: no entra al hash, así los config_hash de v1 no cambian.
    evaluation: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return f"{self.name}__seed{self.seed}"

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "seed": self.seed,
            "model": self.model,
            "augment": self.augment,
            "model_params": self.model_params,
            "training": self.training,
            "augmentation": self.augmentation,
            "latency": self.latency,
        }
        if self.evaluation:
            d["evaluation"] = self.evaluation
        return d

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
    # v1 exige exactamente las 4 configuraciones del plan; v2 declara las suyas en 'required_configs'.
    required = tuple(cfg.get("required_configs", REQUIRED_CONFIGS))
    if tuple(cfg["configs"].keys()) != required:
        raise ConfigError(f"las configuraciones deben ser {required}")
    policy = cfg["augmentation"].get("policy")
    if policy not in KNOWN_POLICIES:
        raise ConfigError(f"política de aumentación desconocida: {policy!r} (válidas: {KNOWN_POLICIES})")
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
                    evaluation=copy.deepcopy(cfg.get("evaluation", {})),
                )
            )
    return runs


def config_names(cfg: dict) -> list[str]:
    """Nombres de las configuraciones en el orden del YAML (v1: las 4 del plan)."""
    return list(cfg["configs"].keys())
