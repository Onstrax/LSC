"""Adapter de gramática del corpus LSC70W (design §9).

La gramática carpeta/nombre se parametriza con una regex NOMBRADA tomada de
``configs/data/lsc70w.yaml``. La regex debe exponer los grupos nombrados
``signer_id``, ``class_label``, ``frame_index`` y, opcionalmente,
``repetition_id`` (si el corpus no tiene repeticiones se usa
``default_repetition_id``, por defecto ``0``). Si la regex no casa una ruta
candidata, se FALLA con un mensaje accionable; el adapter NUNCA adivina.

Este módulo NO lee imágenes ni toca el corpus real: sólo interpreta rutas
relativas (``str``) según la gramática declarada. ``PyYAML`` se importa de forma
perezosa para que ``import src.lsc_data`` funcione sin la dependencia instalada.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError

__all__ = [
    "GrammarConfig",
    "ParsedFrame",
    "load_grammar_config",
    "Lsc70wAdapter",
]

_REQUIRED_GROUPS = ("signer_id", "class_label", "frame_index")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Carga un YAML como ``dict`` (import perezoso de PyYAML).

    Un YAML inexistente, ilegible o que no produce un mapping es
    ``ConfigError`` (exit 2). PyYAML se importa aquí, no en el import del
    paquete (design §16): así el paquete carga sin la dependencia.
    """
    try:
        import yaml  # import perezoso; falta -> ConfigError accionable
    except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
        raise ConfigError(
            "PyYAML no está instalado; instala deployment/requirements-data.in "
            "para leer configs YAML (configs/data/*.yaml)."
        ) from exc

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config no encontrado: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML mal formado en {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"config {config_path} no es un mapping YAML")
    return dict(raw)


@dataclass(frozen=True)
class GrammarConfig:
    """Gramática resuelta del corpus (design §9)."""

    source: str
    source_version: str
    subset: str
    pattern: re.Pattern[str]
    default_repetition_id: int
    label_map: Mapping[str, int]


@dataclass(frozen=True)
class ParsedFrame:
    """Una ruta de frame interpretada por la gramática."""

    signer_id: str
    class_label: str
    class_id: int
    repetition_id: int
    frame_index: int
    frame_path: str


def load_grammar_config(path: str | Path) -> GrammarConfig:
    """Resuelve ``configs/data/lsc70w.yaml`` a un ``GrammarConfig`` validado.

    Falla con ``ConfigError`` (exit 2) si falta un campo, la regex no compila o
    no declara los grupos nombrados obligatorios, o el ``label_map`` no mapea a
    ``class_id`` enteros 0..9.
    """
    data = _load_yaml(path)

    pattern_str = data.get("path_regex")
    if not isinstance(pattern_str, str) or not pattern_str:
        raise ConfigError("lsc70w.yaml: 'path_regex' ausente o no es string")
    try:
        pattern = re.compile(pattern_str)
    except re.error as exc:
        raise ConfigError(f"lsc70w.yaml: 'path_regex' no compila: {exc}") from exc

    missing = [g for g in _REQUIRED_GROUPS if g not in pattern.groupindex]
    if missing:
        raise ConfigError(
            "lsc70w.yaml: 'path_regex' debe declarar los grupos nombrados "
            f"{_REQUIRED_GROUPS}; faltan {missing}"
        )

    label_map_raw = data.get("label_map")
    if not isinstance(label_map_raw, Mapping) or not label_map_raw:
        raise ConfigError("lsc70w.yaml: 'label_map' ausente o vacío")
    label_map: dict[str, int] = {}
    for label, class_id in label_map_raw.items():
        if not isinstance(class_id, int) or isinstance(class_id, bool):
            raise ConfigError(
                f"lsc70w.yaml: label_map[{label!r}] debe ser int, no {class_id!r}"
            )
        if not (0 <= class_id <= 9):
            raise ConfigError(
                f"lsc70w.yaml: label_map[{label!r}]={class_id} fuera de 0..9"
            )
        label_map[str(label)] = class_id

    default_rep = data.get("default_repetition_id", 0)
    if not isinstance(default_rep, int) or isinstance(default_rep, bool):
        raise ConfigError("lsc70w.yaml: 'default_repetition_id' debe ser int")

    source = data.get("source")
    source_version = data.get("source_version")
    subset = data.get("subset", source)
    if not isinstance(source, str) or not isinstance(source_version, str):
        raise ConfigError("lsc70w.yaml: 'source' y 'source_version' deben ser strings")

    return GrammarConfig(
        source=source,
        source_version=source_version,
        subset=str(subset),
        pattern=pattern,
        default_repetition_id=default_rep,
        label_map=label_map,
    )


class Lsc70wAdapter:
    """Interpreta rutas relativas de frames según la gramática del corpus."""

    def __init__(self, config: GrammarConfig) -> None:
        self._config = config

    @property
    def config(self) -> GrammarConfig:
        return self._config

    def parse_frame(self, frame_path: str) -> ParsedFrame:
        """Interpreta una ruta de frame; falla accionablemente si no casa.

        ``frame_path`` es una ruta relativa POSIX (se normalizan separadores
        ``\\`` de Windows). Si la regex no casa, o ``class_label`` no está en
        ``label_map``, se lanza ``ConfigError`` (design §9: nunca adivina).
        """
        normalized = frame_path.replace("\\", "/")
        match = self._config.pattern.match(normalized)
        if match is None:
            raise ConfigError(
                f"ruta no casa la gramática LSC70W: {frame_path!r}. "
                f"patrón esperado: {self._config.pattern.pattern!r}"
            )
        groups = match.groupdict()
        class_label = groups["class_label"]
        if class_label not in self._config.label_map:
            raise ConfigError(
                f"class_label {class_label!r} no está en label_map "
                f"(ruta {frame_path!r})"
            )
        repetition_raw = groups.get("repetition_id")
        repetition_id = (
            int(repetition_raw)
            if repetition_raw is not None
            else self._config.default_repetition_id
        )
        return ParsedFrame(
            signer_id=groups["signer_id"],
            class_label=class_label,
            class_id=self._config.label_map[class_label],
            repetition_id=repetition_id,
            frame_index=int(groups["frame_index"]),
            frame_path=normalized,
        )
