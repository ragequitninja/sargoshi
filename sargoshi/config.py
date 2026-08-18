"""Typed service configuration.

The service is configured by a single YAML file that the operator provides.
There are no built-in defaults: the config file is REQUIRED and must define
every key. A missing file or a missing/invalid key raises `ConfigError`,
and the service must not start. Defaults are documented in config.example.yaml
"""

from __future__ import annotations

import asyncio
import copy
import os
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "Config",
    "ConfigService",
    "BackendConfig",
    "FrontendsConfig",
    "ListenerConfig",
    "WyomingConfig",
    "SpeakerConfig",
    "StorageConfig",
    "LoggingConfig",
    "ConfigError",
    "load_config",
    "update_config_file",
]

# Speaker-label delivery modes.
DELIVERY_MODES = ("metadata", "inline", "structured")
# Valid root logging levels (logging.level).
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


# ---------------------------------------------------------------------------
# Typed config structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackendConfig:
    device: str
    model: str
    compute_type: str
    language: str | None
    beam_size: int
    word_timestamps: bool
    cpu_threads: int
    num_workers: int


@dataclass(frozen=True, slots=True)
class ListenerConfig:
    """A host:port frontend (the management UI)."""

    enabled: bool
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class WyomingConfig:
    enabled: bool
    uri: str


@dataclass(frozen=True, slots=True)
class FrontendsConfig:
    wyoming: WyomingConfig
    ui: ListenerConfig


@dataclass(frozen=True, slots=True)
class SpeakerConfig:
    enabled: bool
    model: str
    device: str
    threshold: float
    db_path: str
    delivery: tuple[str, ...]
    inline_template: str
    unknown_label: str


@dataclass(frozen=True, slots=True)
class StorageConfig:
    model_cache: str


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True, slots=True)
class Config:
    logging: LoggingConfig
    backend: BackendConfig
    frontends: FrontendsConfig
    speaker: SpeakerConfig
    storage: StorageConfig

    # -- constructors

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> Config:
        """Build from a COMPLETE nested mapping. Every key is required."""
        if not isinstance(data, Mapping):
            raise ConfigError("config must be a mapping at the top level")

        lg = _section(data, "logging")
        b = _section(data, "backend")
        f = _section(data, "frontends")
        s = _section(data, "speaker")
        st = _section(data, "storage")

        backend = BackendConfig(
            device=_backend_device(_req(b, "device", "backend.device")),
            model=_nonempty_str(_req(b, "model", "backend.model"), "backend.model"),
            compute_type=_nonempty_str(_req(b, "compute_type", "backend.compute_type"), "backend.compute_type"),
            language=_opt_str(_req(b, "language", "backend.language")),
            beam_size=_positive_int(_req(b, "beam_size", "backend.beam_size"), "backend.beam_size"),
            word_timestamps=_to_bool(_req(b, "word_timestamps", "backend.word_timestamps"), "backend.word_timestamps"),
            cpu_threads=_nonneg_int(_req(b, "cpu_threads", "backend.cpu_threads"), "backend.cpu_threads"),
            num_workers=_positive_int(_req(b, "num_workers", "backend.num_workers"), "backend.num_workers"),
        )
        frontends = FrontendsConfig(
            wyoming=_wyoming(_section(f, "wyoming", "frontends.wyoming")),
            ui=_listener(_section(f, "ui", "frontends.ui"), "frontends.ui"),
        )
        speaker = SpeakerConfig(
            enabled=_to_bool(_req(s, "enabled", "speaker.enabled"), "speaker.enabled"),
            model=_nonempty_str(_req(s, "model", "speaker.model"), "speaker.model"),
            device=_speaker_device(_req(s, "device", "speaker.device")),
            threshold=_unit_float(_req(s, "threshold", "speaker.threshold"), "speaker.threshold"),
            db_path=_db_path(_req(s, "db_path", "speaker.db_path")),
            delivery=_delivery(_req(s, "delivery", "speaker.delivery")),
            inline_template=str(_req(s, "inline_template", "speaker.inline_template")),
            unknown_label=str(_req(s, "unknown_label", "speaker.unknown_label")),
        )
        storage = StorageConfig(
            model_cache=_nonempty_str(_req(st, "model_cache", "storage.model_cache"), "storage.model_cache"),
        )
        logging_cfg = LoggingConfig(level=_log_level(_req(lg, "level", "logging.level")))
        return cls(
            logging=logging_cfg,
            backend=backend,
            frontends=frontends,
            speaker=speaker,
            storage=storage,
        )

    # -- serialization

    def to_dict(self) -> dict[str, object]:
        """Canonical nested dict (for the UI Config page / YAML persistence)."""
        return {
            "logging": {"level": self.logging.level},
            "backend": {
                "device": self.backend.device,
                "model": self.backend.model,
                "compute_type": self.backend.compute_type,
                "language": self.backend.language,
                "beam_size": self.backend.beam_size,
                "word_timestamps": self.backend.word_timestamps,
                "cpu_threads": self.backend.cpu_threads,
                "num_workers": self.backend.num_workers,
            },
            "frontends": {
                "wyoming": {
                    "enabled": self.frontends.wyoming.enabled,
                    "uri": self.frontends.wyoming.uri,
                },
                "ui": _listener_dict(self.frontends.ui),
            },
            "speaker": {
                "enabled": self.speaker.enabled,
                "model": self.speaker.model,
                "device": self.speaker.device,
                "threshold": self.speaker.threshold,
                "db_path": self.speaker.db_path,
                "delivery": list(self.speaker.delivery),
                "inline_template": self.speaker.inline_template,
                "unknown_label": self.speaker.unknown_label,
            },
            "storage": {
                "model_cache": self.storage.model_cache,
            },
        }


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------


def load_config(path: str | os.PathLike[str] | None) -> Config:
    """Load and validate the config from `path`.

    The file is REQUIRED and never auto-created. A missing file, or an
    incomplete config, raises `ConfigError` and the service must not start.
    """
    if not path:
        raise ConfigError("no config file specified (use --config PATH)")
    p = pathlib.Path(path)
    if not p.is_file():
        raise ConfigError(
            f"config file not found: {p} — create it (see the config.example.yaml "
            f"for a full example listing every key and its default)."
        )
    return Config.from_dict(_read_yaml(p))


def update_config_file(changes: Mapping[str, object], path: str | os.PathLike[str]) -> None:
    """Apply `changes` to an existing YAML file, preserving comments/formatting.

    Round-trips the file through ruamel.yaml so untouched keys keep their exact
    comments, ordering, quoting, and layout; only the changed leaf values are
    rewritten. Atomic (temp file and rename). The file must already exist.
    """
    from ruamel.yaml import YAML

    p = pathlib.Path(path)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.representer.add_representer(
        type(None),
        lambda rep, _data: rep.represent_scalar("tag:yaml.org,2002:null", "null"),
    )
    with p.open(encoding="utf-8") as fh:
        doc = yaml.load(fh)
    if doc is None:
        raise ConfigError(f"{p}: config file is empty")
    if not isinstance(doc, Mapping):
        raise ConfigError(f"{p}: expected a YAML mapping at the top level")
    _deep_merge(doc, changes)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        yaml.dump(doc, fh)
    os.replace(tmp, p)


def _read_yaml(path: str | os.PathLike[str]) -> dict:
    p = pathlib.Path(path)
    import yaml

    with p.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ConfigError(f"{p}: config file is empty")
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: expected a YAML mapping at the top level")
    return data


# ---------------------------------------------------------------------------
# Runtime config service
# ---------------------------------------------------------------------------


class ConfigService:
    """Holds the current `Config` and persists runtime changes to the YAML file.

    A runtime edit such as switching the active model in the UI will survive a restart.
    """

    def __init__(self, config: Config, path: str | os.PathLike[str]) -> None:
        self._config = config
        self._path = pathlib.Path(path)
        self._lock = asyncio.Lock()

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ConfigService:
        return cls(load_config(path), path)

    @property
    def current(self) -> Config:
        """The live config. Read this per request so edits take effect at once."""
        return self._config

    @property
    def path(self) -> pathlib.Path:
        return self._path

    async def update(self, changes: Mapping[str, object]) -> Config:
        """Merge `changes` into the current config, validate, persist, then swap.

        Serialized against concurrent writers. Raises `ConfigError` (leaving both
        memory and file untouched) if the merged result is invalid.
        """
        async with self._lock:
            merged = _deep_merge(copy.deepcopy(self._config.to_dict()), changes)
            new = Config.from_dict(merged)
            await asyncio.to_thread(update_config_file, changes, self._path)
            self._config = new
            return new

    async def set(self, dotted_key: str, value: object) -> Config:
        """Update a single dotted key, e.g. ``set("backend.model", "small")``."""
        return await self.update(_nest(dotted_key, value))

    async def save(self) -> None:
        """Re-persist the current config as-is, preserving comments and formatting."""
        async with self._lock:
            await asyncio.to_thread(update_config_file, self._config.to_dict(), self._path)


def _nest(dotted_key: str, value: object) -> dict:
    node: dict = {}
    cursor = node
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return node


def _deep_merge(base, override: Mapping):
    """Recursively merge `override` into `base` in place.

    `base` may be a plain dict or a ruamel round-trip mapping (`CommentedMap`);
    the `Mapping` check keeps nested comment-carrying maps intact so only leaf
    values are replaced.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], Mapping) and isinstance(value, Mapping):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# Field coercion / validation
# ---------------------------------------------------------------------------


def _section(data: Mapping, key: str, path: str | None = None) -> Mapping:
    path = path or key
    node = data.get(key)
    if not isinstance(node, Mapping):
        raise ConfigError(f"missing or invalid section: {path}")
    return node


def _req(node: Mapping, key: str, path: str) -> object:
    if key not in node:
        raise ConfigError(f"missing required key: {path}")
    return node[key]


def _device(value: object, key: str, supported: tuple[str, ...]) -> str:
    """Validate a device name against a subsystem's supported set.

    Each subsystem owns which devices it can actually run (the CTranslate2 STT
    backend can't use ``rocm`` or ``openvino``; the speaker embedders accept the
    torch devices, and WavLM additionally accepts the ``openvino*`` targets). The
    speaker set is the union across models, so an out-of-set value errors here at
    config load; a valid-but-model-incompatible pair (e.g. ECAPA + ``openvino``) is
    caught when the embedder is built at startup.
    """
    v = str(value).strip().lower()
    if v not in supported:
        raise ConfigError(f"{key}: {v!r} is not supported by this subsystem; expected one of {', '.join(supported)}")
    return v


def _backend_device(value: object) -> str:
    from .backends import SUPPORTED_DEVICES

    return _device(value, "backend.device", SUPPORTED_DEVICES)


def _speaker_device(value: object) -> str:
    from .speaker.embed import SUPPORTED_DEVICES

    return _device(value, "speaker.device", SUPPORTED_DEVICES)


def _delivery(value: object) -> tuple[str, ...]:
    modes = _to_str_list(value, "speaker.delivery")
    for mode in modes:
        if mode not in DELIVERY_MODES:
            raise ConfigError(f"speaker.delivery: unknown mode {mode!r}; expected any of {', '.join(DELIVERY_MODES)}")
    return modes


def _log_level(value: object) -> str:
    v = str(value).strip().upper()
    if v not in LOG_LEVELS:
        raise ConfigError(f"logging.level: expected one of {', '.join(LOG_LEVELS)}, got {value!r}")
    return v


def _wyoming(node: Mapping[str, object]) -> WyomingConfig:
    enabled = _to_bool(_req(node, "enabled", "frontends.wyoming.enabled"), "frontends.wyoming.enabled")
    uri = str(_req(node, "uri", "frontends.wyoming.uri")).strip()
    if enabled and not (uri.startswith("tcp://") or uri.startswith("unix://")):
        raise ConfigError(f"frontends.wyoming.uri: expected a tcp:// or unix:// URI, got {uri!r}")
    return WyomingConfig(enabled=enabled, uri=uri)


def _listener(node: Mapping[str, object], name: str) -> ListenerConfig:
    return ListenerConfig(
        enabled=_to_bool(_req(node, "enabled", f"{name}.enabled"), f"{name}.enabled"),
        host=_nonempty_str(_req(node, "host", f"{name}.host"), f"{name}.host"),
        port=_port(_req(node, "port", f"{name}.port"), f"{name}.port"),
    )


def _listener_dict(cfg: ListenerConfig) -> dict[str, object]:
    return {"enabled": cfg.enabled, "host": cfg.host, "port": cfg.port}


def _to_bool(value: object, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off", ""}:
            return False
    raise ConfigError(f"{key}: expected a boolean, got {value!r}")


def _to_int(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{key}: expected an integer, got boolean {value!r}")
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ConfigError(f"{key}: expected an integer, got {value!r}") from None


def _positive_int(value: object, key: str) -> int:
    n = _to_int(value, key)
    if n < 1:
        raise ConfigError(f"{key}: expected a positive integer, got {n}")
    return n


def _nonneg_int(value: object, key: str) -> int:
    n = _to_int(value, key)
    if n < 0:
        raise ConfigError(f"{key}: expected a non-negative integer, got {n}")
    return n


def _port(value: object, key: str) -> int:
    n = _to_int(value, key)
    if not 1 <= n <= 65535:
        raise ConfigError(f"{key}: port out of range 1-65535, got {n}")
    return n


def _unit_float(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{key}: expected a number, got boolean {value!r}")
    try:
        f = float(str(value).strip())
    except (TypeError, ValueError):
        raise ConfigError(f"{key}: expected a number, got {value!r}") from None
    if not 0.0 <= f <= 1.0:
        raise ConfigError(f"{key}: expected a value in [0.0, 1.0], got {f}")
    return f


def _nonempty_str(value: object, key: str) -> str:
    s = str(value).strip()
    if s == "":
        raise ConfigError(f"{key}: must not be empty")
    return s


def _db_path(value: object) -> str:
    """Validate the voiceprint DB path: a non-empty, file-backed SQLite path.

    In-memory SQLite (``:memory:``) is rejected — the store must persist to a
    file so voiceprints survive restarts and can be migrated by Alembic.
    """
    s = _nonempty_str(value, "speaker.db_path")
    if s == ":memory:":
        raise ConfigError("speaker.db_path: in-memory SQLite is not supported; use a file path")
    return s


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _to_str_list(value: object, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise ConfigError(f"{key}: expected a list or comma-separated string, got {value!r}")
