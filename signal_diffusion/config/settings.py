"""Settings loader for the Signal Diffusion project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import tomllib
from typing import Any, Mapping

CONFIG_ENV_VAR = "SIGNAL_DIFFUSION_CONFIG"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "default.toml"


@dataclass(slots=True)
class DatasetSettings:
    """Filesystem locations for a specific dataset."""

    name: str
    root: Path
    output: Path
    timeseries_output: Path | None = None
    min_db: float | None = None
    max_db: float | None = None

    def resolve(self, *parts: os.PathLike[str] | str) -> Path:
        """Return a path inside the dataset root."""
        return self.root.joinpath(*parts)

    def resolve_output(self, *parts: os.PathLike[str] | str) -> Path:
        """Return a path inside the dataset output directory."""
        return self.output.joinpath(*parts)


@dataclass(slots=True)
class Settings:
    """Project-wide settings derived from TOML configuration."""

    config_path: Path
    data_root: Path
    output_root: Path
    max_sampling_weight: float | None
    output_type: str = "db-only"
    data_type: str = "spectrogram"  # "spectrogram" or "timeseries"
    datasets: dict[str, DatasetSettings] = field(default_factory=dict)
    # Model paths for shared resources (VAE, text encoders, etc.)
    models: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.data_type not in {"spectrogram", "timeseries"}:
            raise ValueError(
                f"data_type must be 'spectrogram' or 'timeseries', got {self.data_type!r}"
            )

    def dataset(self, name: str) -> DatasetSettings:
        try:
            return self.datasets[name]
        except KeyError as exc:
            raise KeyError(f"Dataset '{name}' is not configured. Available: {sorted(self.datasets)}") from exc

    def dataset_path(self, name: str) -> Path:
        return self.dataset(name).root

    def dataset_output_path(self, name: str) -> Path:
        return self.dataset(name).output

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, config_path: Path) -> "Settings":
        config_dir = config_path.parent
        data_section = mapping.get("data", {})
        data_root = _expand_root(data_section.get("root", "."), base=config_dir)
        output_root = _expand_root(data_section.get("output_root", data_root), base=config_dir)
        max_sampling_weight = data_section.get("max_sampling_weight", None)
        output_type = data_section.get("output_type", "db-only")
        data_type = data_section.get("data_type", "spectrogram")

        # Load shared model paths
        models_section = mapping.get("models", {})
        models: dict[str, str] = {}
        if isinstance(models_section, Mapping):
            for key, value in models_section.items():
                if value is not None:
                    models[str(key)] = str(value)

        datasets_section = mapping.get("datasets", {})
        datasets: dict[str, DatasetSettings] = {}
        for name, section in datasets_section.items():
            if not isinstance(section, Mapping):
                raise TypeError(f"Dataset entry '{name}' must be a mapping, got {type(section)!r}")
            raw_root = section.get("root")
            raw_output = section.get("output")
            raw_timeseries_output = section.get("timeseries_output")
            min_db = _parse_db_bound(section.get("min_db"), dataset=name, field="min_db")
            max_db = _parse_db_bound(section.get("max_db"), dataset=name, field="max_db")
            if min_db is not None and max_db is not None and max_db <= min_db:
                raise ValueError(
                    f"Dataset '{name}' max_db must be greater than min_db (got {min_db}..{max_db})"
                )
            dataset_root = _expand_dataset_path(
                raw_root,
                dataset=name,
                base_dir=config_dir,
                default_base=data_root,
                fallback=data_root / name,
            )
            dataset_output = _expand_dataset_path(
                raw_output,
                dataset=name,
                base_dir=config_dir,
                default_base=output_root,
                fallback=output_root / name,
            )
            timeseries_output = _expand_dataset_path(
                raw_timeseries_output,
                dataset=name,
                base_dir=config_dir,
                default_base=output_root,
                fallback=dataset_output.parent / "timeseries",
            )
            datasets[name] = DatasetSettings(
                name=name,
                root=dataset_root,
                output=dataset_output,
                timeseries_output=timeseries_output,
                min_db=min_db,
                max_db=max_db,
            )

        return cls(
            config_path=config_path,
            data_root=data_root,
            output_root=output_root,
            max_sampling_weight=max_sampling_weight,
            output_type=output_type,
            data_type=data_type,
            datasets=datasets,
            models=models,
        )


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    """Load settings from TOML.

    Resolution order:
    1. Explicit *path* parameter.
    2. Environment variable ``SIGNAL_DIFFUSION_CONFIG``.
    3. Repository default ``config/default.toml``.
    """
    config_path = _determine_config_path(path)
    with config_path.open("rb") as fp:
        data = tomllib.load(fp)
    return Settings.from_mapping(data, config_path=config_path)


def _determine_config_path(path: str | os.PathLike[str] | None) -> Path:
    if path is not None:
        candidate = Path(path)
    else:
        env_path = os.getenv(CONFIG_ENV_VAR)
        candidate = Path(env_path) if env_path else _DEFAULT_CONFIG
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        candidate = (_REPO_ROOT / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Configuration file not found: {candidate}")
    return candidate


def _expand_root(value: Any, *, base: Path) -> Path:
    if isinstance(value, (str, os.PathLike)):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (base / path).resolve()
        return path
    if isinstance(value, Path):
        return value
    raise TypeError(f"Expected string or path-like, got {type(value)!r}")


def _expand_dataset_path(
    value: Any,
    *,
    dataset: str,
    base_dir: Path,
    default_base: Path,
    fallback: Path,
) -> Path:
    if value in (None, ""):
        return fallback.resolve()

    if isinstance(value, (str, os.PathLike)):
        path = Path(value).expanduser()
    elif isinstance(value, Path):
        path = value
    else:
        raise TypeError(f"Dataset '{dataset}' path must be string or path-like, got {type(value)!r}")

    if not path.is_absolute():
        text = str(path)
        if text.startswith("./") or text.startswith("../"):
            path = (base_dir / path).resolve()
        else:
            path = (default_base / path).resolve()
    return path


def _parse_db_bound(value: Any, *, dataset: str, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Dataset '{dataset}' {field} must be numeric, got {type(value)!r}")
