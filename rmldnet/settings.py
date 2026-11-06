"""Validated portable platform configuration."""

from dataclasses import dataclass
from pathlib import Path
import os
import yaml


@dataclass(frozen=True)
class Settings:
    database: str = "runs/platform.sqlite3"
    bundle: str = "results/via"
    output: str = "runs/platform"
    data: str = "data/via"
    seed: int = 7
    bootstrap_replicates: int = 2000

    @classmethod
    def load(cls, path=None):
        values = yaml.safe_load(Path(path).read_text()) if path else {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        if set(values) - allowed:
            raise ValueError(f"Unknown settings: {sorted(set(values)-allowed)}")
        if "RMLDNET_DB" in os.environ:
            values["database"] = os.environ["RMLDNET_DB"]
        result = cls(**values)
        if (
            type(result.seed) is not int
            or type(result.bootstrap_replicates) is not int
            or result.seed < 0
            or result.bootstrap_replicates < 100
        ):
            raise ValueError("Invalid reproducibility settings")
        for key in ["database", "bundle", "output", "data"]:
            if not isinstance(getattr(result, key), str) or not getattr(result, key):
                raise ValueError(f"{key} must be a nonempty path")
        return result
