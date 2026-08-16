"""Atomic persistence for machine-readable run artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from evalreliability.errors import ArtifactError
from evalreliability.models import EvaluationRun


def write_run(run: EvaluationRun, path: str | Path) -> Path:
    return write_json(run.to_dict(), path)


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write any JSON object atomically with finite-number enforcement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except (OSError, TypeError, ValueError) as exc:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        raise ArtifactError(f"could not write run artifact {destination}: {exc}") from exc
    return destination


def read_run(path: str | Path) -> EvaluationRun:
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("artifact root must be an object")
        run = EvaluationRun.from_dict(payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"could not read run artifact {source}: {exc}") from exc
    if run.metadata.schema_version != "1.0":
        raise ArtifactError(
            f"unsupported run schema_version {run.metadata.schema_version!r}; expected '1.0'"
        )
    if len(run.examples) != run.metadata.dataset.example_count:
        raise ArtifactError("artifact example count does not match dataset metadata")
    return run
