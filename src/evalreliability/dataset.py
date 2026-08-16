"""Versioned JSONL evaluation dataset loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalreliability.errors import DatasetValidationError
from evalreliability.models import DatasetDescriptor, EvaluationExample


@dataclass(frozen=True)
class EvaluationDataset:
    descriptor: DatasetDescriptor
    examples: tuple[EvaluationExample, ...]


def load_dataset(directory: str | Path, *, source: str | None = None) -> EvaluationDataset:
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_json_object(manifest_path, "dataset manifest")

    required = ("schema_version", "dataset_id", "version", "data_file")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise DatasetValidationError(f"manifest missing required fields: {', '.join(missing)}")
    invalid = [
        key for key in required if not isinstance(manifest[key], str) or not manifest[key].strip()
    ]
    if invalid:
        raise DatasetValidationError(
            f"manifest fields must be non-empty strings: {', '.join(invalid)}"
        )
    if manifest["schema_version"] != "1":
        raise DatasetValidationError(
            f"unsupported dataset schema_version {manifest['schema_version']!r}; expected '1'"
        )

    data_path = (root / str(manifest["data_file"])).resolve()
    if not data_path.is_relative_to(root):
        raise DatasetValidationError("manifest data_file must stay within the dataset directory")
    if not data_path.is_file():
        raise DatasetValidationError(f"dataset data file does not exist: {data_path}")

    examples: list[EvaluationExample] = []
    seen_ids: set[str] = set()
    with data_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = _loads_strict_json(raw_line, f"{data_path}:{line_number}")
            if not isinstance(record, dict):
                raise DatasetValidationError(f"{data_path}:{line_number} must be a JSON object")
            example = _parse_example(record, data_path, line_number)
            if example.id in seen_ids:
                raise DatasetValidationError(
                    f"{data_path}:{line_number} duplicates example id {example.id!r}"
                )
            seen_ids.add(example.id)
            examples.append(example)
    if not examples:
        raise DatasetValidationError("dataset must contain at least one example")

    canonical = json.dumps(
        {"manifest": manifest, "examples": [_example_payload(item) for item in examples]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    checksum = hashlib.sha256(canonical).hexdigest()
    known = {"schema_version", "dataset_id", "version", "data_file"}
    metadata = {key: value for key, value in manifest.items() if key not in known}
    descriptor = DatasetDescriptor(
        dataset_id=str(manifest["dataset_id"]),
        version=str(manifest["version"]),
        checksum_sha256=checksum,
        example_count=len(examples),
        source=source or str(Path(directory)),
        metadata=metadata,
    )
    return EvaluationDataset(descriptor=descriptor, examples=tuple(examples))


def select_examples(
    dataset: EvaluationDataset, example_ids: list[str] | tuple[str, ...]
) -> EvaluationDataset:
    """Return a provenance-preserving ordered subset for smoke and focused runs."""

    if not example_ids:
        raise DatasetValidationError("example selection must contain at least one ID")
    if not all(isinstance(example_id, str) and example_id for example_id in example_ids):
        raise DatasetValidationError("selected example IDs must be non-empty strings")
    if len(example_ids) != len(set(example_ids)):
        raise DatasetValidationError("selected example IDs must be unique")
    by_id = {example.id: example for example in dataset.examples}
    missing = [example_id for example_id in example_ids if example_id not in by_id]
    if missing:
        raise DatasetValidationError(f"selected example IDs not found: {', '.join(missing)}")

    selected = tuple(by_id[example_id] for example_id in example_ids)
    selection = {
        "parent_checksum_sha256": dataset.descriptor.checksum_sha256,
        "example_ids": list(example_ids),
    }
    canonical = json.dumps(selection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    metadata = dict(dataset.descriptor.metadata)
    metadata["selection"] = selection
    descriptor = DatasetDescriptor(
        dataset_id=dataset.descriptor.dataset_id,
        version=dataset.descriptor.version,
        checksum_sha256=hashlib.sha256(canonical).hexdigest(),
        example_count=len(selected),
        source=dataset.descriptor.source,
        metadata=metadata,
    )
    return EvaluationDataset(descriptor=descriptor, examples=selected)


def _parse_example(record: dict[str, Any], path: Path, line_number: int) -> EvaluationExample:
    example_id = record.get("id")
    prompt = record.get("input")
    if not isinstance(example_id, str) or not example_id.strip():
        raise DatasetValidationError(f"{path}:{line_number} requires a non-empty string id")
    if not isinstance(prompt, str):
        raise DatasetValidationError(f"{path}:{line_number} requires a string input")
    expected = record.get("expected", {})
    metadata = record.get("metadata", {})
    if not isinstance(expected, dict):
        raise DatasetValidationError(f"{path}:{line_number} expected must be an object")
    if not isinstance(metadata, dict):
        raise DatasetValidationError(f"{path}:{line_number} metadata must be an object")
    return EvaluationExample(example_id, prompt, expected, metadata)


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetValidationError(f"missing {description}: {path}")
    try:
        value = _loads_strict_json(path.read_text(encoding="utf-8"), str(path))
    except OSError as exc:
        raise DatasetValidationError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{description} must be a JSON object")
    return value


def _loads_strict_json(value: str, location: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-finite number {constant}")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise DatasetValidationError(f"invalid JSON in {location}: {exc}") from exc


def _example_payload(example: EvaluationExample) -> dict[str, Any]:
    return {
        "id": example.id,
        "input": example.input,
        "expected": example.expected,
        "metadata": example.metadata,
    }
