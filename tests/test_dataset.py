from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalreliability.dataset import load_dataset
from evalreliability.errors import DatasetValidationError


def _write_dataset(root: Path, records: list[dict[str, object]]) -> None:
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dataset_id": "contract-tests",
                "version": "1.2.0",
                "data_file": "data.jsonl",
                "provenance": "test fixture",
            }
        ),
        encoding="utf-8",
    )
    (root / "data.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_load_dataset_validates_and_hashes_canonical_content(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    records = [
        {
            "id": "one",
            "input": "Question",
            "expected": {"text": "Answer"},
            "metadata": {"slices": {"domain": "unit"}},
        }
    ]
    _write_dataset(root, records)

    first = load_dataset(root)
    second = load_dataset(root)

    assert first.descriptor.dataset_id == "contract-tests"
    assert first.descriptor.version == "1.2.0"
    assert first.descriptor.example_count == 1
    assert first.descriptor.checksum_sha256 == second.descriptor.checksum_sha256
    assert len(first.descriptor.checksum_sha256) == 64
    assert first.examples[0].expected == {"text": "Answer"}


def test_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(
        root,
        [
            {"id": "duplicate", "input": "a"},
            {"id": "duplicate", "input": "b"},
        ],
    )

    with pytest.raises(DatasetValidationError, match="duplicates example id"):
        load_dataset(root)


def test_dataset_rejects_non_finite_numbers(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(root, [{"id": "one", "input": "a"}])
    (root / "data.jsonl").write_text(
        '{"id":"one","input":"a","expected":{"score":NaN}}\n', encoding="utf-8"
    )

    with pytest.raises(DatasetValidationError, match="non-finite"):
        load_dataset(root)


def test_dataset_rejects_manifest_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (tmp_path / "outside.jsonl").write_text('{"id":"one","input":"a"}\n', encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dataset_id": "escape",
                "version": "1.0.0",
                "data_file": "../outside.jsonl",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="stay within"):
        load_dataset(root)


def test_dataset_rejects_non_string_manifest_identity(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(root, [{"id": "one", "input": "a"}])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["version"] = 1
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="non-empty strings: version"):
        load_dataset(root)
