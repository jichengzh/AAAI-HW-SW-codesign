#!/usr/bin/env python3
"""Compile a public P4 external-resource registry from local-only inputs."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


_RELEASE_TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if _RELEASE_TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, _RELEASE_TOOL_DIRECTORY)

_registry_validator = importlib.import_module("validate_external_inputs")
REGISTRY_FORMAT = _registry_validator.REGISTRY_FORMAT


P4_DISPOSITION = "external_contract_p4"
RESOLUTION_FORMAT = "aaai27_p4_private_resource_resolution_v1"
COVERAGE_FORMAT = "aaai27_p4_external_resource_coverage_v1"

_PUBLIC_RECORD_FIELDS = (
    "input_id",
    "asset_kind",
    "source",
    "license",
    "version",
    "relative_path",
    "intended_use",
    "consumer_ids",
    "availability",
    "unavailable_reason",
    "sha256",
)


class CompilationError(ValueError):
    """Raised when local P3 inputs cannot safely produce public outputs."""


def _load_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompilationError(f"cannot load {label}") from error


def _required_string(record: Mapping[str, object], field: str, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise CompilationError(f"invalid {label}")
    return value


def _p4_candidate_keys(review_document: object) -> frozenset[tuple[str, str]]:
    if not isinstance(review_document, Mapping) or not isinstance(
        review_document.get("records"), list
    ):
        raise CompilationError("invalid local P3 review")

    keys: set[tuple[str, str]] = set()
    for record in review_document["records"]:
        if not isinstance(record, Mapping):
            raise CompilationError("invalid local P3 review")
        if record.get("disposition") != P4_DISPOSITION:
            continue
        key = (
            _required_string(record, "origin", "local P3 review"),
            _required_string(record, "path", "local P3 review"),
        )
        if key in keys:
            raise CompilationError("local P3 review has duplicate P4 candidates")
        keys.add(key)
    return frozenset(keys)


def _parse_resources(resolution: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw_resources = resolution.get("resources")
    if not isinstance(raw_resources, list):
        raise CompilationError("resolution resources must be a list")

    resources: dict[str, dict[str, object]] = {}
    for resource in raw_resources:
        if not isinstance(resource, Mapping):
            raise CompilationError("invalid resolution resource")
        public_record = {
            field: resource[field] for field in _PUBLIC_RECORD_FIELDS if field in resource
        }
        input_id = _required_string(public_record, "input_id", "resolution resource")
        if input_id in resources:
            raise CompilationError("resolution resource_id values must be unique")
        resources[input_id] = public_record

    return resources


def _parse_decisions(
    resolution: Mapping[str, object], candidates: frozenset[tuple[str, str]]
) -> tuple[dict[tuple[str, str], str | None], int, int]:
    raw_decisions = resolution.get("decisions")
    if not isinstance(raw_decisions, list):
        raise CompilationError("resolution decisions must be a list")

    decisions: dict[tuple[str, str], str | None] = {}
    document_count = 0
    asset_count = 0
    for record in raw_decisions:
        if not isinstance(record, Mapping):
            raise CompilationError("invalid resolution decision")
        key = (
            _required_string(record, "origin", "resolution decision"),
            _required_string(record, "path", "resolution decision"),
        )
        if key in decisions:
            raise CompilationError("resolution has duplicate decisions")
        if key not in candidates:
            raise CompilationError("resolution includes a non-P4 decision")

        decision = record.get("decision")
        if decision == "document":
            if "resource_id" in record:
                raise CompilationError("document decision must not include resource_id")
            decisions[key] = None
            document_count += 1
        elif decision == "asset":
            decisions[key] = _required_string(record, "resource_id", "asset decision")
            asset_count += 1
        else:
            raise CompilationError("resolution decision is invalid")

    if set(decisions) != candidates:
        raise CompilationError("resolution must cover every P4 candidate")
    return decisions, document_count, asset_count


def _registry_record(resource: Mapping[str, object]) -> dict[str, object]:
    return {field: resource[field] for field in _PUBLIC_RECORD_FIELDS if field in resource}


def _reject_private_consumer_ids(
    resources: Mapping[str, Mapping[str, object]],
    candidates: frozenset[tuple[str, str]],
) -> None:
    private_tokens = {value for candidate in candidates for value in candidate}
    for resource in resources.values():
        consumer_ids = resource.get("consumer_ids")
        if isinstance(consumer_ids, list) and any(
            isinstance(consumer_id, str)
            and any(token in consumer_id for token in private_tokens)
            for consumer_id in consumer_ids
        ):
            raise CompilationError("resource consumer_ids must not contain private P3 keys")


def _contains_private_token(value: object, private_tokens: set[str]) -> bool:
    if isinstance(value, str):
        return any(token in value for token in private_tokens)
    if isinstance(value, Mapping):
        return any(_contains_private_token(item, private_tokens) for item in value.values())
    if isinstance(value, list):
        return any(_contains_private_token(item, private_tokens) for item in value)
    return False


def _reject_private_tokens_in_public_resources(
    resources: Mapping[str, Mapping[str, object]],
    candidates: frozenset[tuple[str, str]],
) -> None:
    private_tokens = {value for candidate in candidates for value in candidate}
    if any(
        _contains_private_token(_registry_record(resource), private_tokens)
        for resource in resources.values()
    ):
        raise CompilationError("public resource records must not contain private P3 keys")


def _coverage(
    candidate_count: int,
    document_count: int,
    asset_count: int,
    registry_inputs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    asset_kind_counts = Counter(
        str(record["asset_kind"]) for record in registry_inputs
    )
    availability_counts = Counter(
        str(record["availability"]) for record in registry_inputs
    )
    return {
        "format": COVERAGE_FORMAT,
        "p3_p4_candidate_count": candidate_count,
        "document_candidate_count": document_count,
        "asset_candidate_count": asset_count,
        "distinct_resource_count": len(registry_inputs),
        "asset_kind_counts": [
            {"asset_kind": asset_kind, "count": count}
            for asset_kind, count in sorted(asset_kind_counts.items())
        ],
        "availability_counts": [
            {"availability": availability, "count": count}
            for availability, count in sorted(availability_counts.items())
        ],
    }


def compile_registry(
    review_path: Path, resolution_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Compile validated, path-free public outputs from local-only P3 inputs."""
    candidates = _p4_candidate_keys(_load_json(review_path, "local P3 review"))
    resolution = _load_json(resolution_path, "local resolution")
    if not isinstance(resolution, Mapping) or resolution.get("format") != RESOLUTION_FORMAT:
        raise CompilationError("resolution header is invalid")

    resources = _parse_resources(resolution)
    decisions, document_count, asset_count = _parse_decisions(resolution, candidates)
    _reject_private_consumer_ids(resources, candidates)
    _reject_private_tokens_in_public_resources(resources, candidates)
    used_resource_ids = {resource_id for resource_id in decisions.values() if resource_id}
    if not used_resource_ids.issubset(resources):
        raise CompilationError("asset decision references an unknown resource_id")
    if set(resources) != used_resource_ids:
        raise CompilationError("every resource must be used by an asset decision")

    registry_inputs = sorted(
        (_registry_record(resources[resource_id]) for resource_id in used_resource_ids),
        key=lambda record: str(record["input_id"]),
    )
    registry = {
        "format": REGISTRY_FORMAT,
        "registry_version": 1,
        "inputs": registry_inputs,
    }
    try:
        _registry_validator.parse_registry_document(registry)
    except _registry_validator.RegistryError as error:
        raise CompilationError("resolution resources are not valid public registry records") from error

    return _registry_document(registry), _coverage(
        len(candidates), document_count, asset_count, registry_inputs
    )


def _registry_document(registry: Mapping[str, object]) -> dict[str, object]:
    """Return a fresh public registry document after validation."""
    return {
        "format": registry["format"],
        "registry_version": registry["registry_version"],
        "inputs": list(registry["inputs"]),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p3-review", required=True, type=Path)
    parser.add_argument("--resolution", required=True, type=Path)
    parser.add_argument("--registry-output", required=True, type=Path)
    parser.add_argument("--coverage-output", required=True, type=Path)
    return parser.parse_args(argv)


def _stage_json(path: Path, document: Mapping[str, object]) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_file.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return Path(temporary_file.name)


def _backup_existing_output(path: Path) -> Path | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise CompilationError("public output path must be a file")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".backup",
        delete=False,
    ) as backup_file:
        backup_file.write(path.read_bytes())
        return Path(backup_file.name)


def _restore_output(path: Path, backup: Path | None) -> None:
    try:
        if backup is not None:
            backup.replace(path)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def _remove_temporary_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _write_public_outputs(
    registry_output: Path,
    coverage_output: Path,
    registry: Mapping[str, object],
    coverage: Mapping[str, object],
) -> None:
    if registry_output.resolve() == coverage_output.resolve():
        raise CompilationError("public output paths must be distinct")

    registry_temporary: Path | None = None
    coverage_temporary: Path | None = None
    registry_backup: Path | None = None
    coverage_backup: Path | None = None
    try:
        registry_output.parent.mkdir(parents=True, exist_ok=True)
        coverage_output.parent.mkdir(parents=True, exist_ok=True)
        if registry_output.exists() and not registry_output.is_file():
            raise CompilationError("public output path must be a file")
        if coverage_output.exists() and not coverage_output.is_file():
            raise CompilationError("public output path must be a file")
        registry_temporary = _stage_json(registry_output, registry)
        coverage_temporary = _stage_json(coverage_output, coverage)
        registry_backup = _backup_existing_output(registry_output)
        coverage_backup = _backup_existing_output(coverage_output)
        registry_temporary.replace(registry_output)
        registry_temporary = None
        coverage_temporary.replace(coverage_output)
        coverage_temporary = None
    except OSError as error:
        _restore_output(registry_output, registry_backup)
        _restore_output(coverage_output, coverage_backup)
        registry_backup = None
        coverage_backup = None
        raise CompilationError("cannot write public outputs") from error
    finally:
        for temporary_path in (
            registry_temporary,
            coverage_temporary,
            registry_backup,
            coverage_backup,
        ):
            _remove_temporary_file(temporary_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        registry, coverage = compile_registry(args.p3_review, args.resolution)
        _write_public_outputs(
            args.registry_output, args.coverage_output, registry, coverage
        )
    except CompilationError:
        print("p4 external registry compilation failed", file=sys.stderr)
        return 2

    print(
        "p4 external registry compiled: "
        f"p3_p4_candidates={coverage['p3_p4_candidate_count']} "
        f"documents={coverage['document_candidate_count']} "
        f"assets={coverage['asset_candidate_count']} "
        f"resources={coverage['distinct_resource_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
