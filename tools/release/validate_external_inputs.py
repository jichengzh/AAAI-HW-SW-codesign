from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REGISTRY_FORMAT = "aaai27_external_input_registry_v1"
RESULT_FORMAT = "aaai27_external_input_validation_v1"

_LICENSE_STATUSES = frozenset({"confirmed", "unconfirmed", "not_redistributable"})
_AVAILABILITY_STATUSES = frozenset({"available_for_verification", "unavailable"})


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ExternalInput:
    input_id: str
    asset_kind: str
    source: str
    license_status: str
    license_reference: str
    version: str
    relative_path: str
    intended_use: str
    consumer_ids: tuple[str, ...]
    availability: str
    unavailable_reason: str | None
    sha256: str | None


@dataclass(frozen=True)
class InputResult:
    input_id: str
    relative_path: str
    status: str
    reason: str | None


def _required_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{field} must be a non-empty string")
    return value


def _optional_sha256(record: Mapping[str, Any]) -> str | None:
    value = record.get("sha256")
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RegistryError("sha256 must be a lowercase 64-character hexadecimal string")
    return value


def _relative_asset_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RegistryError("relative_path must stay within asset-root")
    return value


def _parse_record(record: object) -> ExternalInput:
    if not isinstance(record, Mapping):
        raise RegistryError("each input must be an object")

    license_data = record.get("license")
    if not isinstance(license_data, Mapping):
        raise RegistryError("license must be an object")
    license_status = license_data.get("status")
    if license_status not in _LICENSE_STATUSES:
        raise RegistryError("license.status is invalid")
    license_reference = license_data.get("reference")
    if not isinstance(license_reference, str):
        raise RegistryError("license.reference must be a string")

    consumer_ids = record.get("consumer_ids")
    if not isinstance(consumer_ids, list) or not all(
        isinstance(consumer_id, str) for consumer_id in consumer_ids
    ):
        raise RegistryError("consumer_ids must be a list of strings")

    availability = record.get("availability")
    if availability not in _AVAILABILITY_STATUSES:
        raise RegistryError("availability is invalid")
    unavailable_reason = record.get("unavailable_reason")
    if availability == "unavailable":
        if not isinstance(unavailable_reason, str) or not unavailable_reason:
            raise RegistryError("unavailable_reason is required when unavailable")

    return ExternalInput(
        input_id=_required_string(record, "input_id"),
        asset_kind=_required_string(record, "asset_kind"),
        source=_required_string(record, "source"),
        license_status=license_status,
        license_reference=license_reference,
        version=_required_string(record, "version"),
        relative_path=_relative_asset_path(_required_string(record, "relative_path")),
        intended_use=_required_string(record, "intended_use"),
        consumer_ids=tuple(consumer_ids),
        availability=availability,
        unavailable_reason=unavailable_reason,
        sha256=_optional_sha256(record),
    )


def load_registry(path: Path) -> tuple[ExternalInput, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"cannot load registry: {error}") from error

    return parse_registry_document(document)


def parse_registry_document(document: object) -> tuple[ExternalInput, ...]:
    """Parse and validate a public external-input registry document."""

    if not isinstance(document, Mapping):
        raise RegistryError("registry must be an object")
    if document.get("format") != REGISTRY_FORMAT or document.get("registry_version") != 1:
        raise RegistryError("registry header is invalid")
    records = document.get("inputs")
    if not isinstance(records, list):
        raise RegistryError("inputs must be a list")

    inputs = tuple(_parse_record(record) for record in records)
    if len({item.input_id for item in inputs}) != len(inputs):
        raise RegistryError("input_id values must be unique")
    return inputs


def validate_inputs(
    inputs: tuple[ExternalInput, ...], asset_root: Path
) -> tuple[InputResult, ...]:
    results: list[InputResult] = []
    for item in inputs:
        if item.availability == "unavailable":
            results.append(
                InputResult(
                    item.input_id,
                    item.relative_path,
                    "unavailable",
                    item.unavailable_reason,
                )
            )
            continue
        target = asset_root / item.relative_path
        if not target.is_file():
            results.append(
                InputResult(item.input_id, item.relative_path, "unavailable", "asset_missing")
            )
        elif (
            item.sha256 is not None
            and hashlib.sha256(target.read_bytes()).hexdigest() != item.sha256
        ):
            results.append(
                InputResult(
                    item.input_id,
                    item.relative_path,
                    "unavailable",
                    "asset_sha256_mismatch",
                )
            )
        else:
            results.append(InputResult(item.input_id, item.relative_path, "verified", None))
    return tuple(results)


def render_result(results: tuple[InputResult, ...]) -> dict[str, object]:
    status = "verified" if all(item.status == "verified" for item in results) else "unavailable"
    return {
        "format": RESULT_FORMAT,
        "status": status,
        "inputs": [
            {
                "input_id": item.input_id,
                "relative_path": item.relative_path,
                "status": item.status,
                "reason": item.reason,
            }
            for item in results
        ],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _write_json(output: Path, payload: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = render_result(validate_inputs(load_registry(args.registry), args.asset_root))
    except RegistryError:
        payload = {
            "format": RESULT_FORMAT,
            "status": "unavailable",
            "inputs": [],
            "reason": "registry_invalid",
        }
    _write_json(args.output, payload)
    return 0 if payload["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
