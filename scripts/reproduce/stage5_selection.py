#!/usr/bin/env python3
"""Train the frozen Stage5 bundle and publish selection-only measurement requests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2 import canonical_search_v3 as canonical  # noqa: E402
from framework.stage5 import production_search_v1 as production  # noqa: E402
from framework.stage5 import genome_contract_v1 as genome  # noqa: E402
from framework.stage5 import single_target_search_v2 as single  # noqa: E402


MIGRATION_LINEAGE_MODULE_SHA256 = {
    "canonical_search_v3.py": (
        "bab5ddf6f2427d168c4785050f1963c9d99991bdf64f53ce531ced0a539ac183"
    ),
    "genome_contract_v1.py": (
        "1aa183c576510cf4612d3a96028a56e37b8fd4281058867b7d3b4a08660fbe7b"
    ),
    "production_search_v1.py": (
        "6d373c312e236222d7630d91deb8cb122d5c928b464e5d1b60f7070ea33db2e2"
    ),
    "single_target_search_v2.py": (
        "7d3694a82cee4e9f9265471fd32371d1012279a71a6c150bf8443240b870df35"
    ),
}
EXPECTED_MODULE_SHA256 = {
    "canonical_search_v3.py": (
        "12b384d5f455d6d978a2e49280be7c34eeef3ca5771848bf282d40b300494718"
    ),
    "genome_contract_v1.py": (
        "3b2666576822a42d3bd7079e9b65070ab1d2b476089ee78254961b875c863b46"
    ),
    "production_search_v1.py": (
        "b4d128bfac5a3aa2cf19dd99816a38e58c0b2dc263c9c93a76d08ac8ab30bdb1"
    ),
    "single_target_search_v2.py": (
        "dc8a9d52b433d3cd1ca9e834176070502cfd95de649569ef5ea1c5d5e75ef767"
    ),
}
SOURCE_MODULES = {
    "canonical_search_v3.py": canonical,
    "genome_contract_v1.py": genome,
    "production_search_v1.py": production,
    "single_target_search_v2.py": single,
}
INPUT_NAMES = (
    "measurements",
    "graph_features",
    "capability_profiles",
    "stage4_closure",
    "candidate_registry",
    "search_task",
)
OUTPUT_NAMES = ("selected_ids.json", "measurement_request.json", "manifest.json")
FORBIDDEN_CANDIDATE_TOKENS = (
    "latency",
    "energy",
    "ap30",
    "ap50",
    "ap70",
    "cache",
    "terminal_status",
    "measurement_status",
    "performance_status",
)


class PublicInputError(ValueError):
    """A validation error whose message is safe for the public CLI."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUT_NAMES:
        option = "--" + name.replace("_", "-")
        parser.add_argument(option, required=True, help=f"Explicit {name} JSON/JSONL input.")
        parser.add_argument(
            f"{option}-provenance",
            required=True,
            help=f"Non-path public provenance label for {name}.",
        )
    parser.add_argument("--seed", required=True, help="Frozen deterministic fit seed.")
    parser.add_argument("--output-root", required=True, help="Explicit caller-owned output root.")
    return parser.parse_args(argv)


def _safe_path(value: str, *, label: str, require_file: bool) -> Path:
    requested = Path(value).expanduser()
    lexical = requested.absolute()
    resolved = requested.resolve(strict=False)
    if lexical != resolved:
        raise PublicInputError(f"{label} must not traverse a symlink or parent escape")
    if resolved in {Path("/"), ROOT.resolve()}:
        raise PublicInputError(f"{label} must not be the filesystem or repository root")
    if require_file:
        if not resolved.is_file() or resolved.is_symlink():
            raise PublicInputError(f"{label} must be an existing regular file")
    elif resolved.exists() and (not resolved.is_dir() or resolved.is_symlink()):
        raise PublicInputError(f"{label} must be a directory")
    return resolved


def _safe_output_root(value: str) -> Path:
    root = _safe_path(value, label="output-root", require_file=False)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _provenance(value: str, *, label: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or "/" in normalized
        or "\\" in normalized
        or normalized in {".", ".."}
    ):
        raise PublicInputError(f"{label} must be a non-path provenance label")
    return normalized


def _seed(value: str) -> int:
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise PublicInputError("seed must be a non-negative 32-bit integer") from exc
    if seed < 0 or seed > 2**32 - 1:
        raise PublicInputError("seed must be a non-negative 32-bit integer")
    return seed


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicInputError(f"{label} must contain valid JSON") from exc


def _load_records(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PublicInputError(f"unable to read {label}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        values = payload
    elif payload is not None:
        raise PublicInputError(f"{label} must be a JSON array or JSONL objects")
    else:
        values = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise PublicInputError(
                    f"{label} line {line_number} must be a JSON object"
                )
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PublicInputError(
                    f"{label} line {line_number} must be valid JSON"
                ) from exc
    if not values or any(not isinstance(value, dict) for value in values):
        raise PublicInputError(f"{label} must contain JSON objects")
    return [dict(value) for value in values]


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    value = _load_json(path, label=label)
    if not isinstance(value, dict):
        raise PublicInputError(f"{label} must be a JSON object")
    return dict(value)


def _load_profiles(path: Path) -> list[dict[str, Any]]:
    return [
        canonical.validate_capability_profile(item)
        for item in _load_records(path, label="capability-profiles")
    ]


def _contains_forbidden_candidate_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered != "source_status" and any(
                token in lowered for token in FORBIDDEN_CANDIDATE_TOKENS
            ):
                return True
            if _contains_forbidden_candidate_field(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_candidate_field(item) for item in value)
    return False


def _selection_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    if _contains_forbidden_candidate_field(registry):
        raise PublicInputError("forbidden candidate field before selection")
    groups = registry.get("groups")
    if not isinstance(groups, list) or any(not isinstance(group, Mapping) for group in groups):
        raise PublicInputError("candidate-registry groups must be an array of objects")
    # Source identity is not a performance/cache/terminal label. Preserve it
    # byte-for-byte so the frozen measurement-request hash binds the caller's
    # explicit source contract and evidence digest.
    return copy.deepcopy(dict(registry))


def _build_task(
    payload: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
) -> tuple[single.SearchTask, int]:
    required = {
        "task_id",
        "target_model",
        "hardware_id",
        "capability_profile_id",
        "sample_budget",
        "batch_size",
        "round_count",
        "round_index",
    }
    if required - set(payload):
        raise PublicInputError("search-task is missing required contract fields")
    profile_by_id = {
        str(profile["capability_profile_id"]): profile for profile in profiles
    }
    profile = profile_by_id.get(str(payload["capability_profile_id"]))
    if profile is None:
        raise PublicInputError("search-task capability profile is not declared")
    try:
        task = single.SearchTask(
            task_id=str(payload["task_id"]),
            target_model=str(payload["target_model"]),
            hardware_id=str(payload["hardware_id"]),
            capability_profile=profile,
            sample_budget=int(payload["sample_budget"]),
            batch_size=int(payload["batch_size"]),
            round_count=int(payload["round_count"]),
        )
        round_index = int(payload["round_index"])
    except (TypeError, ValueError) as exc:
        raise PublicInputError("search-task numeric fields must be integers") from exc
    single.validate_search_task(task)
    return task, round_index


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _contract_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(encoded)


def _render_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _dependency_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")
    return {name: importlib.metadata.version(name) for name in names}


def _module_sha256(module: Any) -> str:
    return _sha256(Path(module.__file__).read_bytes())


def _verify_source_identity() -> dict[str, str]:
    current = {
        filename: _module_sha256(module)
        for filename, module in SOURCE_MODULES.items()
    }
    if current != EXPECTED_MODULE_SHA256:
        raise PublicInputError("source identity verification failed")
    return current


def _build_production_request(
    *,
    task: single.SearchTask,
    task_contract: Mapping[str, Any],
    selection: Mapping[str, Any],
    round_index: int,
) -> dict[str, Any]:
    if round_index < 0 or round_index >= task.round_count:
        raise ValueError("round_index outside frozen task budget")
    identity_fields = (
        "schema_version",
        "row_id",
        "manifest_job_id",
        "group_id",
        "model",
        "width",
        "genome",
        "strategy_id",
        "q_mode",
        "capability_profile_id",
        "capability_digest",
        "dispatch_key",
        "source_status",
        "source_contract",
        "source_contract_sha256",
        "source_evidence_sha256",
        "graph_features",
    )
    selection_policy = "predicted_frontier_diversity"
    rows = []
    for selected_row in selection["selected_rows"]:
        candidate_identity = {
            key: copy.deepcopy(selected_row[key])
            for key in identity_fields
            if key in selected_row
        }
        rows.append(
            {
                **candidate_identity,
                "task_id": task.task_id,
                "task_sha256": task_contract["task_sha256"],
                "hardware_id": task.hardware_id,
                "round_index": round_index,
                "selection_policy": selection_policy,
            }
        )
    row_sha = {str(row["row_id"]): _contract_sha256(row) for row in rows}
    payload = {
        "schema_version": "stage5_production_selection_request_v1",
        "policy": selection_policy,
        "selection_interface": (
            "framework.stage5.production_search_v1."
            "select_predicted_frontier_diversity"
        ),
        "task_id": task.task_id,
        "task_sha256": task_contract["task_sha256"],
        "hardware_id": task.hardware_id,
        "hardware_execution": "external",
        "execution_performed": False,
        "round_index": round_index,
        "selected_group_ids": list(selection["selected_group_ids"]),
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "row_sha256": row_sha,
        "rows": rows,
    }
    return {**payload, "measurement_request_sha256": _contract_sha256(payload)}


def _manifest(
    *,
    seed: int,
    inputs: Mapping[str, Mapping[str, str]],
    bundle: production.ProductionBundle,
    task_contract: Mapping[str, Any],
    request: Mapping[str, Any],
    selected_bytes: bytes,
    request_bytes: bytes,
    current_module_sha256: Mapping[str, str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "stage5_selection_manifest_v1",
        "inputs": inputs,
        "seed": seed,
        "dependency_versions": _dependency_versions(),
        "expected_module_sha256": dict(EXPECTED_MODULE_SHA256),
        "current_module_sha256": dict(current_module_sha256),
        "migration_lineage_module_sha256": dict(MIGRATION_LINEAGE_MODULE_SHA256),
        "selection_interface": (
            "framework.stage5.production_search_v1."
            "select_predicted_frontier_diversity"
        ),
        "selection_policy": "predicted_frontier_diversity",
        "bundle_config_sha256": bundle.manifest["bundle_config_sha256"],
        "task_sha256": task_contract["task_sha256"],
        "measurement_request_sha256": request["measurement_request_sha256"],
        "outputs": {
            "selected_ids.json": _sha256(selected_bytes),
            "measurement_request.json": _sha256(request_bytes),
        },
    }
    payload["outputs"]["manifest.json"] = {
        "sha256": _sha256(_render_json(payload)),
        "sha256_scope": "manifest_body_before_self_digest",
    }
    return payload


def _preflight_artifacts(output_root: Path, artifacts: Mapping[str, bytes]) -> None:
    for name, data in artifacts.items():
        path = output_root / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise PublicInputError(f"refusing to overwrite non-file output: {name}")
        if path.exists() and path.read_bytes() != data:
            raise PublicInputError(f"refusing to overwrite different output: {name}")


def _write_artifacts(output_root: Path, artifacts: Mapping[str, bytes]) -> None:
    _preflight_artifacts(output_root, artifacts)
    for name, data in artifacts.items():
        path = output_root / name
        if not path.exists():
            path.write_bytes(data)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        current_module_sha256 = _verify_source_identity()
        seed = _seed(args.seed)
        paths = {
            name: _safe_path(
                getattr(args, name),
                label=name.replace("_", "-"),
                require_file=True,
            )
            for name in INPUT_NAMES
        }
        output_root = _safe_output_root(args.output_root)
        inputs = {
            name: {
                "sha256": _sha256(paths[name].read_bytes()),
                "provenance": _provenance(
                    getattr(args, f"{name}_provenance"),
                    label=f"{name.replace('_', '-')} provenance",
                ),
            }
            for name in INPUT_NAMES
        }
        measurements = _load_records(paths["measurements"], label="measurements")
        graph_features = _load_records(
            paths["graph_features"], label="graph-features"
        )
        profiles = _load_profiles(paths["capability_profiles"])
        closure = _load_object(paths["stage4_closure"], label="stage4-closure")
        registry = _selection_registry(
            _load_object(paths["candidate_registry"], label="candidate-registry")
        )
        task_payload = _load_object(paths["search_task"], label="search-task")
        task, round_index = _build_task(task_payload, profiles)
        task_contract = single.validate_search_task(task)

        bundle = production.fit_production_bundle(
            measurements,
            graph_features,
            profiles,
            closure,
            seed=seed,
        )
        frozen_holdout = closure.get("frozen_holdout")
        if not isinstance(frozen_holdout, Mapping):
            raise PublicInputError("stage4-closure must declare frozen_holdout")
        candidate_manifest = production.build_candidate_manifest(
            registry,
            measured_group_ids={str(row["group_id"]) for row in measurements},
            frozen_holdout=frozen_holdout,
            capability_profiles=profiles,
        )
        predicted = production.predict_candidate_rows(
            bundle,
            candidate_manifest["rows"],
            profiles,
        )
        selection = production.select_predicted_frontier_diversity(
            predicted,
            measured_rows=measurements,
            measured_graph_features=graph_features,
            group_budget_by_model={task.target_model: task.batch_size // 4},
        )
        request = _build_production_request(
            task=task,
            task_contract=task_contract,
            selection=selection,
            round_index=round_index,
        )
        selected_payload = {
            "schema_version": "stage5_ordered_selected_ids_v1",
            "task_sha256": task_contract["task_sha256"],
            "ordered_selected_ids": selection["selected_group_ids"],
            "ordered_selected_row_ids": [
                str(row["row_id"]) for row in selection["selected_rows"]
            ],
        }
        selected_bytes = _render_json(selected_payload)
        request_bytes = _render_json(request)
        manifest_bytes = _render_json(
            _manifest(
                seed=seed,
                inputs=inputs,
                bundle=bundle,
                task_contract=task_contract,
                request=request,
                selected_bytes=selected_bytes,
                request_bytes=request_bytes,
                current_module_sha256=current_module_sha256,
            )
        )
        _write_artifacts(
            output_root,
            dict(zip(OUTPUT_NAMES, (selected_bytes, request_bytes, manifest_bytes))),
        )
    except PublicInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (KeyError, TypeError, ValueError):
        print("error: selection contract validation failed", file=sys.stderr)
        return 2
    print(
        "stage5_selection_ok "
        f"task={task_contract['task_sha256']} "
        f"selected={len(selection['selected_group_ids'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
