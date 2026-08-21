"""Current-run provenance evidence for P6 recipe-v2 source reuse."""
from __future__ import annotations
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal
from framework.stage6.p6_history_recipe_profiles_v1 import SHARED_SOURCE_PATH_KEYS
RUN_METADATA_RELATIVE_ROOT: Final = Path(".p6-materializer-training-bridge-v1")
RUN_CONTEXT_RELATIVE_PATH: Final = RUN_METADATA_RELATIVE_ROOT / "run-context.json"
GROUP_RECEIPT_RELATIVE_ROOT: Final = RUN_METADATA_RELATIVE_ROOT / "group-receipts"
SOURCE_OUTPUT_RELATIVE_ROOT: Final = Path("materialized")
PLAN_RELATIVE_PATH: Final = Path("pyramid_candidate_plan.json")
REGISTRY_RELATIVE_PATH: Final = Path("source_registry.json")
RUN_CONTEXT_SCHEMA_VERSION: Final = "p6_materializer_fresh_run_context_v1"
PLAN_SCHEMA_VERSION: Final = "p6_pyramid_candidate_plan_v2"
REGISTRY_SCHEMA_VERSION: Final = "stage5_candidate_source_registry_v2"
SOURCE_ARTIFACT_KEYS: Final = (
    "checkpoint_path", "checkpoint_dir", "config_path", "onnx_path",
    "onnx_report_path", "calibration_root", "calibration_npz",
    "calibration_summary", "trt_calibration_dir",
)
SOURCE_MARKER_KEYS: Final = ("training_done_marker", "source_done_marker")
_DIRECTORY_ARTIFACT_KEYS: Final = frozenset({"checkpoint_dir", "calibration_root", "trt_calibration_dir"})
_HEX_CHARS: Final = frozenset("0123456789abcdef")
PrivateCategory = Literal["p6_source_reuse_partial", "p6_source_reuse_stale", "p6_source_reuse_mismatch"]
PublicCategory = Literal["history_execution_invalid", "unsafe_destination"]
class P6SourceReuseEvidenceError(ValueError):
    private_category: PrivateCategory | None
    public_category: PublicCategory
    def __init__(self, *, public_category: PublicCategory, private_category: PrivateCategory | None = None) -> None:
        if (type(public_category) is not str or public_category not in ("history_execution_invalid", "unsafe_destination")
            or private_category is not None and (type(private_category) is not str
                or private_category not in ("p6_source_reuse_partial", "p6_source_reuse_stale", "p6_source_reuse_mismatch"))):
            raise TypeError("invalid_error_category")
        super().__init__(public_category)
        self.private_category, self.public_category = private_category, public_category
@dataclass(frozen=True)
class P6SourceReusePaths:
    local_output_root: Path
    metadata_root: Path
    run_context: Path
    receipt_root: Path
@dataclass(frozen=True)
class P6FreshRunContext:
    schema_version: str
    run_nonce: str
    task_id: str
    task_sha256: str
    code_revision: str
    local_output_root_sha256: str
    candidate_plan_schema_version: str
    candidate_plan_sha256: str
    source_registry_schema_version: str
    source_registry_sha256: str
    created_before_round_index: int
    run_context_sha256: str
@dataclass(frozen=True)
class P6ArtifactDigest:
    kind: Literal["regular_file", "directory_tree"]
    sha256: str
@dataclass(frozen=True)
class P6GroupSourceReceipt:
    schema_version: str
    status: str
    run_context_sha256: str
    run_nonce: str
    task_id: str
    task_sha256: str
    group_id: str
    group_key_sha256: str
    source_contract_sha256: str
    source_evidence_sha256: str
    producer_round_index: int
    producer_measurement_request_sha256: str
    producer_row_id: str
    producer_row_sha256: str
    producer_q_mode: str
    artifact_digests: tuple[tuple[str, P6ArtifactDigest], ...]
    marker_digests: tuple[tuple[str, str], ...]
    receipt_sha256: str
@dataclass(frozen=True)
class P6GroupReuseDecision:
    group_id: str
    state: str
    receipt_path: Path
def canonical_json_sha256(value: Any) -> str:
    try:
        encoded = _canonical_json_bytes(value)
    except (TypeError, ValueError):
        _fail()
    return hashlib.sha256(encoded).hexdigest()
def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
def _fail(private_category: PrivateCategory | None = "p6_source_reuse_mismatch", *, public_category: PublicCategory = "history_execution_invalid") -> None:
    raise P6SourceReuseEvidenceError(
        public_category=public_category, private_category=private_category) from None
def _is_lower_hex(value: Any, length: int = 64) -> bool:
    return (isinstance(value, str) and len(value) == length
            and set(value).issubset(_HEX_CHARS))
def _is_nonempty_string(value: Any) -> bool:
    return type(value) is str and bool(value)
def _has_exact_hashes(raw: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return all(_is_lower_hex(raw.get(key)) for key in keys)
def _validate_context_shape(raw: Mapping[str, Any]) -> None:
    if (set(raw) != set(P6FreshRunContext.__dataclass_fields__)
        or raw.get("schema_version") != RUN_CONTEXT_SCHEMA_VERSION
        or not _has_exact_hashes(raw, ("run_nonce", "task_sha256",
            "local_output_root_sha256", "candidate_plan_sha256", "source_registry_sha256",
            "run_context_sha256"))
        or not _is_nonempty_string(raw.get("task_id"))
        or not _is_nonempty_string(raw.get("code_revision"))
        or raw.get("candidate_plan_schema_version") != PLAN_SCHEMA_VERSION
        or raw.get("source_registry_schema_version") != REGISTRY_SCHEMA_VERSION
        or type(raw.get("created_before_round_index")) is not int
        or raw.get("created_before_round_index") != 0):
        _fail()
def _validate_receipt_shape(raw: Mapping[str, Any]) -> None:
    if (set(raw) != set(P6GroupSourceReceipt.__dataclass_fields__)
        or raw.get("schema_version") != "p6_group_source_reuse_receipt_v1"
        or raw.get("status") != "READY_CURRENT_RUN"
        or not _has_exact_hashes(raw, (
            "run_context_sha256", "run_nonce", "task_sha256", "group_key_sha256",
            "source_contract_sha256", "source_evidence_sha256", "producer_row_sha256",
            "producer_measurement_request_sha256", "receipt_sha256"))
        or not all(_is_nonempty_string(raw.get(key)) for key in (
            "task_id", "group_id", "producer_row_id", "producer_q_mode"))
        or raw.get("producer_q_mode") not in {"fp16", "int8"}
        or type(raw.get("producer_round_index")) is not int
        or raw.get("producer_round_index") not in range(4)):
        _fail()
def _validate_real_directory(path: Path) -> Path:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        _fail(None, public_category="unsafe_destination")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        _fail(None, public_category="unsafe_destination")
    if resolved != path:
        _fail(None, public_category="unsafe_destination")
    current = path
    while True:
        try:
            mode = current.lstat().st_mode
        except OSError:
            _fail(None, public_category="unsafe_destination")
        if stat.S_ISLNK(mode):
            _fail(None, public_category="unsafe_destination")
        if current == path and not stat.S_ISDIR(mode):
            _fail(None, public_category="unsafe_destination")
        if current == current.parent:
            break
        current = current.parent
    return resolved
def _ensure_absent(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        _fail()
    _fail()
def _validate_existing_directory(path: Path, *, missing_category: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError:
        _fail(missing_category)
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        _fail()
def _validate_regular_single_link(path: Path, *, missing_category: str) -> None:
    try:
        info = path.lstat()
    except OSError:
        _fail(missing_category)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
        _fail()
def plan_source_reuse_paths(local_output_root: Path) -> P6SourceReusePaths:
    if not isinstance(local_output_root, Path):
        _fail()
    root = _validate_real_directory(local_output_root)
    paths = P6SourceReusePaths(
        local_output_root=root,
        metadata_root=root / RUN_METADATA_RELATIVE_ROOT,
        run_context=root / RUN_CONTEXT_RELATIVE_PATH,
        receipt_root=root / GROUP_RECEIPT_RELATIVE_ROOT,
    )
    for child in (
        paths.metadata_root,
        paths.run_context,
        paths.receipt_root,
        root / SOURCE_OUTPUT_RELATIVE_ROOT,
    ):
        try:
            if stat.S_ISLNK(child.lstat().st_mode):
                _fail()
        except FileNotFoundError:
            pass
        except OSError:
            _fail()
    return paths
def resolve_existing_source_reuse_paths(local_output_root: Path) -> P6SourceReusePaths:
    paths = plan_source_reuse_paths(local_output_root)
    _validate_existing_directory(
        paths.metadata_root, missing_category="p6_source_reuse_partial"
    )
    _validate_existing_directory(
        paths.receipt_root, missing_category="p6_source_reuse_partial"
    )
    _validate_regular_single_link(
        paths.run_context, missing_category="p6_source_reuse_partial"
    )
    return paths
def receipt_path_for_group(paths: P6SourceReusePaths, group_id: str) -> Path:
    if not isinstance(paths, P6SourceReusePaths) or not isinstance(group_id, str) or not group_id:
        _fail()
    name = hashlib.sha256(
        b"p6-group-receipt-v1\0" + group_id.encode("utf-8")
    ).hexdigest()
    return paths.receipt_root / f"{name}.json"
def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail()
        result[key] = value
    return result
def _read_strict_json(path: Path, *, missing_category: str) -> dict[str, Any]:
    _validate_regular_single_link(path, missing_category=missing_category)
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail()
    if not isinstance(raw, dict):
        _fail()
    return raw
def _validate_context_inputs(*, paths: P6SourceReusePaths,
    task_contract: Mapping[str, Any], candidate_plan: Mapping[str, Any],
    source_registry: Mapping[str, Any], code_revision: str) -> tuple[str, str]:
    if not all(
        isinstance(value, Mapping)
        for value in (task_contract, candidate_plan, source_registry)
    ):
        _fail()
    task_id = task_contract.get("task_id")
    task_sha256 = task_contract.get("task_sha256")
    if not isinstance(task_id, str) or not task_id or not _is_lower_hex(task_sha256):
        _fail()
    if (candidate_plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or source_registry.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or not isinstance(code_revision, str) or not code_revision):
        _fail()
    persisted_plan = _read_strict_json(paths.local_output_root / PLAN_RELATIVE_PATH,
                                       missing_category="p6_source_reuse_partial")
    persisted_registry = _read_strict_json(paths.local_output_root / REGISTRY_RELATIVE_PATH,
                                           missing_category="p6_source_reuse_partial")
    if persisted_plan != dict(candidate_plan) or persisted_registry != dict(source_registry):
        _fail()
    _validate_registry_paths(paths, source_registry)
    return task_id, task_sha256
def _validated_declared_path(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        _fail()
    path = Path(raw)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        _fail()
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail()
    if not relative.parts or relative.parts[0] != SOURCE_OUTPUT_RELATIVE_ROOT.name:
        _fail()
    return path
def _validate_registry_paths(
    paths: P6SourceReusePaths, source_registry: Mapping[str, Any]
) -> None:
    groups = source_registry.get("groups")
    if not isinstance(groups, list):
        _fail()
    seen_groups: set[str] = set()
    seen_paths: set[Path] = set()
    for group in groups:
        if not isinstance(group, Mapping):
            _fail()
        group_id = group.get("group_id")
        contract = group.get("source_contract")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in seen_groups
            or not isinstance(contract, Mapping)
            or not set(SHARED_SOURCE_PATH_KEYS).issubset(contract)
        ):
            _fail()
        seen_groups.add(group_id)
        for key in SHARED_SOURCE_PATH_KEYS:
            declared = _validated_declared_path(
                paths.local_output_root, contract.get(key)
            )
            if declared in seen_paths:
                _fail()
            seen_paths.add(declared)
def _root_sha256(root: Path) -> str:
    return hashlib.sha256(b"p6-local-output-root-v1\0" + str(root).encode()).hexdigest()
def _context_without_hash(*, run_nonce: str, task_id: str, task_sha256: str,
    code_revision: str, root_sha256: str, plan_sha256: str,
    registry_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": RUN_CONTEXT_SCHEMA_VERSION,
        "run_nonce": run_nonce,
        "task_id": task_id,
        "task_sha256": task_sha256,
        "code_revision": code_revision,
        "local_output_root_sha256": root_sha256,
        "candidate_plan_schema_version": PLAN_SCHEMA_VERSION,
        "candidate_plan_sha256": plan_sha256,
        "source_registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "source_registry_sha256": registry_sha256,
        "created_before_round_index": 0,
    }
def _exclusive_publish_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.publish-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    published = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                _fail()
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        published = True
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        _fail()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if not published:
            _ensure_absent(path)
def create_fresh_run_context(*, local_output_root: Path,
    task_contract: Mapping[str, Any], candidate_plan: Mapping[str, Any],
    source_registry: Mapping[str, Any], code_revision: str) -> P6FreshRunContext:
    paths = plan_source_reuse_paths(local_output_root)
    task_id, task_sha256 = _validate_context_inputs(
        paths=paths,
        task_contract=task_contract,
        candidate_plan=candidate_plan,
        source_registry=source_registry,
        code_revision=code_revision,
    )
    for path in (
        paths.metadata_root,
        paths.receipt_root,
        paths.local_output_root / SOURCE_OUTPUT_RELATIVE_ROOT,
    ):
        _ensure_absent(path)
    registry_groups = source_registry["groups"]
    for group in registry_groups:
        contract = group["source_contract"]
        _ensure_absent(receipt_path_for_group(paths, group["group_id"]))
        for key in SHARED_SOURCE_PATH_KEYS:
            _ensure_absent(Path(contract[key]))
    run_nonce = secrets.token_bytes(32).hex()
    without_hash = _context_without_hash(
        run_nonce=run_nonce,
        task_id=task_id,
        task_sha256=task_sha256,
        code_revision=code_revision,
        root_sha256=_root_sha256(paths.local_output_root),
        plan_sha256=canonical_json_sha256(candidate_plan),
        registry_sha256=canonical_json_sha256(source_registry),
    )
    serialized = {**without_hash, "run_context_sha256": canonical_json_sha256(without_hash)}
    context = P6FreshRunContext(**serialized)
    try:
        paths.metadata_root.mkdir(mode=0o700)
        paths.receipt_root.mkdir(mode=0o700)
    except OSError:
        _fail()
    _exclusive_publish_json(paths.run_context, serialized)
    return context
def load_fresh_run_context(*, local_output_root: Path, expected_task_id: str,
                           expected_task_sha256: str) -> P6FreshRunContext:
    if (not isinstance(expected_task_id, str) or not expected_task_id
        or not _is_lower_hex(expected_task_sha256)):
        _fail()
    paths = resolve_existing_source_reuse_paths(local_output_root)
    candidate_plan = _read_strict_json(paths.local_output_root / PLAN_RELATIVE_PATH,
                                       missing_category="p6_source_reuse_partial")
    source_registry = _read_strict_json(paths.local_output_root / REGISTRY_RELATIVE_PATH,
                                        missing_category="p6_source_reuse_partial")
    if (candidate_plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or source_registry.get("schema_version") != REGISTRY_SCHEMA_VERSION):
        _fail()
    _validate_registry_paths(paths, source_registry)
    raw = _read_strict_json(
        paths.run_context, missing_category="p6_source_reuse_partial"
    )
    _validate_context_shape(raw)
    without_hash = {key: value for key, value in raw.items() if key != "run_context_sha256"}
    if not _is_lower_hex(raw.get("run_context_sha256")) or raw[
        "run_context_sha256"
    ] != canonical_json_sha256(without_hash):
        _fail()
    expected = _context_without_hash(
        run_nonce=raw.get("run_nonce"),
        task_id=expected_task_id,
        task_sha256=expected_task_sha256,
        code_revision=raw.get("code_revision"),
        root_sha256=_root_sha256(paths.local_output_root),
        plan_sha256=canonical_json_sha256(candidate_plan),
        registry_sha256=canonical_json_sha256(source_registry),
    )
    if not _is_lower_hex(raw.get("run_nonce")) or without_hash != expected:
        _fail("p6_source_reuse_stale")
    return P6FreshRunContext(**raw)
def requires_current_run_source_evidence(request: Mapping[str, Any]) -> bool:
    rows = request.get("rows") if isinstance(request, Mapping) else None
    if not isinstance(rows, list) or not rows:
        _fail()
    signals = [
        isinstance(row, Mapping)
        and isinstance(row.get("source_contract"), Mapping)
        and set(SHARED_SOURCE_PATH_KEYS).issubset(row["source_contract"])
        for row in rows
    ]
    if any(signals) and not all(signals):
        _fail()
    return all(signals)
def _group_key(group_id: str) -> str:
    return hashlib.sha256(
        b"p6-group-receipt-v1\0" + group_id.encode("utf-8")
    ).hexdigest()
def _selected_groups(
    request: Mapping[str, Any], root: Path
) -> tuple[tuple[str, Mapping[str, Any], Mapping[str, Any]], ...]:
    rows = request.get("rows") if isinstance(request, Mapping) else None
    round_index = request.get("round_index") if isinstance(request, Mapping) else None
    if (not isinstance(rows, list) or not rows or isinstance(round_index, bool)
        or not isinstance(round_index, int) or round_index not in range(4)):
        _fail()
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            _fail()
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            _fail()
        grouped.setdefault(group_id, []).append(row)
    result = []
    for group_id in sorted(grouped):
        group_rows = sorted(grouped[group_id], key=lambda item: str(item.get("row_id")))
        first = group_rows[0]
        contract = first.get("source_contract")
        contract_sha = first.get("source_contract_sha256")
        evidence_sha = first.get("source_evidence_sha256")
        if (
            not isinstance(contract, Mapping)
            or not _is_lower_hex(contract_sha)
            or canonical_json_sha256(contract) != contract_sha
            or not _is_lower_hex(evidence_sha)
            or contract.get("source_evidence_sha256") != evidence_sha
        ):
            _fail()
        for row in group_rows:
            if (
                row.get("source_contract") != contract
                or row.get("source_contract_sha256") != contract_sha
                or row.get("source_evidence_sha256") != evidence_sha
            ):
                _fail()
        _contract_paths(contract, root)
        result.append((group_id, first, contract))
    return tuple(result)
def _contract_paths(contract: Mapping[str, Any], root: Path) -> dict[str, Path]:
    if not set(SHARED_SOURCE_PATH_KEYS).issubset(contract):
        _fail()
    result = {
        key: _validated_declared_path(root, contract.get(key))
        for key in SHARED_SOURCE_PATH_KEYS
    }
    if len(set(result.values())) != len(result):
        _fail()
    for path in result.values():
        current = root
        for component in path.relative_to(root).parts[:-1]:
            current /= component
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError:
                break
            except OSError:
                _fail()
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                _fail()
    return result
def _leaf_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        _fail()
    return True
def _sha_file(path: Path) -> str:
    _validate_regular_single_link(path, missing_category="p6_source_reuse_partial")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        _fail()
    return digest.hexdigest()
def _directory_digest(root: Path) -> P6ArtifactDigest:
    _validate_existing_directory(root, missing_category="p6_source_reuse_partial")
    entries: list[dict[str, Any]] = []
    def visit(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError:
            _fail()
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                _fail()
            if stat.S_ISDIR(info.st_mode):
                entries.append({"kind": "directory", "path": relative})
                visit(path)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                entries.append(
                    {
                        "kind": "regular_file", "path": relative,
                        "sha256": _sha_file(path), "size": info.st_size,
                    }
                )
            else:
                _fail()
    visit(root)
    return P6ArtifactDigest("directory_tree", canonical_json_sha256(entries))
def _bundle_digests(
    paths: Mapping[str, Path],
) -> tuple[tuple[tuple[str, P6ArtifactDigest], ...], tuple[tuple[str, str], ...]]:
    artifacts = tuple(
        (
            key,
            _directory_digest(paths[key])
            if key in _DIRECTORY_ARTIFACT_KEYS
            else P6ArtifactDigest("regular_file", _sha_file(paths[key])),
        )
        for key in SOURCE_ARTIFACT_KEYS
    )
    markers = tuple((key, _sha_file(paths[key])) for key in SOURCE_MARKER_KEYS)
    return artifacts, markers
def _receipt_to_json(receipt: P6GroupSourceReceipt) -> dict[str, Any]:
    raw = asdict(receipt)
    raw["artifact_digests"] = {
        key: asdict(digest) for key, digest in receipt.artifact_digests
    }
    raw["marker_digests"] = dict(receipt.marker_digests)
    return raw
def _parse_receipt(path: Path) -> P6GroupSourceReceipt:
    raw = _read_strict_json(path, missing_category="p6_source_reuse_partial")
    _validate_receipt_shape(raw)
    stored_hash = raw.get("receipt_sha256")
    without_hash = {key: value for key, value in raw.items() if key != "receipt_sha256"}
    artifacts = raw.get("artifact_digests")
    markers = raw.get("marker_digests")
    if (
        not _is_lower_hex(stored_hash)
        or stored_hash != canonical_json_sha256(without_hash)
        or not isinstance(artifacts, dict)
        or set(artifacts) != set(SOURCE_ARTIFACT_KEYS)
        or not isinstance(markers, dict)
        or set(markers) != set(SOURCE_MARKER_KEYS)
    ):
        _fail()
    parsed_artifacts = []
    for key in SOURCE_ARTIFACT_KEYS:
        item = artifacts[key]
        expected_kind = "directory_tree" if key in _DIRECTORY_ARTIFACT_KEYS else "regular_file"
        if (
            not isinstance(item, dict)
            or set(item) != {"kind", "sha256"}
            or item.get("kind") != expected_kind
            or not _is_lower_hex(item.get("sha256"))
        ):
            _fail()
        parsed_artifacts.append((key, P6ArtifactDigest(**item)))
    if any(not _is_lower_hex(markers.get(key)) for key in SOURCE_MARKER_KEYS):
        _fail()
    converted = dict(raw)
    converted["artifact_digests"] = tuple(parsed_artifacts)
    converted["marker_digests"] = tuple((key, markers[key]) for key in SOURCE_MARKER_KEYS)
    return P6GroupSourceReceipt(**converted)
def _producer_request(receipt: P6GroupSourceReceipt, *, consumer_round: int, interface: Mapping[str, Any], private_root: Path) -> tuple[dict[str, Any], Path]:
    producer_round = receipt.producer_round_index
    layout = interface.get("output_layout")
    template = layout.get("round_root_template") if isinstance(layout, Mapping) else None
    if (isinstance(producer_round, bool) or not isinstance(producer_round, int)
        or producer_round not in range(4) or producer_round > consumer_round
        or not isinstance(template, str) or template.count("{round_id}") != 1):
        _fail()
    relative = Path(template.replace("{round_id}", str(producer_round)))
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        _fail()
    request_path = private_root / relative / "measurement-request.json"
    try:
        current = private_root
        for component in request_path.relative_to(private_root).parts[:-1]:
            current /= component
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                _fail()
    except (OSError, RuntimeError, ValueError):
        _fail()
    request = _read_strict_json(request_path, missing_category="p6_source_reuse_mismatch")
    body = {key: value for key, value in request.items() if key != "measurement_request_sha256"}
    rows, row_hashes = request.get("rows"), request.get("row_sha256")
    if (request.get("schema_version") != "stage5_measurement_request_v2"
        or canonical_json_sha256(body) != request.get("measurement_request_sha256")
        or request.get("measurement_request_sha256") != receipt.producer_measurement_request_sha256
        or type(request.get("round_index")) is not int or request.get("round_index") != producer_round
        or request.get("task_id") != receipt.task_id or request.get("task_sha256") != receipt.task_sha256
        or not isinstance(rows, list) or not rows or not isinstance(row_hashes, Mapping)
        or any(not isinstance(row, Mapping) for row in rows)):
        _fail()
    row_ids = []
    for row in rows:
        row_id = row.get("row_id")
        if (not _is_nonempty_string(row_id) or not _is_nonempty_string(row.get("group_id"))
            or row.get("q_mode") not in ("fp16", "int8")
            or not _is_lower_hex(row.get("source_contract_sha256"))
            or not _is_lower_hex(row.get("source_evidence_sha256"))):
            _fail()
        row_ids.append(row_id)
    if (len(set(row_ids)) != len(row_ids) or set(row_hashes) != set(row_ids)
        or any(not _is_lower_hex(row_hashes.get(key)) or row_hashes[key] != canonical_json_sha256(row)
               for key, row in zip(row_ids, rows, strict=True))):
        _fail()
    return request, request_path
def _validate_receipt(receipt: P6GroupSourceReceipt, *, group_id: str, row: Mapping[str, Any],
    contract: Mapping[str, Any], request: Mapping[str, Any], context: P6FreshRunContext,
    paths: Mapping[str, Path], interface: Mapping[str, Any], private_root: Path) -> None:
    if (receipt.run_context_sha256 != context.run_context_sha256
        or receipt.run_nonce != context.run_nonce or receipt.task_id != context.task_id
        or receipt.task_sha256 != context.task_sha256):
        _fail("p6_source_reuse_stale")
    if (receipt.schema_version != "p6_group_source_reuse_receipt_v1"
        or receipt.status != "READY_CURRENT_RUN" or receipt.group_id != group_id
        or receipt.group_key_sha256 != _group_key(group_id)
        or receipt.source_contract_sha256 != row.get("source_contract_sha256")
        or receipt.source_evidence_sha256 != row.get("source_evidence_sha256")):
        _fail()
    consumer_round = request.get("round_index")
    if isinstance(consumer_round, bool) or not isinstance(consumer_round, int):
        _fail()
    producer, _ = _producer_request(receipt, consumer_round=consumer_round, interface=interface, private_root=private_root)
    producer_rows = [item for item in producer["rows"] if item["group_id"] == group_id]
    if not producer_rows:
        _fail()
    producer_row = min(producer_rows, key=lambda item: item.get("row_id", ""))
    row_id = producer_row["row_id"]
    if (receipt.producer_row_id != row_id
        or producer["row_sha256"][row_id] != receipt.producer_row_sha256
        or canonical_json_sha256(producer_row) != receipt.producer_row_sha256
        or producer_row.get("q_mode") != receipt.producer_q_mode
        or producer_row.get("source_contract_sha256") != receipt.source_contract_sha256
        or producer_row.get("source_evidence_sha256") != receipt.source_evidence_sha256):
        _fail()
    artifacts, markers = _bundle_digests(paths)
    if artifacts != receipt.artifact_digests or markers != receipt.marker_digests:
        _fail()
def classify_selected_group_sources(
    request: Mapping[str, Any], *, run_context: P6FreshRunContext,
    local_output_root: Path, interface: Mapping[str, Any], private_root: Path,
) -> tuple[P6GroupReuseDecision, ...]:
    paths = resolve_existing_source_reuse_paths(local_output_root)
    decisions = []
    try:
        groups = _selected_groups(request, paths.local_output_root)
    except P6SourceReuseEvidenceError:
        group_ids = sorted(
            {str(row.get("group_id")) for row in request.get("rows", []) if isinstance(row, Mapping)}
        )
        return tuple(
            P6GroupReuseDecision(group_id, "INVALID_MISMATCH", receipt_path_for_group(paths, group_id))
            for group_id in group_ids
        )
    for group_id, row, contract in groups:
        receipt_path = receipt_path_for_group(paths, group_id)
        try:
            outputs = _contract_paths(contract, paths.local_output_root)
            receipt_exists = _leaf_exists(receipt_path)
            present = tuple(_leaf_exists(outputs[key]) for key in SHARED_SOURCE_PATH_KEYS)
            if not receipt_exists and not any(present):
                state = "UNSEEN"
            elif not receipt_exists or not all(present):
                state = "INVALID_PARTIAL"
            else:
                receipt = _parse_receipt(receipt_path)
                _validate_receipt(
                    receipt, group_id=group_id, row=row, contract=contract,
                    request=request, context=run_context, paths=outputs,
                    interface=interface, private_root=private_root,
                )
                state = "READY_CURRENT_RUN"
        except P6SourceReuseEvidenceError as exc:
            state = {"p6_source_reuse_partial": "INVALID_PARTIAL", "p6_source_reuse_stale": "INVALID_STALE"}.get(
                exc.private_category, "INVALID_MISMATCH")
        decisions.append(P6GroupReuseDecision(group_id, state, receipt_path))
    return tuple(decisions)
def first_use_group_ids(decisions: Sequence[P6GroupReuseDecision]) -> tuple[str, ...]:
    invalid = next((item.state for item in decisions if item.state not in {"UNSEEN", "READY_CURRENT_RUN"}), None)
    if invalid:
        _fail({"INVALID_PARTIAL": "p6_source_reuse_partial", "INVALID_STALE": "p6_source_reuse_stale"}.get(invalid, "p6_source_reuse_mismatch"))
    return tuple(sorted(item.group_id for item in decisions if item.state == "UNSEEN"))
def validate_and_publish_group_receipt(
    request: Mapping[str, Any], *, group_id: str, run_context: P6FreshRunContext,
    local_output_root: Path, interface: Mapping[str, Any], private_root: Path,
) -> P6GroupSourceReceipt:
    paths = resolve_existing_source_reuse_paths(local_output_root)
    groups = {item[0]: item for item in _selected_groups(request, paths.local_output_root)}
    if group_id not in groups:
        _fail()
    _, row, contract = groups[group_id]
    receipt_path = receipt_path_for_group(paths, group_id)
    _ensure_absent(receipt_path)
    outputs = _contract_paths(contract, paths.local_output_root)
    artifacts, markers = _bundle_digests(outputs)
    producer_round = request.get("round_index")
    if isinstance(producer_round, bool) or not isinstance(producer_round, int):
        _fail()
    producer_rows = sorted(
        (item for item in request["rows"] if item.get("group_id") == group_id),
        key=lambda item: item.get("row_id", ""),
    )
    producer_row = producer_rows[0]
    provisional = P6GroupSourceReceipt(
        "p6_group_source_reuse_receipt_v1", "READY_CURRENT_RUN",
        run_context.run_context_sha256, run_context.run_nonce, run_context.task_id,
        run_context.task_sha256, group_id, _group_key(group_id),
        row["source_contract_sha256"], row["source_evidence_sha256"], producer_round,
        request["measurement_request_sha256"], producer_row["row_id"],
        request["row_sha256"][producer_row["row_id"]], producer_row["q_mode"],
        artifacts, markers, "",
    )
    producer_request, request_path = _producer_request(
        provisional, consumer_round=producer_round, interface=interface, private_root=private_root
    )
    del producer_request
    training = outputs["training_done_marker"].stat().st_mtime_ns
    source = outputs["source_done_marker"].stat().st_mtime_ns
    if not request_path.stat().st_mtime_ns <= training <= source:
        _fail()
    without_hash = _receipt_to_json(provisional)
    without_hash.pop("receipt_sha256")
    receipt = replace(
        provisional, receipt_sha256=canonical_json_sha256(without_hash)
    )
    _exclusive_publish_json(receipt_path, _receipt_to_json(receipt))
    return receipt
def require_selected_groups_ready_current_run(
    request: Mapping[str, Any], *, run_context: P6FreshRunContext,
    local_output_root: Path, interface: Mapping[str, Any], private_root: Path,
) -> tuple[P6GroupSourceReceipt, ...]:
    decisions = classify_selected_group_sources(
        request, run_context=run_context, local_output_root=local_output_root,
        interface=interface, private_root=private_root,
    )
    if any(item.state != "READY_CURRENT_RUN" for item in decisions):
        _fail()
    return tuple(_parse_receipt(item.receipt_path) for item in decisions)
