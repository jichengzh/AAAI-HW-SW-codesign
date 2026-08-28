"""Model-native configuration capture for formal Stage1 trace builds."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
import sys
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterator

from framework.stage1.formal_checkpoint_snapshot import FormalCheckpointSnapshot
from framework.stage1.formal_config_snapshot import FormalConfigSnapshot
from framework.stage1.formal_config_tags import validate_model_native_yaml_tags


_IMPORT_LOCK = RLock()
_ACTIVE_ROOT: ContextVar[Path | None] = ContextVar(
    "stage1_formal_model_import_root", default=None
)


class FormalTraceBuildMixin:
    """Shared formal-build method without expanding the adapter registry file."""

    model_config_import_root: Path | None = None

    def build_formal_trace_net(
        self,
        device: str,
        checkpoint_authority: Any,
        config_authority: Any = None,
        loaded_model_config: Any = None,
    ) -> tuple[Any, Any, Any]:
        return build_model_native_formal_trace(
            self, device, checkpoint_authority, config_authority, loaded_model_config
        )


def _formal_build_inputs(
    checkpoint_authority: FormalCheckpointSnapshot | None,
    loaded_model_config: Any | None,
) -> tuple[Path, Any] | None:
    if checkpoint_authority is None and loaded_model_config is None:
        return None
    if checkpoint_authority is None or loaded_model_config is None:
        raise ValueError("formal trace build authorities are incomplete")
    return checkpoint_authority.loader_path, loaded_model_config


def adapter_trace_net(
    adapter: Any,
    device: str,
    checkpoint_authority: FormalCheckpointSnapshot | None,
    config_authority: FormalConfigSnapshot | None = None,
    loaded_model_config: Any | None = None,
) -> tuple[Any, Any, Any | None]:
    """Keep diagnostic builds unchanged and require formal config capture."""

    if checkpoint_authority is None:
        net, inputs = adapter.build_trace_net(device)
        return net, inputs, None
    builder = getattr(adapter, "build_formal_trace_net", None)
    if not callable(builder):
        raise ValueError("formal graph scan requires model-native config capture")
    return builder(
        device,
        checkpoint_authority=checkpoint_authority,
        config_authority=config_authority,
        loaded_model_config=loaded_model_config,
    )


def build_model_native_formal_trace(
    adapter: Any,
    device: str,
    checkpoint_authority: FormalCheckpointSnapshot,
    config_authority: FormalConfigSnapshot | None,
    loaded_model_config: Any | None,
) -> tuple[Any, Any, Any]:
    """Load a native config once, then reuse that object for every model build."""

    config = loaded_model_config
    if config is None:
        config = _load_model_native_config(adapter, config_authority)
    net, inputs = adapter.build_trace_net(
        device,
        checkpoint_authority=checkpoint_authority,
        loaded_model_config=config,
    )
    return net, inputs, config


def _load_model_native_config(
    adapter: Any, authority: FormalConfigSnapshot | None
) -> Any:
    if not isinstance(authority, FormalConfigSnapshot):
        raise ValueError("formal model config authority is required")
    import_root = _model_import_root(adapter)
    try:
        validate_model_native_yaml_tags(authority.loader_path)
        active_root = _ACTIVE_ROOT.get()
        if active_root is None:
            with _selected_opencood_root(import_root):
                return _invoke_native_loader(authority)
        if active_root != import_root:
            raise ValueError("formal model import transaction conflicts")
        return _invoke_native_loader(authority)
    except Exception as error:
        raise ValueError("formal model config load is invalid") from error


@contextmanager
def formal_model_import_transaction(adapter: Any) -> Iterator[None]:
    """Keep one model root selected for the complete formal scan."""

    with _selected_opencood_root(_model_import_root(adapter)):
        yield


@contextmanager
def _selected_opencood_root(root: Path) -> Iterator[None]:
    with _IMPORT_LOCK:
        if _ACTIVE_ROOT.get() is not None:
            raise ValueError("formal model import transaction conflicts")
        original_cwd = Path.cwd()
        original_path = tuple(sys.path)
        original_modules = _opencood_modules()
        token = _ACTIVE_ROOT.set(root)
        try:
            _clear_opencood_modules()
            root_text = str(root)
            sys.path[:] = [root_text, *(item for item in sys.path if item != root_text)]
            yield
        finally:
            restore_error = _restore_import_state(
                original_cwd, original_path, original_modules, token
            )
            if restore_error is not None:
                raise ValueError(
                    "formal model import state restoration failed"
                ) from restore_error


def _restore_import_state(
    original_cwd: Path,
    original_path: tuple[str, ...],
    original_modules: dict[str, Any],
    token: Any,
) -> Exception | None:
    failures: list[Exception] = []
    actions = (
        lambda: os.chdir(original_cwd),
        lambda: sys.path.__setitem__(slice(None), original_path),
        _clear_opencood_modules,
        lambda: sys.modules.update(original_modules),
        lambda: _ACTIVE_ROOT.reset(token),
    )
    for action in actions:
        _attempt_restore(action, failures)
    return failures[0] if failures else None


def _attempt_restore(
    action: Callable[[], Any], failures: list[Exception]
) -> None:
    try:
        action()
    except Exception as error:  # noqa: BLE001 - every restore must still run.
        failures.append(error)


def _invoke_native_loader(authority: FormalConfigSnapshot) -> Any:
    from opencood.hypes_yaml.yaml_utils import load_yaml

    return load_yaml(str(authority.loader_path))


def _model_import_root(adapter: Any) -> Path:
    root = getattr(adapter, "model_config_import_root", None)
    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("formal model config loader is invalid")
    return root


def _opencood_modules() -> dict[str, Any]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "opencood" or name.startswith("opencood.")
    }


def _clear_opencood_modules() -> None:
    for name in tuple(_opencood_modules()):
        sys.modules.pop(name, None)


__all__ = [
    "FormalTraceBuildMixin",
    "adapter_trace_net",
    "build_model_native_formal_trace",
    "formal_model_import_transaction",
]
