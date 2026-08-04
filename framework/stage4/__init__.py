"""Stage4 reproducible cost-model selection interfaces."""

__all__ = ["run_nested_selection"]


def __getattr__(name: str):
    if name == "run_nested_selection":
        from .cost_model_selection_v1 import run_nested_selection

        return run_nested_selection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
