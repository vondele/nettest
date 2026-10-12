import importlib

__all__ = ["parse_recipe", "run_data_update", "run_step", "run_test", "execute"]

_submodules = {
    "parse_recipe": "generate_pipeline",
    "run_data_update": "ensure_data",
    "run_step": "train",
    "run_test": "test",
    "execute": "execute_recipe",
}


def __getattr__(name):
    if name in _submodules:
        module = importlib.import_module("." + _submodules[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
