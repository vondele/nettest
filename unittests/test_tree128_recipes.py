"""Keep upstream recipe hyperparameters and normal conversion/crosscheck intact."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL = {"ls-keys", "ls-factors", "ls-remap-data"}


def normalized(value):
    if isinstance(value, dict):
        if "owner" in value and "sha" in value:
            return "pinned-code"
        return {k: normalized(v) for k, v in value.items() if k not in STRUCTURAL}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def test_preserves_upstream_hyperparameters():
    upstream = yaml.safe_load((ROOT / "threats.yaml").read_text())
    for name in ("tree128", "tree128-unfactorized", "tree128-control"):
        recipe = yaml.safe_load((ROOT / f"{name}.yaml").read_text())
        assert normalized(recipe) == normalized(upstream)
        reference = recipe["testing"]["reference"]["code"]
        assert reference["owner"] == "official-stockfish"
        assert reference["sha"] == "3e8fde75412e7c47fc140cd82022a395bc681d92"
        assert recipe["model_config"]["ls-remap-data"] == ""
        for step in recipe["training"]["steps"]:
            flags = step["run"]["other_options"]
            assert flags["ls-keys"] == recipe["model_config"]["ls-keys"]
            assert flags["ls-factors"] == recipe["model_config"]["ls-factors"]
            assert step["convert"]["checkpoint2nnue"]["ls-keys"] == flags["ls-keys"]
            assert step["convert"]["optimize"]["ls-keys"] == flags["ls-keys"]
        assert recipe["testing"]["crosscheck"]["other_options"]["ls-keys"] == recipe["model_config"]["ls-keys"]
