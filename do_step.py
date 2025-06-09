import yaml
import sys
from pprint import pprint
from pathlib import Path


def run_step(current_sha, previous_sha, workspace_dir):

    with open(workspace_dir / current_sha / "step.yaml") as f:
        step = yaml.safe_load(f)

    assert step["sha"] == current_sha

    pprint(step)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python do_step.py current_sha previous_sha workspace_dir")
        sys.exit(1)

    current_sha = sys.argv[1]
    previous_sha = sys.argv[2]
    workspace_dir = Path(sys.argv[3])

    run_step(current_sha, previous_sha, workspace_dir)
