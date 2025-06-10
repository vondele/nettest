import yaml
import sys
import subprocess
from pprint import pprint
from pathlib import Path


def execute(name, cmd, cwd, fail_is_ok):

    print(f"\n→ [{name}] {' '.join(cmd)} (cwd={cwd or '$(current)'})")
    print("-------------------------------------------------------------")

    cwd.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    while True:
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()

        if stdout_line:
            print(stdout_line, end="")
        if stderr_line:
            print(stderr_line, end="")

        if not stdout_line and not stderr_line and process.poll() is not None:
            break

    if process.returncode:
        print(f"❌ Step '{name}' failed with exit code {process.returncode}")
        assert fail_is_ok
    else:
        print(f"✅ Step '{name}' completed successfully.")

    return


def ensure_trainer(current_sha, workspace_dir, trainer):

    trainer_dir = workspace_dir / current_sha / "trainer"
    owner = trainer["owner"]
    repo = f"https://github.com/{owner}/nnue-pytorch.git"
    execute(
        "clone trainer",
        ["git", "clone", repo],
        trainer_dir,
        True,
    )

    trainer_dir = trainer_dir / "nnue-pytorch"
    sha = trainer["sha"]
    execute("checkout sha", ["git", "checkout", sha], trainer_dir, False)
    execute(
        "build data loader", ["bash", "compile_data_loader.bat"], trainer_dir, False
    )

    return


def run_step(current_sha, previous_sha, workspace_dir):

    with open(workspace_dir / current_sha / "step.yaml") as f:
        step = yaml.safe_load(f)

    assert step["sha"] == current_sha

    pprint(step)

    ensure_trainer(current_sha, workspace_dir, step["trainer"])


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python do_step.py current_sha previous_sha workspace_dir")
        sys.exit(1)

    current_sha = sys.argv[1]
    previous_sha = sys.argv[2]
    workspace_dir = Path(sys.argv[3])

    run_step(current_sha, previous_sha, workspace_dir)
