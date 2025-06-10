import yaml
import sys
import subprocess
import shutil
from pprint import pprint
from pathlib import Path


def execute(name, cmd, cwd, fail_is_ok):
    """
    wrapper to execute a shell command
    """

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
    """
    Install the specified nnue-pytorch trainer
    """

    trainer_dir = workspace_dir / "scratch" / current_sha / "trainer"
    owner = trainer["owner"]
    repo = f"https://github.com/{owner}/nnue-pytorch.git"
    execute(
        "clone trainer",
        ["git", "clone", repo],
        trainer_dir,
        True,
    )

    nnue_pytorch_dir = trainer_dir / "nnue-pytorch"
    sha = trainer["sha"]
    execute("checkout sha", ["git", "checkout", sha], nnue_pytorch_dir, False)
    execute(
        "build data loader",
        ["bash", "compile_data_loader.bat"],
        nnue_pytorch_dir,
        False,
    )

    return


def run_trainer(current_sha, previous_sha, workspace_dir, run):
    """
    Run the training recipe for this step
    """

    nnue_pytorch_dir = (
        workspace_dir / "scratch" / current_sha / "trainer" / "nnue-pytorch"
    )
    data_dir = workspace_dir / "data"
    cmd = ["python", "train.py"]

    for binpack in run["binpacks"]:
        cmd.append(str(data_dir / binpack))

    # some architecture specific options
    cmd.append("--gpus=0,")
    cmd.append("--threads=16")
    cmd.append("--num-workers=16")

    # TODO this is too much output by default
    cmd.append("--enable_progress_bar=false")

    # append all options
    cmd = cmd + run["other_options"]

    # TODO eventually deal with training that would exceed the maximum time limit (roughly 300+ epochs), probably needs splitting, restarting, etc.
    max_epochs = int(run["max_epochs"])
    assert max_epochs <= 300
    cmd.append(f"--max_epochs={max_epochs}")
    cmd.append(f"--network-save-period={max_epochs}")

    # Where to store logs and eventually checkpoints
    root_dir = workspace_dir / "scratch" / current_sha / "run"

    # TODO handle the case of an interrupted/restarted run more cleanly, now just delete whatever is there
    # this is also a race...
    if root_dir.exists():
        shutil.rmtree(root_dir)

    assert not root_dir.exists()

    cmd.append(f"--default_root_dir={root_dir}")

    if run["resume"].lower() == "none":
        assert previous_sha.lower() == "none"
    elif run["resume"].lower() == "previous_checkpoint":
        assert previous_sha.lower() != "none"
        previous_checkpoint = (
            workspace_dir
            / "scratch"
            / previous_sha
            / "run"
            / "lightning_logs"
            / "version_0"
            / "checkpoints"
            / "last.ckpt"
        )
        cmd.append(f"--resume_from_checkpoint={previous_checkpoint}")
    else:
        assert False

    execute("Train network", cmd, nnue_pytorch_dir, False)

    return


def run_step(current_sha, previous_sha, workspace_dir):
    """
    Driver to run the step
    """

    with open(workspace_dir / "scratch" / current_sha / "step.yaml") as f:
        step = yaml.safe_load(f)

    assert step["sha"] == current_sha

    pprint(step)

    ensure_trainer(current_sha, workspace_dir, step["trainer"])
    run_trainer(current_sha, previous_sha, workspace_dir, step["run"])


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python do_step.py current_sha previous_sha workspace_dir")
        sys.exit(1)

    current_sha = sys.argv[1]
    previous_sha = sys.argv[2]
    workspace_dir = Path(sys.argv[3])

    run_step(current_sha, previous_sha, workspace_dir)
