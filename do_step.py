import yaml
import sys
import subprocess
import shutil
from pathlib import Path
import hashlib
import utils
from utils import execute, MyDumper, sha256sum


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
    elif (
        run["resume"].lower() == "previous_checkpoint"
        or run["resume"].lower() == "previous_model"
    ):
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
        if run["resume"].lower() == "previous_checkpoint":
            cmd.append(f"--resume_from_checkpoint={previous_checkpoint}")
        else:
            cmd.append(f"--resume_from_model={previous_checkpoint}")
    else:
        assert False

    execute("Train network", cmd, nnue_pytorch_dir, False)

    return


def run_conversion(current_sha, workspace_dir, ci_project_dir, convert):
    """
    Convert the final checkpoint into a .nnue
    """

    nnue_pytorch_dir = (
        workspace_dir / "scratch" / current_sha / "trainer" / "nnue-pytorch"
    )

    checkpoint_dir = (
        workspace_dir
        / "scratch"
        / current_sha
        / "run"
        / "lightning_logs"
        / "version_0"
        / "checkpoints"
    )

    checkpoint = checkpoint_dir / "last.ckpt"
    nnue = checkpoint_dir / "last.nnue"
    binpack = workspace_dir / "data" / convert["binpack"]

    # run the conversion to nnue
    cmd = [
        "python",
        "serialize.py",
        f"{checkpoint}",
        f"{nnue}",
        "--ft_compression=leb128",
        f"--ft_optimize_data={binpack}",
    ]
    cmd = cmd + convert["other_options"]
    execute("Convert to nnue", cmd, nnue_pytorch_dir, False)

    # get sha
    sha = sha256sum(nnue)
    sha_short = sha[:12]
    short_nnue = f"nn-{sha_short}.nnue"
    std_nnue = checkpoint_dir / short_nnue
    shutil.copy(nnue, std_nnue)
    print(f"Last nnue for step {current_sha} is {short_nnue}")
    print(f"nnue available as {std_nnue}")

    # store as an artifact for this run
    artifact_dir = ci_project_dir / f"step_{current_sha}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_nnue = artifact_dir / short_nnue
    shutil.copy(nnue, artifact_nnue)
    print(f"nnue available as artifact step_{current_sha}")

    final_file = workspace_dir / "scratch" / current_sha / "final.yaml"
    final = {"short_nnue": f"{short_nnue}", "std_nnue": f"{std_nnue}"}

    with Path(final_file).open(mode="w", encoding="utf-8") as f:
        yaml.dump(final, f, Dumper=MyDumper, default_flow_style=False, width=300)

    return


def run_step(current_sha, previous_sha, workspace_dir, ci_project_dir):
    """
    Driver to run the step
    """

    with open(workspace_dir / "scratch" / current_sha / "step.yaml") as f:
        step = yaml.safe_load(f)

    assert step["sha"] == current_sha

    ensure_trainer(current_sha, workspace_dir, step["trainer"])
    run_trainer(current_sha, previous_sha, workspace_dir, step["run"])
    run_conversion(current_sha, workspace_dir, ci_project_dir, step["convert"])

    return


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print(
            "Usage: python do_step.py current_sha previous_sha workspace_dir ci_project_dir"
        )
        sys.exit(1)

    current_sha = sys.argv[1]
    previous_sha = sys.argv[2]
    workspace_dir = Path(sys.argv[3])
    ci_project_dir = Path(sys.argv[4])

    run_step(current_sha, previous_sha, workspace_dir, ci_project_dir)
