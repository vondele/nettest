import yaml
import sys
import subprocess
import shutil
from pathlib import Path
import hashlib
from utils import execute, MyDumper, sha256sum, find_most_recent
import torch


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


def ckpt_reached_end(ckpt_path, max_epochs):

    reached_end = False
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        epoch = ckpt["epoch"]
        print(f"The {ckpt_path} was trained for {epoch + 1} epochs")
        reached_end = epoch + 1 >= max_epochs

    return reached_end


def run_trainer(current_sha, previous_sha, workspace_dir, run):
    """
    Run the training recipe for this step
    """

    nnue_pytorch_dir = (
        workspace_dir / "scratch" / current_sha / "trainer" / "nnue-pytorch"
    )
    data_dir = workspace_dir / "data"

    # first check all binpacks are available in non-compressed form
    for binpack in run["binpacks"]:
        full_path = data_dir / binpack
        # check if it is available in compressed form, and uncompress as needed
        if not full_path.exists():
            full_path_zst = Path(str(full_path) + ".zst")
            if full_path_zst.exists():
                cmd = ["zstd", "-d", str(full_path_zst), "-o", str(full_path)]
                execute("Uncompress binpack.zst", cmd, nnue_pytorch_dir, False)
            else:
                assert False, f"The following binpack could not be found: {binpack}"

    # binding all threads to the same socket is important for performance TODO fix domain
    cmd = ["numactl", "--cpunodebind=0", "--membind=0", "python", "-u", "train.py"]

    for binpack in run["binpacks"]:
        cmd.append(str(data_dir / binpack))

    # some architecture specific options TODO: fix GPU
    cmd.append("--gpus=0,")
    cmd.append("--threads=4")
    # large net needs at least 16 threads, small net >64, number of active threads is seems also roughly half specified
    cmd.append("--num-workers=96")

    # append all options
    cmd = cmd + run["other_options"]

    # TODO probably a bit better handling with the maximum time in the pipeline creation.
    # for now, assume 12h minus 30min safety (eventual net conversion).
    max_time = "00:11:30:00"
    max_epochs = int(run["max_epochs"])
    # assert max_epochs <= 300
    cmd.append(f"--max_time={max_time}")
    cmd.append(f"--max_epochs={max_epochs}")
    cmd.append(f"--network-save-period={max_epochs}")

    # Where to store logs and eventually checkpoints
    root_dir = workspace_dir / "scratch" / current_sha / "run"
    cmd.append(f"--default_root_dir={root_dir}")

    # if the root_dir exists, assume we try to restart from the latest found checkpoint
    resume_this_ckpt = None
    if root_dir.exists():
        resume_this_ckpt = find_most_recent(root_dir, "last.ckpt")
        reached_end = ckpt_reached_end(resume_this_ckpt, max_epochs)
    else:
        reached_end = False

    if resume_this_ckpt:
        cmd.append(f"--resume-from-checkpoint={resume_this_ckpt}")
    else:
        # this is a clean run, follow the description in the recipe
        if run["resume"].lower() == "none":
            assert previous_sha.lower() == "none"
        elif (
            run["resume"].lower() == "previous_checkpoint"
            or run["resume"].lower() == "previous_model"
        ):
            assert previous_sha.lower() != "none"

            final_yaml_file = workspace_dir / "scratch" / previous_sha / "final.yaml"
            assert (
                final_yaml_file.exists()
            ), "The final final yaml file does not exist, a previous step training step did not complete"
            with open(final_yaml_file) as f:
                final = yaml.safe_load(f)
            previous_checkpoint = Path(final["checkpoint"])

            if run["resume"].lower() == "previous_checkpoint":
                cmd.append(f"--resume-from-checkpoint={previous_checkpoint}")
            else:
                previous_model = previous_checkpoint.with_suffix(".pt")
                cmd.append(f"--resume-from-model={previous_model}")
        else:
            assert False

    if not reached_end:
        execute("Train network", cmd, nnue_pytorch_dir, False)
        # now verify if we have reached max_epoch or not
        final_ckpt = find_most_recent(root_dir, "last.ckpt")
        reached_end = ckpt_reached_end(final_ckpt, max_epochs)

    if reached_end:
        print("Success: training reached max_epochs")
        return True
    else:
        print(
            "⚠️  Training did not reach max_epochs ... more iterations will be needed to generate a .nnue"
        )
        return False


def run_conversion(current_sha, workspace_dir, ci_project_dir, convert):
    """
    Convert the final checkpoint into a .nnue and a .pt
    """

    nnue_pytorch_dir = (
        workspace_dir / "scratch" / current_sha / "trainer" / "nnue-pytorch"
    )

    root_dir = workspace_dir / "scratch" / current_sha / "run"

    checkpoint = find_most_recent(root_dir, "last.ckpt")

    nnue = checkpoint.with_suffix(".nnue")
    binpack = workspace_dir / "data" / convert["binpack"]

    # run the conversion to model
    model = checkpoint.with_suffix(".pt")
    cmd = [
        "python",
        "-u",
        "serialize.py",
        f"{checkpoint}",
        f"{model}",
    ]
    cmd = cmd + convert["other_options"]
    execute("Convert to pt", cmd, nnue_pytorch_dir, False)

    # run the conversion to nnue TODO fix device
    cmd = [
        "python",
        "-u",
        "serialize.py",
        f"{checkpoint}",
        f"{nnue}",
        "--ft_compression=leb128",
        f"--ft_optimize_data={binpack}",
        "--device=0",
    ]
    cmd = cmd + convert["other_options"]
    execute("Convert to nnue", cmd, nnue_pytorch_dir, False)

    # get sha
    sha = sha256sum(nnue)
    sha_short = sha[:12]
    short_nnue = f"nn-{sha_short}.nnue"
    std_nnue = nnue.parent / short_nnue
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
    final = {
        "short_nnue": f"{short_nnue}",
        "std_nnue": f"{std_nnue}",
        "checkpoint": f"{checkpoint}",
    }

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
    reached_end = run_trainer(current_sha, previous_sha, workspace_dir, step["run"])

    if reached_end:
        run_conversion(current_sha, workspace_dir, ci_project_dir, step["convert"])

    return


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print(
            "Usage: python -u do_step.py current_sha previous_sha workspace_dir ci_project_dir"
        )
        sys.exit(1)

    current_sha = sys.argv[1]
    previous_sha = sys.argv[2]
    workspace_dir = Path(sys.argv[3])
    ci_project_dir = Path(sys.argv[4])

    run_step(current_sha, previous_sha, workspace_dir, ci_project_dir)
