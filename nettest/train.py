import os
from pathlib import Path
import shutil
import torch
import time
from .utils import execute, MyDumper, sha256sum, find_most_recent, supports_numactl
from .default_environment import get_default_environment
import uuid
import yaml


def ensure_trainer(trainer):
    """
    Install the specified nnue-pytorch trainer
    """

    max_retries = 3
    retry_delay = 30

    sha = trainer["sha"]
    owner = trainer["owner"]
    repo = f"https://github.com/{owner}/nnue-pytorch.git"

    trainer_dir = Path.cwd() / f"scratch/packages/trainer/{sha}"
    nnue_pytorch_dir = trainer_dir / "nnue-pytorch"
    artifact = next(nnue_pytorch_dir.rglob("*data_loader*.so"), None)

    if artifact and artifact.exists():
        return nnue_pytorch_dir

    for attempt in range(1, max_retries + 1):
        unique_suffix = str(uuid.uuid4())
        temp_trainer_dir = (
            trainer_dir.parent / f"{trainer_dir.name}_build_{unique_suffix}"
        )
        temp_nnue_pytorch_dir = temp_trainer_dir / "nnue-pytorch"

        try:
            temp_trainer_dir.mkdir(parents=True, exist_ok=True)

            execute(
                f"[attempt {attempt}] init trainer repo",
                ["git", "init"],
                temp_nnue_pytorch_dir,
                False,
            )

            execute(
                f"[attempt {attempt}] add remote",
                ["git", "remote", "add", "origin", repo],
                temp_nnue_pytorch_dir,
                False,
            )

            execute(
                f"[attempt {attempt}] fetch sha {sha}",
                ["git", "fetch", "--depth", "1", "origin", sha],
                temp_nnue_pytorch_dir,
                False,
            )

            execute(
                f"[attempt {attempt}] checkout sha",
                ["git", "checkout", "--detach", sha],
                temp_nnue_pytorch_dir,
                False,
            )

            execute(
                f"[attempt {attempt}] build data loader",
                ["bash", "setup_script.sh"],
                temp_nnue_pytorch_dir,
                False,
            )

            try:
                temp_trainer_dir.rename(trainer_dir)
            except Exception:
                shutil.rmtree(temp_trainer_dir, ignore_errors=True)

            artifact = next(nnue_pytorch_dir.rglob("*data_loader*.so"), None)
            if artifact and artifact.exists():
                return nnue_pytorch_dir
            raise Exception("Trainer build failed, artifact not found")

        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            shutil.rmtree(temp_trainer_dir, ignore_errors=True)

            if attempt < max_retries:
                print(f"🔁 Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("❌ All attempts failed.")
                raise


def ckpt_reached_end(ckpt_path, max_epochs):
    reached_end = False
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        epoch = ckpt["epoch"]
        print(f"The {ckpt_path} was trained for {epoch + 1} epochs")
        reached_end = epoch + 1 >= max_epochs

    return reached_end


def parse_slurm_timelimit(value: str) -> int:
    hh, mm, ss = value.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def seconds_to_ddhhmmss(total: int) -> str:
    dd = total // 86400
    hh = (total % 86400) // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{dd:02d}:{hh:02d}:{mm:02d}:{ss:02d}"


def run_trainer(environment, current_sha, previous_sha, run, nnue_pytorch_dir):
    """
    Run the training recipe for this step
    """

    data_dir = Path.cwd() / "data"

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

    # some architecture specific options
    run_env = os.environ.copy()
    if "train" in environment and "devices" in environment["train"]:
        devices = environment["train"]["devices"]
        run_env["CUDA_VISIBLE_DEVICES"] = devices
    else:
        devices = "0,"

    num_gpus = len([d for d in devices.split(",") if d.strip()])
    local_devices = "".join([f"{i}," for i in range(num_gpus)])
    nproc = max(1, num_gpus)
    if nproc > 1:
        base_cmd = [
            "torchrun",
            f"--nproc-per-node={nproc}",
            "ddp_launcher.py",
            "train.py",
        ]
    else:
        base_cmd = ["python", "-u", "train.py"]

    cmd_prefix = []
    if nproc == 1 and supports_numactl():
        cpunodebind = environment["train"].get("cpunodebind", "0")
        membind = environment["train"].get("membind", "0")
        cmd_prefix = ["numactl", f"--cpunodebind={cpunodebind}", f"--membind={membind}"]

    cmd = cmd_prefix + base_cmd

    for binpack in run["binpacks"]:
        cmd.append(str(data_dir / binpack))

    if "train" in environment and "threads" in environment["train"]:
        num_threads = environment["train"]["threads"]
    else:
        # seems always a reasonable default
        num_threads = 4
    cmd.append(f"--threads={num_threads}")
    cmd.append(f"--gpus={local_devices}")

    # large net needs at least 16 threads, small net >64, number of active threads is seems also roughly half specified
    if "train" in environment and "workers" in environment["train"]:
        workers = environment["train"]["workers"]
    else:
        cpu_count = os.cpu_count()
        workers = cpu_count * 3 // 2 if cpu_count is not None else 16
    cmd.append(f"--num-workers={workers}")

    # append all options
    cmd = cmd + run["other_options"]

    # for now use 30min less than total allowed time to allow for eventual net conversion.
    end = os.environ.get("SLURM_JOB_END_TIME")
    start = os.environ.get("SLURM_JOB_START_TIME")
    if end is not None and start is not None:
        total_seconds = int(end) - int(start) - 30 * 60
        max_time = seconds_to_ddhhmmss(max(total_seconds, 0))
        cmd.append(f"--max_time={max_time}")

    max_epochs = int(run["max_epochs"])
    cmd.append(f"--max_epochs={max_epochs}")
    cmd.append(f"--network-save-period={max_epochs}")

    # Where to store logs and eventually checkpoints
    root_dir = Path.cwd() / "scratch" / current_sha / "run"
    cmd.append(f"--default_root_dir={root_dir}")

    nsys = environment.get("train", {}).get("nsys")
    if nsys:
        assert shutil.which("nsys"), (
            "nsys requested in environment, but it is not available"
        )
        output = root_dir / nsys.get("output", "nsys-profile")
        nsys_cmd = ["nsys", "profile", "--force-overwrite=true", "-o", str(output)]
        nsys_cmd += nsys.get("args", [])
        cmd = cmd_prefix + nsys_cmd + base_cmd + cmd[len(cmd_prefix) + len(base_cmd) :]

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

            final_yaml_file = Path.cwd() / "scratch" / previous_sha / "final.yaml"
            assert final_yaml_file.exists(), (
                "The final final yaml file does not exist, a previous step training step did not complete"
            )
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
        execute("Train network", cmd, nnue_pytorch_dir, False, env=run_env)
        # now verify if we have reached max_epoch or not
        final_ckpt = find_most_recent(root_dir, "last.ckpt")
        reached_end = ckpt_reached_end(final_ckpt, max_epochs)

    if reached_end:
        print("🎉 Success: training reached max_epochs")
        return True
    else:
        print(
            "⚠️  Training did not reach max_epochs ... more repetitions will be needed to generate a .nnue"
        )
        return False


def run_conversion(environment, current_sha, convert, nnue_pytorch_dir):
    """
    Convert the final checkpoint into a .nnue and a .pt
    """

    root_dir = Path.cwd() / "scratch" / current_sha / "run"

    checkpoint = find_most_recent(root_dir, "last.ckpt")
    assert checkpoint is not None, "No checkpoint found in the run directory"

    # run the conversion to model
    model = checkpoint.with_suffix(".pt")
    cmd = [
        "python",
        "-u",
        "serialize.py",
        f"{checkpoint}",
        f"{model}",
    ]
    cmd = cmd + convert["checkpoint2nnue"]
    execute("Convert to pt", cmd, nnue_pytorch_dir, False)

    # run the conversion to nnue, no optimization here
    if "optimize" in convert:
        destination = checkpoint.parent / "nonopt.nnue"
    else:
        destination = checkpoint.with_suffix(".nnue")

    cmd = [
        "python",
        "-u",
        "serialize.py",
        f"{checkpoint}",
        f"{destination}",
    ]
    cmd = cmd + convert["checkpoint2nnue"]
    execute("Convert to nnue", cmd, nnue_pytorch_dir, False)

    # optimize as a second step (see https://github.com/official-stockfish/nnue-pytorch/issues/322)
    if "optimize" in convert:
        assert "binpack" in convert, "optimize on conversion, requires binpack entry"
        binpack = Path.cwd() / "data" / convert["binpack"]
        source = destination
        nnue = checkpoint.with_suffix(".nnue")
        if "train" in environment and "devices" in environment["train"]:
            device = [
                int(x) for x in environment["train"]["devices"].rstrip(",").split(",")
            ][0]
        else:
            device = "0"
        cmd = [
            "python",
            "-u",
            "serialize.py",
            f"{source}",
            f"{nnue}",
            f"--ft_optimize_data={binpack}",
            f"--device={device}",
        ]
        cmd = cmd + convert["optimize"]
        execute("Optimize nnue", cmd, nnue_pytorch_dir, False)
    else:
        nnue = destination

    # get sha
    sha = sha256sum(nnue)
    sha_short = sha[:12]
    short_nnue = f"nn-{sha_short}.nnue"
    std_nnue = nnue.parent / short_nnue
    shutil.copy(nnue, std_nnue)
    print(f"Last nnue for step {current_sha} is {short_nnue}")
    print(f"nnue available as {std_nnue}")

    # store as an artifact for this run
    artifact_dir = Path.cwd() / "cidir" / f"step_{current_sha}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_nnue = artifact_dir / short_nnue
    shutil.copy(nnue, artifact_nnue)
    print(f"nnue available as artifact step_{current_sha}")

    final_file = Path.cwd() / "scratch" / current_sha / "final.yaml"
    final = {
        "short_nnue": f"{short_nnue}",
        "std_nnue": f"{std_nnue}",
        "checkpoint": f"{checkpoint}",
    }

    with Path(final_file).open(mode="w", encoding="utf-8") as f:
        yaml.dump(final, f, Dumper=MyDumper, default_flow_style=False, width=300)

    return


def run_step(environment, current_sha, previous_sha):
    """
    Driver to run the step
    """

    if (Path.cwd() / "scratch" / current_sha / "final.yaml").exists():
        print(
            f"⚠️  Step {current_sha} is already final, no work to be done, quick return! Maybe too many repetititions?"
        )
        return

    with open(Path.cwd() / "scratch" / current_sha / "step.yaml") as f:
        step = yaml.safe_load(f)

    assert step["sha"] == current_sha

    nnue_pytorch_dir = ensure_trainer(step["trainer"])
    reached_end = run_trainer(
        environment, current_sha, previous_sha, step["run"], nnue_pytorch_dir
    )

    if reached_end:
        run_conversion(
            environment,
            current_sha,
            step["convert"],
            nnue_pytorch_dir,
        )

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run step with provided SHAs and directories."
    )
    parser.add_argument(
        "--environment", required=False, help="Definition of the environment file"
    )
    parser.add_argument("current_sha", help="Current SHA")
    parser.add_argument("previous_sha", help="Previous SHA")
    args = parser.parse_args()

    if args.environment:
        print("Using environment file: ", args.environment)
        with open(args.environment) as f:
            environment = yaml.safe_load(f)
    else:
        environment = get_default_environment()

    run_step(environment, args.current_sha, args.previous_sha)
