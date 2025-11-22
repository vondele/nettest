import yaml
import re
from pathlib import Path
from .utils import execute
import shutil
import uuid
import time
import os


def ensure_fastchess(fastchess):
    """
    Install the specified fastchess version
    """

    max_retries = 3
    retry_delay = 30

    sha = fastchess["code"]["sha"]
    owner = fastchess["code"]["owner"]
    repo = f"https://github.com/{owner}/fastchess.git"

    base_dir = Path.cwd() / f"scratch/packages/fastchess/{sha}"
    fastchess_binary = base_dir / "fastchess" / "fastchess"

    if fastchess_binary.exists():
        return fastchess_binary

    for attempt in range(1, max_retries + 1):
        unique_suffix = str(uuid.uuid4())
        temp_build_dir = base_dir.parent / f"{base_dir.name}_build_{unique_suffix}"
        temp_fastchess_dir = temp_build_dir / "fastchess"

        try:
            temp_build_dir.mkdir(parents=True, exist_ok=True)
            execute(
                f"[attempt {attempt}] clone fastchess",
                ["git", "clone", "--no-checkout", repo],
                temp_build_dir,
                False,
            )

            execute(
                f"[attempt {attempt}] checkout sha {sha}",
                ["git", "checkout", "--detach", sha],
                temp_fastchess_dir,
                False,
            )

            execute(
                f"[attempt {attempt}] build fastchess",
                ["make", "-j"],
                temp_fastchess_dir,
                False,
            )

            # Try to atomically move the build to the target location
            try:
                temp_build_dir.rename(base_dir)
            except Exception:
                shutil.rmtree(temp_build_dir, ignore_errors=True)

            assert fastchess_binary.exists(), "The binary should, but does not, exist."
            return fastchess_binary

        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            shutil.rmtree(temp_build_dir, ignore_errors=True)
            if attempt < max_retries:
                print(f"🔁 Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("❌ All attempts to build fastchess failed.")
                raise


def ensure_stockfish(target, test):
    """
    Install the specified Stockfish version
    """

    max_retries = 3
    retry_delay = 30

    target_config = test[target]
    sha = target_config["code"]["sha"]
    owner = target_config["code"]["owner"]
    target_build = target_config["code"].get("target", "profile-build")
    repo = f"https://github.com/{owner}/Stockfish.git"

    target_dir = Path.cwd() / f"scratch/packages/stockfish/{sha}-{target_build}"
    stockfish_binary = target_dir / "Stockfish" / "src" / "stockfish"

    if stockfish_binary.exists():
        return stockfish_binary

    for attempt in range(1, max_retries + 1):
        unique_suffix = str(uuid.uuid4())
        temp_build_dir = target_dir.parent / f"{target_dir.name}_build_{unique_suffix}"
        temp_stockfish_src_dir = temp_build_dir / "Stockfish" / "src"

        try:
            temp_build_dir.mkdir(parents=True, exist_ok=True)
            execute(
                f"[attempt {attempt}] clone Stockfish {target}",
                ["git", "clone", "--no-checkout", repo],
                temp_build_dir,
                False,
            )

            execute(
                f"[attempt {attempt}] checkout sha {sha}",
                ["git", "checkout", "--detach", sha],
                temp_stockfish_src_dir,
                False,
            )

            # explicitly define ARCH, as the variable exists in the CI environment
            execute(
                f"[attempt {attempt}] build Stockfish",
                ["make", "-j", f"{target_build}", "ARCH=native"],
                temp_stockfish_src_dir,
                False,
            )

            # Try to atomically move the build to the target location
            try:
                temp_build_dir.rename(target_dir)
            except Exception:
                # If rename fails clean up (maybe another process was faster), cleanup
                shutil.rmtree(temp_build_dir, ignore_errors=True)

            assert stockfish_binary.exists(), "The binary should, but does not, exist."
            return stockfish_binary

        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            shutil.rmtree(temp_build_dir, ignore_errors=True)
            if attempt < max_retries:
                print(f"🔁 Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("❌ All attempts to build Stockfish failed.")
                raise


def run_fastchess(
    environment,
    test_config_sha,
    test,
    testing_sha,
    fastchess,
    stockfish_reference,
    stockfish_testing,
):
    """
    Run fastchess to rank nets relative to the reference
    """

    assert stockfish_reference.exists()
    assert stockfish_testing.exists()
    assert fastchess.exists()

    # download book for testing as needed
    book_dir = Path.cwd() / "data"
    book = book_dir / "UHO_Lichess_4852_v1.epd"
    if not book.exists():
        execute(
            "download book",
            [
                "wget",
                "https://github.com/official-stockfish/books/raw/refs/heads/master/UHO_Lichess_4852_v1.epd.zip",
            ],
            book_dir,
            False,
        )
        execute("unzip book", ["unzip", "UHO_Lichess_4852_v1.epd.zip"], book_dir, False)

    assert book.exists(), f"{book} does not exist"

    # collect specific options
    option_hash = test["fastchess"]["options"]["hash"]
    tc = test["fastchess"]["options"].get("tc", None)
    nodes = test["fastchess"]["options"].get("nodes", None)

    if "max_rounds" in test["fastchess"]["options"]:
        rounds = test["fastchess"]["options"]["max_rounds"]
    else:
        rounds = test["fastchess"]["sprt"][
            "max_rounds"
        ]  # TODO: remove else after fixing all test configs

    sprt_options = []
    if "sprt" in test["fastchess"]:
        nElo_interval_midpoint = float(
            test["fastchess"]["sprt"]["nElo_interval_midpoint"]
        )
        nElo_interval_width = float(test["fastchess"]["sprt"]["nElo_interval_width"])
        elo0 = nElo_interval_midpoint - nElo_interval_width / 2
        elo1 = elo0 + nElo_interval_width
        sprt_options += [
            "-sprt",
            f"elo0={elo0}",
            f"elo1={elo1}",
            "alpha=0.05",
            "beta=0.05",
            "model=normalized",
        ]

    # take care of small vs big net
    target_net = "EvalFile"
    if "evalfile" in test["fastchess"]["options"]:
        if test["fastchess"]["options"]["evalfile"].lower() == "small":
            target_net = "EvalFileSmall"
        elif test["fastchess"]["options"]["evalfile"].lower() == "big":
            target_net = "EvalFile"
        else:
            assert False, "EvalFile needs to be either small or big"

    sha = testing_sha
    match_dir = Path.cwd() / "scratch" / test_config_sha / "match" / sha
    match_dir.mkdir(parents=True, exist_ok=True)

    # fastchess config
    cmd = [f"{fastchess}"]

    if "test" in environment and "concurrency" in environment["test"]:
        concurrency = environment["test"]["concurrency"]
    else:
        concurrency = os.cpu_count()

    cmd += [
        "-concurrency",
        f"{concurrency}",
    ]

    if "test" in environment and "affinity" in environment["test"]:
        affinity = environment["test"]["affinity"]
        cmd += [
            "-use-affinity",
            f"{affinity}",
        ]

    cmd += ["-rounds", f"{rounds}", "-games", "2", "-repeat", "-srand", "42"]
    cmd += sprt_options
    cmd += ["-ratinginterval", "100"]

    cmd += ["-openings", f"file={book}", "format=epd", "order=random"]
    cmd += ["-report", "penta=true"]
    cmd += ["-pgnout", "file=match.pgn"]

    # add net to be tested
    final_yaml_file = Path.cwd() / "scratch" / sha / "final.yaml"
    assert final_yaml_file.exists(), f"{final_yaml_file} does not exist"
    with open(final_yaml_file) as f:
        final_config = yaml.safe_load(f)
    short_nnue = final_config["short_nnue"]
    std_nnue = final_config["std_nnue"]
    assert Path(std_nnue).exists(), f"{std_nnue} does not exist"

    # configure engines
    name = f"step_{sha}_{short_nnue}"
    cmd += [
        "-engine",
        f"name={name}",
        f"cmd={stockfish_testing}",
        f"option.{target_net}={std_nnue}",
    ]

    if "options" in test["testing"]:
        for option in test["testing"]["options"]:
            cmd += [f"option.{option}"]

    # reference engine
    cmd += ["-engine", "name=reference", f"cmd={stockfish_reference}"]

    if "options" in test["reference"]:
        for option in test["reference"]["options"]:
            cmd += [f"option.{option}"]

    # econfigs used for both engines
    cmd += [
        "-each",
        "proto=uci",
        "option.Threads=1",
        f"option.Hash={option_hash}",
    ]

    if tc is not None:
        cmd += [f"tc={tc}"]

    if nodes is not None:
        cmd += [f"nodes={nodes}"]

    # Done setup, run fastchess
    output = execute(
        f"Run fastchess match for {sha}: {short_nnue}",
        cmd,
        match_dir,
        False,
        r"Finished game|Started game",
    )

    pattern = re.compile(r"nElo\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    winning_net = short_nnue
    nElo = None

    # currently there could be multiple lines with H0/H1 accepted, so repeated output is to be expected
    for line in output:
        if "H0 was accepted" in line:
            print(f"⚠️  No pass: {short_nnue} failed SPRT")

        if "H1 was accepted" in line:
            print(f"🎉 Success: {short_nnue} passed SPRT")

        match = pattern.search(line)
        if match:
            nElo = float(match.group(1))

    return winning_net, nElo


def run_test(environment, test_config_sha, testing_sha):
    """
    Driver to run the test
    """

    print(f"Testing config {test_config_sha} for sha {testing_sha}", flush=True)

    with open(Path.cwd() / "scratch" / test_config_sha / "testing.yaml") as f:
        test = yaml.safe_load(f)

    fastchess = ensure_fastchess(test["fastchess"])
    stockfish_reference = ensure_stockfish("reference", test)
    stockfish_testing = ensure_stockfish("testing", test)
    return run_fastchess(
        environment,
        test_config_sha,
        test,
        testing_sha,
        fastchess,
        stockfish_reference,
        stockfish_testing,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run tests for given SHAs.")
    parser.add_argument(
        "--environment", required=False, help="Definition of the environment file"
    )
    parser.add_argument("test_config_sha", help="Test config SHA")
    parser.add_argument("testing_sha", help="Testing SHA")
    args = parser.parse_args()

    if args.environment:
        print("Using environment file: ", args.environment)
        with open(args.environment) as f:
            environment = yaml.safe_load(f)
    else:
        environment = dict()

    test_config_sha = args.test_config_sha
    testing_sha = args.testing_sha

    winning_net, nElo = run_test(environment, test_config_sha, testing_sha)

    # TODO: exit with error code if winning_net is None ?
