import yaml
import re
from pathlib import Path
from .utils import execute
import shutil
import time


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
    base_dir.mkdir(parents=True, exist_ok=True)

    clone_dir = base_dir / "fastchess"
    fastchess_binary = clone_dir / "fastchess"

    for attempt in range(1, max_retries + 1):
        try:
            if not clone_dir.exists():
                execute(
                    f"[attempt {attempt}] clone fastchess",
                    ["git", "clone", "--no-checkout", repo],
                    base_dir,
                    True,
                )

            if not fastchess_binary.exists():
                execute(
                    f"[attempt {attempt}] checkout sha {sha}",
                    ["git", "checkout", "--detach", sha],
                    clone_dir,
                    False,
                )

                execute(
                    f"[attempt {attempt}] build fastchess",
                    ["make", "-j"],
                    clone_dir,
                    False,
                )

            return fastchess_binary

        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if clone_dir.exists():
                shutil.rmtree(clone_dir, ignore_errors=True)

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
    repo = f"https://github.com/{owner}/Stockfish.git"

    target_dir = Path.cwd() / f"scratch/packages/stockfish/{sha}"
    target_dir.mkdir(parents=True, exist_ok=True)

    clone_dir = target_dir / "Stockfish"
    stockfish_src_dir = clone_dir / "src"
    stockfish_binary = stockfish_src_dir / "stockfish"

    for attempt in range(1, max_retries + 1):
        try:
            if not clone_dir.exists():
                execute(
                    f"[attempt {attempt}] clone Stockfish {target}",
                    ["git", "clone", "--no-checkout", repo],
                    target_dir,
                    True,
                )

            if not stockfish_binary.exists():
                execute(
                    f"[attempt {attempt}] checkout sha {sha}",
                    ["git", "checkout", "--detach", sha],
                    stockfish_src_dir,
                    False,
                )

                # explicitly specify native, as ARCH is defined differently in the CI pipeline
                execute(
                    f"[attempt {attempt}] build Stockfish",
                    ["make", "-j", "profile-build", "ARCH=native"],
                    stockfish_src_dir,
                    False,
                )

            return stockfish_binary

        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if clone_dir.exists():
                shutil.rmtree(clone_dir, ignore_errors=True)

            if attempt < max_retries:
                print(f"🔁 Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("❌ All attempts to build Stockfish failed.")
                raise


def run_fastchess(
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

    # TODO ... cleanup how to get the book in place
    book = Path.cwd() / "data" / "UHO_Lichess_4852_v1.epd"
    assert book.exists(), f"{book} does not exist"

    # collect specific options
    tc = test["fastchess"]["options"]["tc"]
    option_hash = test["fastchess"]["options"]["hash"]
    rounds = test["fastchess"]["sprt"]["max_rounds"]
    nElo_interval_midpoint = float(test["fastchess"]["sprt"]["nElo_interval_midpoint"])
    nElo_interval_width = float(test["fastchess"]["sprt"]["nElo_interval_width"])
    elo0 = nElo_interval_midpoint - nElo_interval_width / 2
    elo1 = elo0 + nElo_interval_width

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
    # TODO: should this be configurable for better local testing?
    # TODO: presence of a config file would imply that a restart is possible?
    cmd = [f"{fastchess}"]
    cmd += [
        "-concurrency",
        "280",
        "-force-concurrency",
        "-use-affinity",
        "2-71,74-143,146-215,218-287",
    ]

    cmd += ["-rounds", f"{rounds}", "-games", "2", "-repeat", "-srand", "42"]
    cmd += [
        "-sprt",
        f"elo0={elo0}",
        f"elo1={elo1}",
        "alpha=0.05",
        "beta=0.05",
        "model=normalized",
    ]
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

    # engine configs
    cmd += [
        "-each",
        "proto=uci",
        "option.Threads=1",
        f"option.Hash={option_hash}",
        f"tc={tc}",
    ]

    output = execute(
        f"Run fastchess match for {sha}: {short_nnue}",
        cmd,
        match_dir,
        False,
        r"Finished game|Started game",
    )

    pattern = re.compile(r"nElo\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    winning_net = None
    nElo = None

    for line in output:
        if "H0 was accepted" in line:
            print(f"⚠️  No pass: {short_nnue} failed SPRT")

        if "H1 was accepted" in line:
            print(f"🎉 Success: {short_nnue} passed SPRT")
            winning_net = short_nnue

        match = pattern.match(line)
        if match:
            nElo = float(match.group(1))

    return winning_net, nElo


def run_test(test_config_sha, testing_sha):
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
    parser.add_argument("test_config_sha", help="Test config SHA")
    parser.add_argument("testing_sha", help="Testing SHA")
    args = parser.parse_args()

    test_config_sha = args.test_config_sha
    testing_sha = args.testing_sha

    winning_net, nElo = run_test(test_config_sha, testing_sha)

    # TODO: exit with error code if winning_net is None ?
