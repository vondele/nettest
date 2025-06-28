import sys
import yaml
import hashlib
import json
from pathlib import Path
from utils import execute


def ensure_fastchess(workspace_dir, ci_commit_sha, fastchess):
    """
    Install the specified fastchess
    """

    fastchess_dir = workspace_dir / "scratch" / ci_commit_sha / "testing"
    fastchess_dir.mkdir(parents=True, exist_ok=True)

    owner = fastchess["code"]["owner"]
    repo = f"https://github.com/{owner}/fastchess.git"

    execute(
        f"clone fastchess",
        ["git", "clone", repo],
        fastchess_dir,
        True,
    )

    fastchess_dir = fastchess_dir / "fastchess"
    sha = fastchess["code"]["sha"]
    execute(f"checkout sha {sha}", ["git", "checkout", sha], fastchess_dir, False)
    execute("build fastchess", ["make", "-j"], fastchess_dir, False)

    return


def ensure_stockfish(workspace_dir, ci_commit_sha, target, test):
    """
    Install the specified stockfish
    """

    target_config = test[target]
    target_dir = workspace_dir / "scratch" / ci_commit_sha / "testing" / target
    target_dir.mkdir(parents=True, exist_ok=True)

    owner = target_config["code"]["owner"]
    repo = f"https://github.com/{owner}/Stockfish.git"

    execute(
        f"clone Stockfish {target} ",
        ["git", "clone", repo],
        target_dir,
        True,
    )

    stockfish_dir = target_dir / "Stockfish" / "src"
    sha = target_config["code"]["sha"]
    execute(f"checkout sha {sha}", ["git", "checkout", sha], stockfish_dir, False)

    # explicitly specify native, as ARCH is defined differently in the CI pipeline
    execute(
        "build Stockfish",
        ["make", "-j", "profile-build", "ARCH=native"],
        stockfish_dir,
        False,
    )

    return


def run_fastchess(workspace_dir, ci_project_dir, ci_commit_sha, test, testing_shas):
    """
    Run fastchess to rank nets relative to the reference
    """

    stockfish_reference = (
        workspace_dir
        / "scratch"
        / ci_commit_sha
        / "testing"
        / "reference"
        / "Stockfish"
        / "src"
        / "stockfish"
    )
    assert stockfish_reference.exists()
    stockfish_testing = (
        workspace_dir
        / "scratch"
        / ci_commit_sha
        / "testing"
        / "testing"
        / "Stockfish"
        / "src"
        / "stockfish"
    )
    assert stockfish_testing.exists()
    fastchess = (
        workspace_dir
        / "scratch"
        / ci_commit_sha
        / "testing"
        / "fastchess"
        / "fastchess"
    )
    assert fastchess.exists()

    match_dir = workspace_dir / "scratch" / ci_commit_sha / "testing" / "match"
    match_dir.mkdir(parents=True, exist_ok=True)

    # TODO ... cleanup how to get the book in place
    book = workspace_dir / "data" / "UHO_Lichess_4852_v1.epd"

    # collect specific options
    rounds = test["fastchess"]["options"]["rounds"]
    tc = test["fastchess"]["options"]["tc"]
    option_hash = test["fastchess"]["options"]["hash"]

    # take care of small vs big net
    target_net = "EvalFile"
    if "evalfile" in test["fastchess"]["options"]:
       if test["fastchess"]["options"]["evalfile"].lower() == "small":
          target_net = "EvalFileSmall"
       elif test["fastchess"]["options"]["evalfile"].lower() == "big":
          target_net = "EvalFile"
       else:
          assert False, "EvalFile needs to be either small or big"

    # fastchess config
    # TODO in principle one could run SPRT instead of fixed games?
    cmd = ["taskset","--cpu-list","0-71",f"{fastchess}"]
    cmd += ["-rounds", f"{rounds}", "-games", "2", "-repeat", "-srand", "42"]

    # TODO should this be configurable for better local testing?
    cmd += ["-concurrency", "70", "--force-concurrency"]
    cmd += ["-openings", f"file={book}", "format=epd", "order=random"]
    cmd += ["-ratinginterval", "100"]
    cmd += ["-report", "penta=true"]
    cmd += ["-pgnout", "file=match.pgn"]

    # reference engine
    cmd += ["-engine", "name=reference", f"cmd={stockfish_reference}"]

    # add nets to be tested
    for sha in testing_shas:
        final_yaml_file = workspace_dir / "scratch" / sha / "final.yaml"
        assert final_yaml_file.exists()
        with open(final_yaml_file) as f:
            final_config = yaml.safe_load(f)
        short_nnue = final_config["short_nnue"]
        std_nnue = final_config["std_nnue"]
        name = f"step_{sha}_{short_nnue}"
        cmd += [
            "-engine",
            f"name={name}",
            f"cmd={stockfish_testing}",
            f"option.{target_net}={std_nnue}",
        ]

    # engine configs
    cmd += [
        "-each",
        "proto=uci",
        "option.Threads=1",
        f"option.Hash={option_hash}",
        f"tc={tc}",
    ]

    execute(
        "Run fastchess match",
        cmd,
        match_dir,
        False,
        r"Finished game|Started game",
    )

    return


def run_test(workspace_dir, ci_project_dir, ci_commit_sha, testing_shas):
    """
    Driver to run the test
    """

    with open(
        workspace_dir / "scratch" / ci_commit_sha / "testing" / "testing.yaml"
    ) as f:
        test = yaml.safe_load(f)

    ensure_fastchess(workspace_dir, ci_commit_sha, test["fastchess"])
    ensure_stockfish(workspace_dir, ci_commit_sha, "reference", test)
    ensure_stockfish(workspace_dir, ci_commit_sha, "testing", test)
    run_fastchess(workspace_dir, ci_project_dir, ci_commit_sha, test, testing_shas)

    return


if __name__ == "__main__":

    if len(sys.argv) < 5:
        print(
            "Usage: python do_testing.py workspace_dir ci_project_dir ci_commit_sha testing_sha1 testing_sha2 ..."
        )
        sys.exit(1)

    workspace_dir = Path(sys.argv[1])
    ci_project_dir = Path(sys.argv[2])
    ci_commit_sha = sys.argv[3]
    testing_shas = sys.argv[4:]

    run_test(workspace_dir, ci_project_dir, ci_commit_sha, testing_shas)
