import sys
import yaml
import hashlib
import json
from pprint import pprint
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
    execute(
        "build Stockfish",
        ["make", "-j", "profile-build"],
        stockfish_dir,
        False,
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

    pprint(test)

    ensure_fastchess(workspace_dir, ci_commit_sha, test["fastchess"])
    ensure_stockfish(workspace_dir, ci_commit_sha, "reference", test)
    ensure_stockfish(workspace_dir, ci_commit_sha, "testing", test)

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
