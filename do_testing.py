import sys
import yaml
import hashlib
import json
from pathlib import Path
from utils import execute


def ensure_fastchess(workspace_dir, ci_commit_sha, fastchess):
    """
    Install the specified fastchess version
    """

    max_retries = 3
    retry_delay = 30

    sha = fastchess["code"]["sha"]
    owner = fastchess["code"]["owner"]
    repo = f"https://github.com/{owner}/fastchess.git"

    base_dir = workspace_dir / f"scratch/packages/fastchess/{sha}"
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


def ensure_stockfish(workspace_dir, ci_commit_sha, target, test):
    """
    Install the specified Stockfish version
    """

    max_retries = 3
    retry_delay = 30

    target_config = test[target]
    sha = target_config["code"]["sha"]
    owner = target_config["code"]["owner"]
    repo = f"https://github.com/{owner}/Stockfish.git"

    target_dir = workspace_dir / f"scratch/packages/stockfish/{sha}"
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
    workspace_dir,
    ci_project_dir,
    ci_commit_sha,
    test,
    testing_shas,
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

    match_dir = workspace_dir / "scratch" / ci_commit_sha / "testing" / "match"
    match_dir.mkdir(parents=True, exist_ok=True)

    # TODO ... cleanup how to get the book in place
    book = workspace_dir / "data" / "UHO_Lichess_4852_v1.epd"

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

    winning_net = None

    for sha in testing_shas:
        # fastchess config
        # TODO should this be configurable for better local testing?
        cmd = [f"{fastchess}"]
        cmd += ["-concurrency", "280", "-force-concurrency", "-use-affinity", "2-71,74-143,146-215,218-287"]

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
        cmd += ["-pgnout", f"file=match-{sha}.pgn"]

        # add net to be tested
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

        # reference engine
        cmd += ["-engine", "name=reference", f"cmd={stockfish_reference}"]

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

        for line in output:
            if "H0 was accepted" in line:
                print(f"⚠️  No pass: {short_nnue} failed SPRT")
                break

            if "H1 was accepted" in line:
                print(f"🎉 Success: {short_nnue} passed SPRT")
                winning_net = short_nnue
                break

    return winning_net


def run_test(workspace_dir, ci_project_dir, ci_commit_sha, testing_shas):
    """
    Driver to run the test
    """

    with open(
        workspace_dir / "scratch" / ci_commit_sha / "testing" / "testing.yaml"
    ) as f:
        test = yaml.safe_load(f)

    fastchess = ensure_fastchess(workspace_dir, ci_commit_sha, test["fastchess"])
    stockfish_reference = ensure_stockfish(
        workspace_dir, ci_commit_sha, "reference", test
    )
    stockfish_testing = ensure_stockfish(workspace_dir, ci_commit_sha, "testing", test)
    winning_net = run_fastchess(
        workspace_dir,
        ci_project_dir,
        ci_commit_sha,
        test,
        testing_shas,
        fastchess,
        stockfish_reference,
        stockfish_testing,
    )

    return winning_net


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

    winning_net = run_test(workspace_dir, ci_project_dir, ci_commit_sha, testing_shas)

    # TODO exit with error code if winning_net is None ?
