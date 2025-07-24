import sys
from utils import execute
from pathlib import Path
from time import sleep


def run_data_update(workspace_dir: Path, owner: str, repo: str):
    owner_dir = workspace_dir / "data" / owner
    repo_dir = owner_dir / repo

    # try a couple of times, since we might be overloading hf
    n_repeats = 3
    while True:
        try:
            if repo_dir.exists():
                cmd = ["git", "pull"]
                execute("pull repo", cmd, repo_dir, False)
            else:
                cmd = [
                    "git",
                    "clone",
                    f"https://huggingface.co/datasets/{owner}/{repo}",
                ]
                execute("clone repo", cmd, owner_dir, False)
            execute("Repo disk usage: ", ["du", "-sh", "."], repo_dir, True)
            return
        except AssertionError as e:
            print(f"Error during repository update: {e}")
            n_repeats -= 1
            if n_repeats > 0:
                print(f"Retrying in 30 seconds... ({n_repeats} attempts left)")
                sleep(30)
            else:
                raise RuntimeError(
                    f"Failed to update repository {owner}/{repo} after multiple attempts."
                ) from e


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python -u ensure_data.py workspace_dir owner repo")
        sys.exit(1)

    workspace_dir = Path(sys.argv[1])
    owner = sys.argv[2]
    repo = sys.argv[3]

    run_data_update(workspace_dir, owner, repo)
