from .utils import execute
from pathlib import Path
from time import sleep
from huggingface_hub import snapshot_download
import threading
import zstandard as zstd
import gzip
import shutil
import os


def decompress_file_zstd(file_path):
    output_path = file_path[:-4]  # Remove .zst extension
    try:
        print(f"Decompressing: {file_path}")
        with open(file_path, "rb") as compressed, open(
            output_path, "wb"
        ) as decompressed:
            dctx = zstd.ZstdDecompressor()
            dctx.copy_stream(compressed, decompressed)
        os.remove(file_path)
        print(f"Decompressed and removed: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def decompress_file_gz(file_path):
    output_path = file_path[:-3]  # Remove .gz extension
    try:
        print(f"Decompressing: {file_path}")
        with open(file_path, "rb") as compressed, open(
            output_path, "wb"
        ) as decompressed:
            with gzip.open(compressed, "rb") as gzfile:
                shutil.copyfileobj(gzfile, decompressed)
        os.remove(file_path)
        print(f"Decompressed and removed: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def decompress_files_in_threads(file_list, decompress_file):
    threads = []
    for file_path in file_list:
        t = threading.Thread(target=decompress_file, args=(file_path,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def run_data_update(owner: str, repo: str, filenames: list[str]) -> None:
    owner_dir = Path().cwd() / "data" / owner
    repo_dir = owner_dir / repo

    print(f"Ensuring data for {owner}/{repo} in {repo_dir} : " + " ".join(filenames))

    # download the filename or e.g. compressed files like *.zst
    pattern_filenames = []
    for filename in filenames:
        stored_file = repo_dir / filename
        if stored_file.exists():
            continue
        pattern_filenames.append(filename)
        pattern_filenames.append(f"{filename}.zst")
        pattern_filenames.append(f"{filename}.gz")

    # try a couple of times, since we might be overloading hf
    n_repeats = 3
    while True:
        try:
            snapshot_download(
                repo_id=f"{owner}/{repo}",
                repo_type="dataset",
                allow_patterns=pattern_filenames,
                cache_dir=repo_dir,
                local_dir=repo_dir,
                etag_timeout=600,
            )
            print("", flush=True)
            execute("Repo disk usage: ", ["du", "-sh", "."], repo_dir, True)
            break
        except Exception as e:
            print(f"Error during repository update: {e}")
            n_repeats -= 1
            if n_repeats > 0:
                print(f"Retrying in 30 seconds... ({n_repeats} attempts left)")
                sleep(30)
            else:
                raise RuntimeError(
                    f"Failed to update repository {owner}/{repo} after multiple attempts."
                ) from e

    # collect the .zst files that need decompression
    zst_files = []
    for filename in filenames:
        zst_file = repo_dir / f"{filename}.zst"
        std_file = repo_dir / filename
        if zst_file.exists() and not std_file.exists():
            zst_files.append(str(zst_file))
    decompress_files_in_threads(zst_files, decompress_file_zstd)

    # collect the .gz files that need decompression
    gz_files = []
    for filename in filenames:
        gz_file = repo_dir / f"{filename}.gz"
        std_file = repo_dir / filename
        if gz_file.exists() and not std_file.exists():
            gz_files.append(str(gz_file))
    decompress_files_in_threads(gz_files, decompress_file_gz)

    execute("Repo disk usage: ", ["du", "-sh", "."], repo_dir, True)

    # final check
    for filename in filenames:
        stored_file = repo_dir / filename
        if not stored_file.exists():
            raise FileNotFoundError(
                f"File {filename} not found in repository {owner}/{repo} after update."
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensure HF data is present in cwd/data for a given owner, and repo."
    )
    parser.add_argument("owner", help="Repository owner")
    parser.add_argument("repo", help="Repository name")
    parser.add_argument(
        "filenames",
        nargs="+",
        help="Optional list of filenames (at least one required)",
    )
    args = parser.parse_args()

    run_data_update(args.owner, args.repo, args.filenames)
