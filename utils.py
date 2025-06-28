import yaml
import sys
import subprocess
import shutil
import re
import hashlib
from pathlib import Path


def find_most_recent(root, file):

    last_files = list(root.rglob(file))
    if not last_files:
        return None

    # Sort by modification time
    last_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return last_files[0]


def sha256sum(filename):
    hash_sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


class MyDumper(yaml.Dumper):
    """
    Adjust yaml output to what is expected in gitlab CI...
    """

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def execute(name, cmd, cwd, fail_is_ok, filter_re=None):
    """
    wrapper to execute a shell command
    """

    if filter_re and isinstance(filter_re, str):
        filter_re = re.compile(filter_re)

    print(f"\n➡️  [{name}] {' '.join(cmd)} (cwd={cwd or '$(current)'})")
    print("-------------------------------------------------------------")

    cwd.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    while True:
        stdout_line = process.stdout.readline()

        if stdout_line:
            if not filter_re or not filter_re.search(stdout_line):
                print(stdout_line, end="")

        if not stdout_line and process.poll() is not None:
            break

    if process.returncode:
        fail_symbol = "⚠️" if fail_is_ok else "❌"
        print(f"{fail_symbol} Step '{name}' failed with exit code {process.returncode}")
        assert fail_is_ok
    else:
        print(f"✅ Step '{name}' completed successfully.")

    return
