import yaml
import sys
import subprocess
import shutil
from pprint import pprint
from pathlib import Path
import hashlib


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


def execute(name, cmd, cwd, fail_is_ok):
    """
    wrapper to execute a shell command
    """

    print(f"\n→ [{name}] {' '.join(cmd)} (cwd={cwd or '$(current)'})")
    print("-------------------------------------------------------------")

    cwd.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    while True:
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()

        if stdout_line:
            print(stdout_line, end="")
        if stderr_line:
            print(stderr_line, end="")

        if not stdout_line and not stderr_line and process.poll() is not None:
            break

    if process.returncode:
        print(f"❌ Step '{name}' failed with exit code {process.returncode}")
        assert fail_is_ok
    else:
        print(f"✅ Step '{name}' completed successfully.")

    return
