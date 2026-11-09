"""Exec the exact argv inside the scope, avoiding systemd's $ expansion."""
import json
import os
from pathlib import Path
import sys


if __name__ == "__main__":
    argv = json.loads(Path(sys.argv[1]).read_text())["argv"]
    os.execvpe(argv[0], argv, os.environ)
