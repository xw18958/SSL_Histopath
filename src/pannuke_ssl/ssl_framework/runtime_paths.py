from __future__ import annotations

import os
import re
import socket
import subprocess
import shutil
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_ENV_FILE = PROJECT_ROOT / ".ssl_server.env"
_REQUIRED = ("SSL_DATA_ROOT", "SSL_MODEL_ROOT", "SSL_RUN_ROOT")
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"Invalid runtime env line in {path}: {raw!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"Invalid empty environment key in {path}")
        values[key] = value
    return values


def load_server_environment() -> dict[str, str]:
    os.environ.setdefault("SSL_PROJECT_ROOT", str(PROJECT_ROOT))
    for key, value in _parse_env_file(LOCAL_ENV_FILE).items():
        os.environ.setdefault(key, value)
    return {key: os.environ[key] for key in (*_REQUIRED, "SSL_PROJECT_ROOT") if key in os.environ}


def require_runtime_environment() -> dict[str, str]:
    values = load_server_environment()
    missing = [key for key in _REQUIRED if not values.get(key)]
    if missing:
        raise RuntimeError(
            "Missing machine-local SSL runtime roots: " + ", ".join(missing)
            + f". Create {LOCAL_ENV_FILE} from configs/ssl_standard/server.env.example."
        )
    return values


def expand_runtime_string(value: str) -> str:
    load_server_environment()
    missing = sorted({name for name in _VAR.findall(value) if not os.environ.get(name)})
    if missing:
        raise RuntimeError(
            f"Cannot resolve runtime path {value!r}; missing environment variables: {missing}. "
            f"Create {LOCAL_ENV_FILE} from configs/ssl_standard/server.env.example."
        )
    return os.path.expanduser(os.path.expandvars(value))


def expand_runtime_paths(value: Any) -> Any:
    if isinstance(value, str):
        return expand_runtime_string(value) if "${" in value or value.startswith("~") else value
    if isinstance(value, list):
        return [expand_runtime_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_runtime_paths(item) for item in value)
    if isinstance(value, Mapping):
        return {key: expand_runtime_paths(item) for key, item in value.items()}
    return value



def _git_executable() -> str | None:
    found = shutil.which("git")
    if found:
        return found
    candidate = Path.home() / ".local/bin/git"
    return str(candidate) if candidate.is_file() else None

def git_commit() -> str:
    git = _git_executable()
    if not git:
        return "unknown"
    try:
        return subprocess.check_output(
            [git, "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def source_tree_clean() -> bool:
    git = _git_executable()
    if not git:
        return False
    try:
        status = subprocess.check_output(
            [git, "status", "--porcelain", "--", "configs", "scripts", "src", "tests", "manifests", "docs"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return not bool(status.strip())
    except Exception:
        return False


def runtime_identity() -> dict[str, Any]:
    env = require_runtime_environment()
    return {
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "source_tree_clean": source_tree_clean(),
        "project_root": str(PROJECT_ROOT),
        "data_root": env["SSL_DATA_ROOT"],
        "model_root": env["SSL_MODEL_ROOT"],
        "run_root": env["SSL_RUN_ROOT"],
        "results_master": os.environ.get("SSL_RESULTS_MASTER"),
    }
