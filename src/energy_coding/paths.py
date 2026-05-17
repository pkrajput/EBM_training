from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def require_submodule(path: str, import_hint: str) -> Path:
    submodule_path = PROJECT_ROOT / path
    if not submodule_path.exists() or not any(submodule_path.iterdir()):
        raise RuntimeError(
            f"Missing required dependency directory: {submodule_path}\n"
            f"Run `bash run/setup_gpu.sh` first. {import_hint}"
        )
    return submodule_path


def add_dependency_paths(require_nanochat: bool = False) -> None:
    ebt_path = require_submodule("EBT", "This repo needs the official EBT code.")
    if str(ebt_path) not in sys.path:
        sys.path.insert(0, str(ebt_path))

    if require_nanochat:
        nanochat_path = require_submodule(
            "nanochat", "CORE and HumanEval evaluation need nanochat."
        )
        if str(nanochat_path) not in sys.path:
            sys.path.insert(0, str(nanochat_path))
