from __future__ import annotations

import os
from pathlib import Path


def load_local_env(project_root: Path | None = None) -> None:
    root = project_root or Path(__file__).resolve().parents[2]
    for filename in (".env", ".env.local"):
        path = root / filename
        if path.exists():
            _load_env_file(path)


def _load_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
