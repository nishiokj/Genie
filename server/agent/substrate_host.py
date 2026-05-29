"""Substrate HTTP client utilities.

The substrate executioner host is a long-running process pre-launched by dev.sh
(or docker-compose). It owns one shared environment that all interview sessions
attach to via Nova; per-session isolation comes from each interview using its
own subdirectory within the shared workspace.

Env-create and reuse-or-create logic live here so the launcher can call
`ensure_shared_environment()` exactly once at boot.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx


SUBSTRATE_HOST_URL = os.environ.get("SUBSTRATE_HOST_URL", "http://127.0.0.1:8765/").rstrip("/") + "/"
SHARED_ENV_ID = os.environ.get("SUBSTRATE_ENVIRONMENT_ID", "interview_shared")
SHARED_WORKSPACE_ROOT = Path(
    os.environ.get("SUBSTRATE_WORKSPACE_ROOT", "/tmp/nova-substrate-shared")
)


def _shared_policy() -> dict[str, Any]:
    # Substrate strictly requires policy roots to be /workspace logical paths.
    # apply_patch dispatches Add ops as substrate Write, which respects this.
    # Bash is allowed (the model occasionally falls back to it) with a small
    # set of safe file-creation commands. Substrate's policy gates which
    # commands actually run.
    return {
        "readRoots": ["/workspace"],
        "writeRoots": ["/workspace"],
        "process": {
            "allowExec": True,
            "allowedCommands": [
                "bash",
                "sh",
                "printf",
                "cat",
                "tee",
                "echo",
                "mkdir",
                "ls",
                "rm",
                "mv",
            ],
            "deniedCommands": [],
            "maxProcesses": None,
        },
        "network": {"enabled": False, "allowHosts": [], "denyHosts": []},
    }


def ensure_shared_environment() -> dict[str, Any]:
    """Idempotently create the shared substrate environment Nova attaches to."""
    SHARED_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    body = {
        "environmentId": SHARED_ENV_ID,
        "workspace": {
            "mode": "existing",
            "root": str(SHARED_WORKSPACE_ROOT),
            "mountAsWorkspace": True,
        },
        "policy": _shared_policy(),
    }
    r = httpx.post(f"{SUBSTRATE_HOST_URL}environments", json=body, timeout=30)
    if r.status_code == 200:
        return r.json()
    if r.status_code == 400 and "already exists" in r.text:
        # Pre-existing env from a previous boot is fine.
        existing = httpx.get(
            f"{SUBSTRATE_HOST_URL}environments/{SHARED_ENV_ID}", timeout=10
        )
        existing.raise_for_status()
        return existing.json()
    r.raise_for_status()
    return r.json()


def host_healthy() -> bool:
    """Cheap connectivity check: GET the shared env. 200 means substrate host
    is up and the env exists.
    """
    try:
        r = httpx.get(
            f"{SUBSTRATE_HOST_URL}environments/{SHARED_ENV_ID}", timeout=2
        )
        return r.status_code == 200
    except httpx.HTTPError:
        return False
