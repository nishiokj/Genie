from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


AGENT_URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8001")
REPO_ROOT = Path(os.environ.get("PIPELINE_REPO_ROOT", Path(__file__).resolve().parents[2]))
DOMAINS_DIR = Path(os.environ.get("DOMAINS_DIR", str(REPO_ROOT / "domains")))
PIPELINE_PYTHON = Path(
    os.environ.get("PIPELINE_PYTHON", str(REPO_ROOT / ".venv" / "bin" / "python"))
)
RUN_LOGS_DIR = Path(os.environ.get("RUN_LOGS_DIR", "/tmp/synth-run-logs"))
RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)

UI_DIR = Path(__file__).parent / "ui"

app = FastAPI(title="domain-author-app")


# ---------- proxy to agent ----------


class MessageIn(BaseModel):
    message: str


@app.post("/api/sessions")
async def create_session() -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{AGENT_URL}/sessions", timeout=30)
        r.raise_for_status()
        return r.json()


@app.post("/api/sessions/{sid}/message")
async def send_message(sid: str, body: MessageIn) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{AGENT_URL}/sessions/{sid}/message",
            json=body.model_dump(),
            timeout=900,
        )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()


@app.get("/api/sessions/{sid}")
async def get_session(sid: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{AGENT_URL}/sessions/{sid}", timeout=30)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()


@app.post("/api/sessions/{sid}/finalize")
async def force_finalize(sid: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{AGENT_URL}/sessions/{sid}/finalize", timeout=30)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()


# ---------- pipeline run ----------


class GenerateIn(BaseModel):
    target_n: int = 1
    seed: int = 42
    run_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None


class _Run:
    __slots__ = (
        "run_id",
        "domain_id",
        "yaml_path",
        "log_path",
        "process",
        "started_at",
        "command",
    )

    def __init__(
        self,
        run_id: str,
        domain_id: str,
        yaml_path: Path,
        log_path: Path,
        process: subprocess.Popen,
        command: list[str],
    ) -> None:
        self.run_id = run_id
        self.domain_id = domain_id
        self.yaml_path = yaml_path
        self.log_path = log_path
        self.process = process
        self.command = command
        self.started_at = time.time()


_runs: dict[str, _Run] = {}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")


@app.post("/api/sessions/{sid}/generate")
async def generate(sid: str, body: GenerateIn) -> dict:
    # 1. Pull the finalized draft from the agent.
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{AGENT_URL}/sessions/{sid}", timeout=30)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    state = r.json()
    if not state.get("finalized"):
        raise HTTPException(409, "session not finalized")
    draft_yaml = state.get("draft_yaml") or ""
    try:
        draft = yaml.safe_load(draft_yaml) or {}
    except yaml.YAMLError as e:
        raise HTTPException(400, f"draft.yaml is not valid YAML: {e}")
    domain_id = draft.get("domain_id")
    if not isinstance(domain_id, str) or not re.fullmatch(r"[a-z0-9_]+", domain_id):
        raise HTTPException(400, "draft.yaml missing valid domain_id (lower_snake)")

    # 2. Persist the YAML into the pipeline's domains/ tree.
    DOMAINS_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = DOMAINS_DIR / f"{domain_id}.yaml"
    yaml_path.write_text(draft_yaml)

    # 3. Prep run_id + log file.
    run_id = body.run_id or _slug(
        f"ui-{domain_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    )
    if run_id in _runs:
        raise HTTPException(409, f"run_id {run_id} already exists")
    log_path = RUN_LOGS_DIR / f"{run_id}.log"

    # 4. Build the command and spawn.
    if not PIPELINE_PYTHON.exists():
        raise HTTPException(
            500,
            f"pipeline python not found at {PIPELINE_PYTHON}. "
            f"Set PIPELINE_PYTHON or build the repo .venv.",
        )
    cmd = [
        str(PIPELINE_PYTHON),
        "main.py",
        "--domain",
        str(yaml_path.relative_to(REPO_ROOT)) if yaml_path.is_relative_to(REPO_ROOT) else str(yaml_path),
        "--target-n",
        str(body.target_n),
        "--seed",
        str(body.seed),
        "--run-id",
        run_id,
        "--no-progress",
        "--overwrite",
    ]
    if body.model:
        cmd += ["--model", body.model]
    if body.provider:
        cmd += ["--provider", body.provider]

    log_fh = open(log_path, "wb", buffering=0)
    log_fh.write(f"$ {' '.join(cmd)}\n\n".encode())
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env={**os.environ},
    )
    _runs[run_id] = _Run(run_id, domain_id, yaml_path, log_path, proc, cmd)

    return {
        "run_id": run_id,
        "domain_id": domain_id,
        "yaml_path": str(yaml_path),
        "log_path": str(log_path),
        "command": cmd,
        "status_url": f"/api/runs/{run_id}/status",
    }


def _artifact_paths(run_id: str, domain_id: str) -> dict[str, str]:
    """Conventional artifact paths the pipeline writes. Existence checked
    inline by the status endpoint."""
    log_root = REPO_ROOT / "logs" / run_id
    return {
        "corpus": str(REPO_ROOT / "data" / "corpus" / "benchmark" / f"{run_id}.jsonl"),
        "outputs": str(REPO_ROOT / "data" / "outputs" / f"{run_id}.jsonl"),
        "stage_records": str(log_root / "stage_records.jsonl"),
        "validation": str(log_root / "validation.jsonl"),
        "rejections": str(log_root / "rejections.jsonl"),
        "metrics": str(log_root / "metrics.json"),
    }


@app.get("/api/runs/{run_id}/status")
def run_status(run_id: str, tail_lines: int = 200) -> dict:
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    rc = run.process.poll()
    if rc is None:
        state = "running"
    elif rc == 0:
        state = "complete"
    else:
        state = "failed"

    log_text = ""
    if run.log_path.exists():
        try:
            with open(run.log_path, "rb") as fh:
                data = fh.read().decode("utf-8", errors="replace")
            lines = data.splitlines()
            log_text = "\n".join(lines[-tail_lines:])
        except OSError:
            log_text = ""

    artifacts = {k: v for k, v in _artifact_paths(run_id, run.domain_id).items() if Path(v).exists()}

    return {
        "run_id": run_id,
        "domain_id": run.domain_id,
        "state": state,
        "exit_code": rc,
        "elapsed_s": round(time.time() - run.started_at, 1),
        "log_path": str(run.log_path),
        "log_tail": log_text,
        "artifacts": artifacts,
        "command": run.command,
    }


@app.post("/api/runs/{run_id}/cancel")
def run_cancel(run_id: str) -> dict:
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if run.process.poll() is None:
        run.process.terminate()
        try:
            run.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            run.process.kill()
    return {"run_id": run_id, "exit_code": run.process.poll()}


# ---------- misc ----------


@app.get("/healthz")
def healthz() -> dict:
    return {
        "ok": True,
        "repo_root": str(REPO_ROOT),
        "domains_dir": str(DOMAINS_DIR),
        "pipeline_python_exists": PIPELINE_PYTHON.exists(),
        "active_runs": len([r for r in _runs.values() if r.process.poll() is None]),
    }


@app.get("/api/system/healthz")
async def system_healthz() -> dict:
    """Aggregate health probe for the UI. Reports each component the user can
    actually see go red, so a 'Nova daemon died' doesn't masquerade as 'agent
    is slow'.

    - app: always ok if this responds.
    - agent: HTTP probe of the agent's /healthz.
    - substrate: pulled from the agent's healthz (which checks the env exists).
    - nova: TCP connectivity to NOVA_HOST:NOVA_PORT (the daemon socket).
    """
    import asyncio
    import socket

    result: dict = {"app": "ok"}

    # Agent + substrate (single round trip)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{AGENT_URL}/healthz", timeout=3)
        if r.status_code == 200:
            body = r.json()
            result["agent"] = "ok"
            result["substrate"] = "ok" if body.get("substrate_ok") else "warn"
        else:
            result["agent"] = "err"
            result["substrate"] = "unknown"
    except Exception:
        result["agent"] = "err"
        result["substrate"] = "unknown"

    # Nova daemon TCP probe
    nova_host = os.environ.get("NOVA_HOST", "127.0.0.1")
    nova_port = int(os.environ.get("NOVA_PORT", "9556"))

    def probe() -> bool:
        try:
            s = socket.socket()
            s.settimeout(0.5)
            s.connect((nova_host, nova_port))
            s.close()
            return True
        except Exception:
            return False

    result["nova"] = "ok" if await asyncio.to_thread(probe) else "err"
    return result


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")
