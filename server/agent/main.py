from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import yaml as _yaml

from interview import run_turn
from state import SUBSTRATE_WORKSPACE_ROOT, SessionStore
import substrate_host as substrate
from yaml_normalize import field_status as _field_status, normalize_and_validate


logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)

# The shared substrate environment is pre-created (by dev.sh / compose) before
# Nova boots, because Nova-side attach config is loaded once at daemon startup.
# Each interview reuses that env and isolates state via its own working-dir
# subdirectory.
SHARED_ENV_ID = os.environ.get("SUBSTRATE_ENVIRONMENT_ID", "interview_shared")

app = FastAPI(title="domain-author-agent")
store = SessionStore()


class MessageIn(BaseModel):
    message: str


@app.post("/sessions")
def create_session() -> dict:
    s = store.create()
    return {
        "session_id": s.id,
        "workspace": str(s.workspace),
        "logical_workspace": s.logical_working_dir,
        "substrate_env_id": SHARED_ENV_ID,
    }


@app.post("/sessions/{sid}/message")
def post_message(sid: str, body: MessageIn) -> dict:
    s = store.get(sid)
    if s is None:
        raise HTTPException(404, "session not found")
    if s.finalized:
        raise HTTPException(409, "session already finalized")
    return run_turn(s, body.message)


@app.get("/sessions/{sid}")
def get_session(sid: str) -> dict:
    s = store.get(sid)
    if s is None:
        raise HTTPException(404, "session not found")
    draft = s.draft_path.read_text() if s.draft_path.exists() else ""
    parsed: dict | None = None
    if draft:
        try:
            parsed = _yaml.safe_load(draft)
            if not isinstance(parsed, dict):
                parsed = None
        except Exception:
            parsed = None
    return {
        "draft_yaml": draft,
        "finalized": s.finalized,
        "workspace": str(s.workspace),
        "substrate_env_id": SHARED_ENV_ID,
        "field_status": _field_status(parsed),
    }


@app.post("/sessions/{sid}/finalize")
def force_finalize(sid: str) -> dict:
    """Manual override: write the FINALIZED marker without waiting for Nova
    to do it. Useful when the model gets stuck in acknowledgment loops.
    The draft's domain_id (parsed off disk) becomes the marker content.
    Returns 422 if the draft doesn't have a valid domain_id yet.
    """
    s = store.get(sid)
    if s is None:
        raise HTTPException(404, "session not found")
    if not s.draft_path.exists():
        raise HTTPException(422, "no draft.yaml to finalize")
    raw = s.draft_path.read_text()
    result = normalize_and_validate(raw)
    parsed = result.parsed or {}
    domain_id = parsed.get("domain_id")
    if not isinstance(domain_id, str) or not domain_id:
        raise HTTPException(422, "draft.yaml has no domain_id")
    s.finalized_marker.write_text(domain_id)
    s.finalized = True
    return {"finalized": True, "domain_id": domain_id}


@app.get("/healthz")
def healthz() -> dict:
    return {
        "ok": True,
        "substrate_ok": substrate.host_healthy(),
        "workspace_root": str(SUBSTRATE_WORKSPACE_ROOT),
        "substrate_env_id": SHARED_ENV_ID,
    }
