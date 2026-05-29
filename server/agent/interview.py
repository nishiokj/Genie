"""Interview turn driver — talks to a Nova daemon over TCP via the official
Python client. Each interview gets its own Nova session key and its own
subdirectory inside the shared substrate workspace. File writes the model
emits via apply_patch land in substrate's effects ledger (because of the
apply_patch → substrate.Write bridge patched into Nova) and on disk via the
shared volume.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

from nova_client import NovaClient

from prompts import KICKOFF_PROMPT, SESSION_HEADER_TURN
from state import Session
from yaml_normalize import field_status, normalize_and_validate


logger = logging.getLogger("interview")

NOVA_HOST = os.environ.get("NOVA_HOST", "127.0.0.1")
NOVA_PORT = int(os.environ.get("NOVA_PORT", "9556"))
NOVA_AUTH_TOKEN = os.environ.get("NOVA_AUTH_TOKEN") or None
NOVA_TURN_TIMEOUT_S = float(os.environ.get("NOVA_TURN_TIMEOUT_S", "300"))


_client_lock = threading.Lock()
_client: NovaClient | None = None


def _on_event_factory(client: NovaClient):
    """Auto-approve every permission request. Nova still gates execution
    through substrate's policy, so this isn't a security bypass — it just
    keeps the orchestrator unblocked. Permission prompts on a headless
    backend service have nowhere to land otherwise.
    """

    def on_event(event: dict[str, Any], channel: str) -> None:
        t = event.get("type")
        data = event.get("data") or {}
        if t == "permission_request":
            rid = data.get("request_id")
            if rid:
                try:
                    client.respond_to_permission(
                        rid,
                        decision="always_allow",
                        pattern=data.get("suggested_pattern"),
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("permission auto-allow failed for %s: %s", rid, exc)

    return on_event


def _get_client() -> NovaClient:
    global _client
    with _client_lock:
        if _client is None or not _client.connected:
            c = NovaClient(host=NOVA_HOST, port=NOVA_PORT, auth_token=NOVA_AUTH_TOKEN)
            c.on_event = _on_event_factory(c)
            c.connect()
            _client = c
        return _client


def _compose_prompt(session: Session, user_message: str) -> str:
    if not session.history:
        return KICKOFF_PROMPT + "\n\nUser: " + user_message
    return SESSION_HEADER_TURN + "\n\nUser: " + user_message


def _read_assistant_text(response: dict[str, Any]) -> str:
    if not isinstance(response, dict):
        return str(response)
    content = response.get("content")
    if isinstance(content, str) and content.strip():
        return content
    err = response.get("error")
    if isinstance(err, dict):
        return f"[nova error] {err.get('type','?')}: {err.get('message','')}"
    return json.dumps(response)[:4000]


def run_turn(session: Session, user_message: str) -> dict[str, Any]:
    client = _get_client()
    working_dir = session.logical_working_dir

    if session.nova_session_key is None:
        client.init_session(working_dir=working_dir)
        session.nova_session_key = client.session_key

    prompt = _compose_prompt(session, user_message)
    session.history.append({"role": "user", "content": user_message})

    try:
        response = client.run_to_completion(
            prompt,
            session_key=session.nova_session_key,
            working_dir=working_dir,
            timeout=NOVA_TURN_TIMEOUT_S,
        )
    except Exception as exc:
        logger.exception("nova run_to_completion failed for session %s", session.id)
        msg = f"[nova client error] {type(exc).__name__}: {exc}"
        session.history.append({"role": "assistant", "content": msg})
        return {
            "assistant_message": msg,
            "draft_yaml": session.draft_path.read_text() if session.draft_path.exists() else "",
            "finalized": session.finalized,
            "nova_returncode": 1,
        }

    assistant_text = _read_assistant_text(response)
    session.history.append({"role": "assistant", "content": assistant_text})
    session.finalized = session.finalized_marker.exists()

    # Read what Nova authored, then run the normalizer + validator. If we
    # repaired anything, write the normalized YAML back to substrate's
    # workspace so the disk and substrate's effects ledger agree.
    draft_yaml = ""
    repairs: list[str] = []
    validation_error: str | None = None
    parsed = None
    if session.draft_path.exists():
        raw = session.draft_path.read_text()
        result = normalize_and_validate(raw)
        if result.changed and result.yaml_text != raw:
            session.draft_path.write_text(result.yaml_text)
            logger.info(
                "session %s: normalized draft.yaml (%s)",
                session.id,
                "; ".join(result.repairs),
            )
        draft_yaml = result.yaml_text
        repairs = result.repairs
        validation_error = result.validation_error or result.parse_error
        parsed = result.parsed

    return {
        "assistant_message": assistant_text,
        "draft_yaml": draft_yaml,
        "finalized": session.finalized,
        "draft_repairs": repairs,
        "draft_validation_error": validation_error,
        "field_status": field_status(parsed),
        "tools_used": response.get("tools_used") if isinstance(response, dict) else None,
        "usage": response.get("usage") if isinstance(response, dict) else None,
    }
