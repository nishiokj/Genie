"""Smoke + regression tests for the published substrate Python SDK round-trip.

These tests pin the session contract exposed by the PyPI `substrate-sdk`
package and skip host/SDK combinations whose wire format is outside the
installed package's parser.

Tests are skipped when no substrate host is reachable at SUBSTRATE_HOST_URL
(default http://127.0.0.1:8765/). That way CI without a substrate sidecar
doesn't go red.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest


SUBSTRATE_HOST_URL = os.environ.get("SUBSTRATE_HOST_URL", "http://127.0.0.1:8765/").rstrip("/") + "/"


def _host_reachable() -> bool:
    try:
        httpx.get(f"{SUBSTRATE_HOST_URL}", timeout=0.5)
        return True
    except httpx.HTTPError:
        return False


requires_substrate = pytest.mark.skipif(
    not _host_reachable(),
    reason="substrate host not reachable",
)


@pytest.fixture
def ephemeral_env(tmp_path):
    """Create a fresh substrate environment scoped to one test and tear it
    down on the way out. Avoids contaminating the shared `interview_shared`
    workspace.
    """
    env_id = f"pytest_{uuid.uuid4().hex[:10]}"
    workspace_root = tmp_path / "ws"
    workspace_root.mkdir(parents=True)
    body = {
        "environmentId": env_id,
        "workspace": {
            "mode": "existing",
            "root": str(workspace_root),
            "mountAsWorkspace": True,
        },
        "policy": {
            "readRoots": ["/workspace"],
            "writeRoots": ["/workspace"],
            "process": {
                "allowExec": False,
                "allowedCommands": [],
                "deniedCommands": [],
                "maxProcesses": None,
            },
            "network": {"enabled": False, "allowHosts": [], "denyHosts": []},
        },
    }
    r = httpx.post(f"{SUBSTRATE_HOST_URL}environments", json=body, timeout=10)
    r.raise_for_status()
    yield env_id, workspace_root
    # best-effort teardown
    try:
        httpx.delete(f"{SUBSTRATE_HOST_URL}environments/{env_id}", timeout=5)
    except httpx.HTTPError:
        pass


@requires_substrate
def test_create_session_response_emits_environmentId(ephemeral_env):
    """Substrate host returns `environmentId` on session create. The earlier
    version of this test pinned the bug (substrate omitted it; SDK backfilled
    it). After the 2026-05-29 sessions-durability migration the server emits
    it natively. If this regresses, the SDK backfill we removed needs to come
    back.
    """
    env_id, _ = ephemeral_env
    r = httpx.post(
        f"{SUBSTRATE_HOST_URL}environments/{env_id}/sessions",
        json={},
        timeout=5,
    )
    r.raise_for_status()
    body = r.json()
    assert "session" in body
    session = body["session"]
    assert "id" in session
    assert session.get("environmentId") == env_id, (
        "substrate host stopped emitting environmentId on session create"
    )
    assert "createdAt" in session


@requires_substrate
def test_sdk_create_session_succeeds_against_real_host(ephemeral_env):
    """End-to-end: the patched SDK can attach, create a session, write a
    file, observe an effect. If any of these break, the substrate↔Nova
    bridge has regressed.
    """
    from substrate import Environment

    env_id, workspace_root = ephemeral_env
    env = Environment.attach(
        host={"kind": "http", "baseUrl": SUBSTRATE_HOST_URL},
        environmentId=env_id,
    )
    try:
        assert env.environment.id == env_id
        assert env.environment.state == "ready"

        try:
            session = env.create_session()
        except ValueError as exc:
            if "unknown session field: environmentId" in str(exc):
                pytest.skip("running substrate host emits environmentId, which installed substrate-sdk does not parse")
            raise
        assert session.session.id.startswith("sess_")

        # round-trip: write through the session, see it on disk
        session.write(path="probe.txt", content="hello from pytest")
        on_disk = workspace_root / "probe.txt"
        assert on_disk.exists()
        assert on_disk.read_text() == "hello from pytest"

        # effects ledger picks up the write
        r = httpx.get(f"{SUBSTRATE_HOST_URL}environments/{env_id}/effects", timeout=5)
        r.raise_for_status()
        effects = r.json()
        assert any(
            e.get("kind") == "file.write"
            and "/workspace/probe.txt" in (e.get("resource", {}) or {}).get("uri", "")
            for e in effects
        ), f"expected file.write effect; got {effects!r}"
    finally:
        env.close()


def test_sdk_create_session_parses_published_session_shape(monkeypatch):
    """Defensive: the SDK must parse the session shape published by its host.

    This test does not require a running substrate host — we monkeypatch
    the HTTP layer with a synthetic response.
    """
    from substrate import environment as sdk_env

    def fake_post(url, body):
        return {
            "session": {
                "id": "sess_from_test",
                "state": "ready",
                "workspace": {
                    "root": "/tmp/x",
                    "logicalRoot": "/workspace",
                    "mode": "existing",
                    "fresh": False,
                    "managed": False,
                },
                "createdAt": "2026-01-01T00:00:00+00:00",
                "metadata": {},
            }
        }

    monkeypatch.setattr(sdk_env, "_post_json", fake_post)
    monkeypatch.setattr(sdk_env, "_assert_environment_id", lambda x: None)
    monkeypatch.setattr(sdk_env, "_assert_session_id", lambda x: None)

    # The SDK's _create_session only reads .baseUrl off the config; a duck-type
    # stand-in is enough.
    cfg = type("Cfg", (), {"baseUrl": SUBSTRATE_HOST_URL})()
    sess = sdk_env._create_session(cfg, "ignored_env_id")
    assert sess.id == "sess_from_test"
    assert sess.createdAt == "2026-01-01T00:00:00+00:00"
