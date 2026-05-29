from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path


# The substrate executioner host binds a single shared workspace root for all
# interviews. Each session gets its own subdir inside that root and tells Nova
# to use `/workspace/<sid>` as its working directory, so per-interview file
# state stays isolated even though substrate's environment is shared.
SUBSTRATE_WORKSPACE_ROOT = Path(
    os.environ.get("SUBSTRATE_WORKSPACE_ROOT", "/tmp/nova-substrate-shared")
)


@dataclass
class Session:
    id: str
    workspace: Path
    nova_session_key: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    finalized: bool = False

    @property
    def draft_path(self) -> Path:
        return self.workspace / "draft.yaml"

    @property
    def finalized_marker(self) -> Path:
        return self.workspace / "FINALIZED"

    @property
    def logical_working_dir(self) -> str:
        # Substrate maps the host root to logical `/workspace`. Our per-session
        # subdir becomes `/workspace/<sid>` from substrate's perspective.
        return f"/workspace/{self.id}"


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        SUBSTRATE_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    def create(self) -> Session:
        sid = uuid.uuid4().hex[:12]
        workspace = SUBSTRATE_WORKSPACE_ROOT / sid
        workspace.mkdir(parents=True, exist_ok=True)
        s = Session(id=sid, workspace=workspace)
        self._sessions[sid] = s
        return s

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)
