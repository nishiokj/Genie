"""Make `server/agent` importable from tests without packaging it."""

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# Also expose the repo root so the normalizer's lazy `from config import DomainConfig`
# resolves to the real pipeline schema (not a stub).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
