from __future__ import annotations

import tempfile
from pathlib import Path

from sample_outputs import _prompt
from services.execution_workspace import ExecutionWorkspace

_TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def _executioner_artifact(files: list[dict[str, str]], commands: dict[str, str]) -> dict:
    temp_dir = tempfile.TemporaryDirectory(prefix="test-executioner-workspace-")
    _TEMP_DIRS.append(temp_dir)
    workspace = ExecutionWorkspace(root=Path(temp_dir.name), commands=commands)
    for item in files:
        workspace.write_file(item["path"], item["content"])
    payload = workspace.artifact_payload()
    workspace.close()
    return {"kind": "executioner_workspace", "payload": payload}


def test_prompt_includes_executioner_workspace_files() -> None:
    candidate = {
        "agent_artifact": {
            "benchmark_case": {
                "setup": "Run the focused test before answering.",
                "prompt": "Patch the bug and explain the causal invariant.",
            },
            "environment_artifact": _executioner_artifact(
                [
                    {"path": "service/reconcile.py", "content": "def summarize(rows):\n    return {}\n"},
                    {"path": "tests/test_reconcile.py", "content": "def test_refund_boundary():\n    assert False\n"},
                    {"path": "README.md", "content": "Run the tests."},
                ],
                {"test": "pytest -q"},
            ),
        },
        "judge_artifact": {
            "proxy_claim": "Hidden judge-facing causal answer must not be rendered to the evaluated agent.",
        },
    }

    prompt = _prompt(candidate)

    assert "SETUP\nRun the focused test before answering." in prompt
    assert '"test": "pytest -q"' in prompt
    assert "--- service/reconcile.py ---" in prompt
    assert "```python\ndef summarize(rows):\n    return {}\n```" in prompt
    assert "BENCHMARK PROMPT\nPatch the bug and explain the causal invariant." in prompt
    assert "Hidden judge-facing causal answer" not in prompt


def test_prompt_without_workspace_preserves_old_prompt_only_behavior() -> None:
    candidate = {"benchmark_case": {"prompt": "Write a haiku under the visible constraints."}}

    assert _prompt(candidate) == "Write a haiku under the visible constraints."
