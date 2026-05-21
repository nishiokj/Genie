from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from config import ModelConfig, load_env_file
from model_client import ModelClient
from models import stable_hash, utc_now_iso
from provider_errors import ProviderError
from services.execution_workspace import ExecutionWorkspace


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate model outputs for committed benchmark cases.")
    parser.add_argument("run_id", help="Corpus run id, e.g. probe-principles-11")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/outputs")
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--auth-file", default=None, help="Provider auth file path. Codex defaults to ~/.codex/auth.json.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--index", type=int, default=None, help="0-based corpus row to run. Defaults to the last rows.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--overwrite", action="store_true", help="Replace output file for this run.")
    args = parser.parse_args()

    corpus_path = Path(args.data_dir) / "corpus" / "benchmark" / f"{args.run_id}.jsonl"
    rows = _read_jsonl(corpus_path)
    if not rows:
        print(f"no committed benchmark cases found at {corpus_path}", file=sys.stderr)
        return 1

    selected = _select_rows(rows, index=args.index, limit=args.limit)
    output_path = Path(args.output_dir) / f"{args.run_id}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    load_env_file()
    client = ModelClient(
        ModelConfig(
            provider=args.provider or "openai",
            model=args.model or "gpt-5.5",
            base_url=args.base_url,
            auth_file=Path(args.auth_file).expanduser() if args.auth_file else None,
        )
    )

    for row_index, item in selected:
        candidate = item.get("candidate", {})
        prompt = _prompt(candidate)
        if not prompt:
            print(f"row {row_index}: missing candidate.agent_artifact.benchmark_case.prompt", file=sys.stderr)
            continue
        system = (
            "You are the model being evaluated by a benchmark. "
            "Follow the user benchmark prompt exactly. Return only the requested output."
        )
        try:
            output, meta = client.complete_text(system=system, user=prompt, temperature=args.temperature)
        except ProviderError as exc:
            print(f"provider error: {exc}", file=sys.stderr)
            return 2

        record = {
            "id": stable_hash({"run_id": args.run_id, "row_index": row_index, "output": output}),
            "run_id": args.run_id,
            "row_index": row_index,
            "candidate_id": candidate.get("id"),
            "ability": (candidate.get("ability_z") or {}).get("name"),
            "environment": (candidate.get("environment_y") or {}).get("name"),
            "prompt": prompt,
            "output": output,
            "model_metadata": meta,
            "created_at": utc_now_iso(),
        }
        _append_jsonl(output_path, record)
        _print_output(record)

    print(f"\noutputs={output_path}")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _select_rows(rows: list[dict[str, Any]], *, index: int | None, limit: int) -> list[tuple[int, dict[str, Any]]]:
    if index is not None:
        if index < 0 or index >= len(rows):
            raise SystemExit(f"--index out of range; corpus has {len(rows)} row(s)")
        return [(index, rows[index])]
    start = max(0, len(rows) - max(1, limit))
    return list(enumerate(rows[start:], start=start))


def _prompt(candidate: dict[str, Any]) -> str:
    agent_artifact = candidate.get("agent_artifact") if isinstance(candidate.get("agent_artifact"), dict) else {}
    benchmark_case = agent_artifact.get("benchmark_case") if isinstance(agent_artifact.get("benchmark_case"), dict) else {}
    if not benchmark_case:
        benchmark_case = candidate.get("benchmark_case") if isinstance(candidate.get("benchmark_case"), dict) else {}
    prompt = str(benchmark_case.get("prompt") or "")
    artifact = agent_artifact.get("environment_artifact")
    if artifact is None:
        artifact = candidate.get("environment_artifact")
    workspace_payload = None
    if isinstance(artifact, dict) and artifact.get("kind") == "executioner_workspace":
        workspace_payload = artifact.get("payload")
    if not isinstance(workspace_payload, dict):
        return prompt

    sections: list[str] = []
    setup = benchmark_case.get("setup")
    if isinstance(setup, str) and setup.strip():
        sections.append("SETUP\n" + setup.strip())

    workspace = ExecutionWorkspace.from_artifact(workspace_payload)
    if workspace.commands:
        sections.append("COMMANDS\n" + json.dumps(workspace.commands, indent=2, sort_keys=True))
    file_sections = ["REPOSITORY FILES"]
    for path in workspace.list_files():
        content = workspace.read_file(path)
        file_sections.append(f"--- {path} ---\n```{_language_for_path(path)}\n{content.rstrip()}\n```")
    workspace.close()
    sections.append("\n\n".join(file_sections))

    if prompt:
        sections.append("BENCHMARK PROMPT\n" + prompt)
    return "\n\n".join(sections)


def _language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".toml": "toml",
        ".sql": "sql",
        ".sh": "bash",
    }.get(suffix, "")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _print_output(record: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"candidate={record.get('candidate_id')}")
    print(f"ability={record.get('ability')}")
    print(f"environment={record.get('environment')}")
    print("-" * 80)
    print(record["output"])


if __name__ == "__main__":
    raise SystemExit(main())
