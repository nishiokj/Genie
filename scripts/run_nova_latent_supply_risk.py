from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Nova on a latent supply-risk case row.")
    parser.add_argument("--mode", required=True, choices=["baseline", "evidence_first"])
    parser.add_argument("--provider", default="codex")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning", default="medium")
    args = parser.parse_args()

    trial_input_path = Path(os.environ["BUCEPHALUS_TRIAL_INPUT_PATH"])
    result_path = Path(os.environ["BUCEPHALUS_RESULT_PATH"])
    events_path = Path(os.environ.get("BUCEPHALUS_TRAJECTORY_PATH", "/bucephalus/out/nova-events.jsonl"))
    trial = json.loads(trial_input_path.read_text(encoding="utf-8"))
    inputs: dict[str, Any] = trial.get("inputs", trial)
    workspace = Path(inputs["workspace_path"])

    augmented = dict(inputs)
    augmented["variant_mode"] = args.mode
    augmented["instructions"] = _instructions(args.mode)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        json.dump(augmented, handle, indent=2, sort_keys=True)
        handle.write("\n")
        augmented_input = handle.name

    command = [
        "nova",
        "run",
        "--input-file",
        augmented_input,
        "--output",
        str(result_path),
        "--events",
        str(events_path),
        "--working-dir",
        str(workspace),
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--reasoning",
        args.reasoning,
        "--timeout-ms",
        "900000",
        "--dangerous",
    ]
    return subprocess.call(command)


def _instructions(mode: str) -> str:
    if mode == "baseline":
        return (
            "Run the daily risk scan and submit the final structured scan. You may inspect files "
            "and use tools as needed, but keep the response concise."
        )
    return (
        "Use an evidence-first workflow. First read the headline pack, then search company records "
        "for candidate materials, products, suppliers, orders, and inventory. For every plausible "
        "world event, build the chain world event -> latent edge hypothesis -> enterprise dependency "
        "-> business exposure before alerting. Use tools/risk_tools.py calculate_exposure for any "
        "candidate dependency. Submit no_alert when a chain is missing. Cite records and separate "
        "observed facts from hypotheses."
    )


if __name__ == "__main__":
    raise SystemExit(main())
