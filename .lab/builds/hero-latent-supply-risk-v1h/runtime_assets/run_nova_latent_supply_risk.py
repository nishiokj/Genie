from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["baseline", "evidence_first"])
    parser.add_argument("--provider", default="codex")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning", default="medium")
    args = parser.parse_args()

    trial = json.loads(Path(os.environ["BUCEPHALUS_TRIAL_INPUT_PATH"]).read_text(encoding="utf-8"))
    inputs = trial.get("inputs", trial)
    instructions = "Run the daily risk scan and submit the final structured scan."
    if args.mode == "evidence_first":
        instructions = (
            "Use an evidence-first workflow. Read headlines, search company records, build "
            "the chain from world event to latent edge to enterprise dependency to business "
            "exposure, call the risk_tools calculate_exposure helper for candidate dependencies, "
            "and submit no_alert when any chain link is missing."
        )

    augmented = dict(inputs)
    augmented["variant_mode"] = args.mode
    augmented["instructions"] = instructions

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        json.dump(augmented, handle, indent=2, sort_keys=True)
        handle.write("\n")
        augmented_input = handle.name

    events_path = os.environ.get("BUCEPHALUS_TRAJECTORY_PATH", "/bucephalus/out/nova-events.jsonl")
    command = [
        "nova",
        "run",
        "--input-file",
        augmented_input,
        "--output",
        os.environ["BUCEPHALUS_RESULT_PATH"],
        "--events",
        events_path,
        "--working-dir",
        inputs["workspace_path"],
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


if __name__ == "__main__":
    raise SystemExit(main())
