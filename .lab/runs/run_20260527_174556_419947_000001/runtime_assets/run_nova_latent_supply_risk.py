from __future__ import annotations

import argparse
import json
import os
import shutil
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
    if "workspace_path" not in inputs:
        _write_contract_probe_result()
        return 0
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
    nova_command = _nova_command()
    command = [
        *nova_command,
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


def _nova_command() -> list[str]:
    if shutil.which("nova"):
        return ["nova", "run"]
    launcher = Path("/opt/agent/packages/apps/launcher/dist/index.js")
    if launcher.exists() and shutil.which("bun"):
        return ["bun", str(launcher), "run"]
    return ["nova", "run"]


def _write_contract_probe_result() -> None:
    result_path = Path(os.environ["BUCEPHALUS_RESULT_PATH"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "scan": {
                    "run_id": "preflight-contract-probe",
                    "result": "no_alert",
                    "alerts": [],
                    "no_alert_rationale": "Bucephalus contract probe; no case workspace was supplied.",
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    events_path = os.environ.get("BUCEPHALUS_TRAJECTORY_PATH")
    if events_path:
        Path(events_path).parent.mkdir(parents=True, exist_ok=True)
        Path(events_path).write_text("", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
