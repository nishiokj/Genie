from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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
    inputs = trial.get("case", {}).get("inputs") or trial.get("inputs") or trial
    if os.environ.get("BUCEPHALUS_PREFLIGHT_SMOKE") == "1" or "workspace_path" not in inputs:
        _write_contract_probe_result()
        return 0
    output_contract = (
        'Final output must be only one compact JSON object with top-level key "scan". '
        'Never use top-level keys like "decision", "headline_id", "event", "chain_of_evidence", '
        'or "business_exposure" outside scan.alerts[]. Use this shape exactly: '
        '{"scan":{"run_id":"<case run_id>","result":"alerts_found|no_alert","alerts":[{"id":"alert-001",'
        '"confidence":"low|medium|high","source_headlines":["Hxx"],"latent_edge_hypothesis":"...",'
        '"real_world_prior":"...","enterprise_path":["input_id","component_id","product_id","order_id"],'
        '"business_exposure":{"revenue_at_risk":0,"affected_orders":[],"affected_customers":[],'
        '"constrained_inventory":false},"reasoning":"...","citations":{"headlines":["Hxx"],'
        '"records":["records/bom.yaml","records/suppliers.yaml","records/orders.yaml","records/inventory.yaml"]},'
        '"uncertainty":["..."]}],"no_alert_rationale":null}}. '
        'For no_alert, use alerts: [] and a concise no_alert_rationale.'
    )
    direct_read_hint = (
        "Tool-use constraint: read the known files directly instead of broad Grep: "
        "events/headline-pack.md, records/company-profile.md, records/products.yaml, "
        "records/bom.yaml, records/suppliers.yaml, records/orders.yaml, and records/inventory.yaml. "
        "If you use Grep, maxResults must be 50 or lower. "
        "After those files are read, submit the scan without additional exploration."
    )
    instructions = (
        "You are Nova running the RULES_ONLY control variant of a latent-edge experiment. "
        "Read the headline pack and company records, but only raise an alert when the external "
        "event explicitly names a supplier, material, component, product, region, or logistics "
        "lane that also appears in the company records. Do not introduce unstated upstream "
        "chemistry, commodity, process, or geography bridges. If the path requires latent world "
        "knowledge, submit no_alert. "
        + direct_read_hint
        + " "
        + output_contract
    )
    if args.mode == "evidence_first":
        instructions = (
            "Use an evidence-first workflow. Read headlines, search company records, build "
            "the chain from world event to latent edge to enterprise dependency to business "
            "exposure, call the risk_tools calculate_exposure helper for candidate dependencies, "
            "and submit no_alert when any chain link is missing. Before the final answer, you "
            "must run `python3 tools/risk_tools.py calculate_exposure --entity-id <id>` for the "
            "candidate material, component, or product and copy the helper's revenue_at_risk, "
            "open_orders, affected_customers, and constrained_inventory into business_exposure. "
            "The enterprise_path array must include exact record IDs for the input/material, "
            "component, product, and affected open order, for example `input_id -> component_id -> "
            "product_id -> order_id`. "
            "For a plausible latent edge with no current open-order exposure, call calculate_exposure "
            "for the strongest candidate input or product; if the helper returns no open_orders or "
            "revenue_at_risk 0, submit no_alert immediately with that rationale. "
            "Do not keep analyzing after reading the known files and one exposure-helper check. "
            "Do not hand-calculate a net exposure when the helper returns a value. "
            + direct_read_hint
            + " "
            + output_contract
        )

    augmented = {key: value for key, value in inputs.items() if not key.startswith("host_")}
    augmented["prompt"] = instructions
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
    completed = subprocess.run(command, capture_output=True, text=True)
    _normalize_result_envelope(Path(os.environ["BUCEPHALUS_RESULT_PATH"]), inputs["run_id"])
    _write_nova_log("nova-stdout.log", completed.stdout)
    _write_nova_log("nova-stderr.log", completed.stderr)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr[-4000:])
        sys.stderr.flush()
    return completed.returncode


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


def _write_nova_log(name: str, content: str) -> None:
    result_path = Path(os.environ["BUCEPHALUS_RESULT_PATH"])
    log_path = result_path.parent / name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")


def _normalize_result_envelope(result_path: Path, case_id: str) -> None:
    if not result_path.exists():
        return
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(result, dict):
        return
    result["case_id"] = case_id
    if isinstance(result.get("response"), str):
        try:
            response = json.loads(result["response"])
        except json.JSONDecodeError:
            response = None
        if isinstance(response, dict):
            scan = response.get("scan")
            if isinstance(scan, dict):
                scan["run_id"] = case_id
                result["response"] = json.dumps(response, separators=(",", ":"), sort_keys=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
