#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "latent_supply_risk" / "evaluation" / "hero-latent-supply-risk-v1-demo-artifacts.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract demo-ready Latent Edge artifacts from a Bucephalus run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-id", default="LSR-052")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    trials = _load_case_trials(args.run_dir, args.case_id)
    if set(trials) != {"baseline", "evidence_first"}:
        raise SystemExit(f"expected baseline and evidence_first trials for {args.case_id}, found {sorted(trials)}")

    artifact = {
        "artifact_schema": "latent_supply_risk_demo_artifact_v1",
        "source_run": str(args.run_dir),
        "case_id": args.case_id,
        "demo_frame": (
            "Bucephalus runs Nova variants over a Latent Edge experiment, then the demo zooms into "
            "one case to show why evidence-first succeeds where the rules-only control misses."
        ),
        "hot_path": {
            "world_event": None,
            "latent_edge": None,
            "enterprise_path": [],
            "business_exposure": None,
            "final_result": None,
        },
        "variants": {
            variant: _variant_artifact(args.case_id, trial)
            for variant, trial in sorted(trials.items())
        },
    }
    evidence_scan = artifact["variants"]["evidence_first"]["scan"]
    alerts = evidence_scan.get("alerts") or []
    if alerts:
        alert = alerts[0]
        artifact["hot_path"] = {
            "world_event": ", ".join(alert.get("source_headlines", [])),
            "latent_edge": alert.get("latent_edge_hypothesis"),
            "enterprise_path": alert.get("enterprise_path", []),
            "business_exposure": alert.get("business_exposure"),
            "final_result": evidence_scan.get("result"),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(args.out)}, sort_keys=True))
    return 0


def _load_case_trials(run_dir: Path, case_id: str) -> dict[str, Path]:
    trials: dict[str, Path] = {}
    for trial in sorted((run_dir / "trials").glob("trial_*"), key=lambda path: int(path.name.split("_")[1])):
        summary = _load_json(trial / "summary.json")
        if summary["ids"]["task_id"] == case_id:
            trials[summary["ids"]["variant_id"]] = trial
    return trials


def _variant_artifact(case_id: str, trial: Path) -> dict[str, Any]:
    summary = _load_json(trial / "summary.json")
    mapped = _load_json(trial / "grader" / "mapped_output.json")
    result = _load_json(trial / "agent" / "result.json")
    response = _parse_response(result)
    scan = response.get("scan", {}) if isinstance(response, dict) else {}
    events = _parse_events(trial / "agent" / "events.jsonl")
    risk_tool_calls = [_risk_tool_call(event) for event in events if _risk_tool_call(event)]
    exposure_calls = [call for call in risk_tool_calls if call["command_name"] == "calculate_exposure"]
    submit_calls = [call for call in risk_tool_calls if call["command_name"] == "submit_risk_scan"]

    return {
        "case_id": case_id,
        "variant": summary["ids"]["variant_id"],
        "agent_exit_status": summary["agent"]["exit_status"],
        "pass": bool(mapped["payload"]["pass_rate"]),
        "score": mapped["payload"]["score"],
        "scan": scan,
        "risk_tool_trace": risk_tool_calls,
        "exposure_calls": exposure_calls,
        "submit_calls": submit_calls,
        "divergence_label": _divergence_label(scan),
    }


def _risk_tool_call(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("event_type") != "tool_call_end" or event.get("tool", {}).get("name") != "Bash":
        return None
    args = event.get("input", {}).get("arguments", {})
    command = args.get("command")
    if not isinstance(command, str) or "tools/risk_tools.py" not in command:
        return None
    command_name = None
    for candidate in [
        "read_headline_pack",
        "search_company_records",
        "read_record",
        "calculate_exposure",
        "submit_risk_scan",
    ]:
        if f" {candidate}" in command:
            command_name = candidate
            break
    output_text = event.get("output", {}).get("text", "")
    parsed_output: Any = None
    if isinstance(output_text, str):
        try:
            parsed_output = json.loads(output_text)
        except json.JSONDecodeError:
            parsed_output = output_text[:1200]
    return {
        "command_name": command_name or "unknown",
        "command": command,
        "output": parsed_output,
    }


def _divergence_label(scan: dict[str, Any]) -> str:
    result = scan.get("result")
    alerts = scan.get("alerts") if isinstance(scan.get("alerts"), list) else []
    if result == "alerts_found" and alerts:
        return "alert_found"
    if result == "no_alert":
        return "no_alert"
    return "invalid_or_missing_scan"


def _parse_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _parse_response(result: dict[str, Any]) -> dict[str, Any]:
    response = result.get("response")
    if isinstance(response, str):
        return json.loads(response)
    if isinstance(response, dict):
        return response
    return result


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
