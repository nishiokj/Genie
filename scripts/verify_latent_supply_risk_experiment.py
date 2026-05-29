#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "latent_supply_risk"
DOMAIN = ROOT / "domains" / "latent_supply_risk.yaml"
TASK_ROWS = DATA_ROOT / "tasks" / "hero-latent-supply-risk-v1-seed-curated.public_case_rows.jsonl"
ORACLES = DATA_ROOT / "evaluation" / "hero-latent-supply-risk-v1-seed-curated.hidden_oracles.json"
REQUIRED_WORKSPACE_FILES = {
    "README.md",
    "events/headline-pack.md",
    "records/company-profile.md",
    "records/products.yaml",
    "records/bom.yaml",
    "records/suppliers.yaml",
    "records/orders.yaml",
    "records/inventory.yaml",
    "tools/risk_tools.py",
    "output/output_schema.json",
}
INTERNAL_ID_RE = re.compile(
    r"\b(?:LSR-[0-9]+|SO-[0-9]+|SUP-[A-Z0-9-]+|SITE-[A-Z0-9-]+|"
    r"CMC-[0-9]+|MTG-[0-9]+|RSG-[0-9]+|EVB-[0-9]+|LTH-[0-9]+|"
    r"CR-[0-9]+|ASM-[0-9]+|POT-[0-9]+|FJ-[0-9]+|LCD-[0-9]+)\b"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Latent Supply Risk corpus and optional Bucephalus run.")
    parser.add_argument("--run-dir", type=Path, help="Bucephalus run directory to verify.")
    args = parser.parse_args()

    failures: list[str] = []
    oracles = _load_json(ORACLES)
    task_rows = [_load_json_line(line) for line in TASK_ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]

    failures.extend(_verify_case_schema(oracles))
    failures.extend(_verify_corpus_shape(oracles, task_rows))
    failures.extend(_verify_workspaces(oracles))
    if args.run_dir:
        failures.extend(_verify_run(args.run_dir, oracles))

    report = {
        "ok": not failures,
        "checked": {
            "cases": len(oracles),
            "task_rows": len(task_rows),
            "run_dir": str(args.run_dir) if args.run_dir else None,
        },
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


def _verify_case_schema(oracles: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    domain = yaml.safe_load(DOMAIN.read_text(encoding="utf-8"))
    validator = Draft202012Validator(domain["benchmark_case_schema"])
    for case_id in sorted(oracles):
        case_path = DATA_ROOT / "cases" / case_id / "benchmark_case.json"
        if not case_path.exists():
            failures.append(f"{case_id}: benchmark_case.json missing")
            continue
        benchmark_case = _load_json(case_path)
        errors = sorted(validator.iter_errors(benchmark_case), key=lambda err: list(err.path))
        if errors:
            rendered = "; ".join(f"{'/'.join(map(str, err.path))}: {err.message}" for err in errors[:3])
            failures.append(f"{case_id}: benchmark_case.json violates domain schema: {rendered}")
    return failures


def _verify_corpus_shape(oracles: dict[str, Any], task_rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    task_ids = [row.get("id") for row in task_rows]
    if len(task_ids) != len(set(task_ids)):
        failures.append("task rows contain duplicate ids")
    if set(task_ids) != set(oracles):
        failures.append(f"task rows and hidden oracles differ: tasks={sorted(task_ids)} oracles={sorted(oracles)}")

    positives = [case_id for case_id, oracle in oracles.items() if oracle.get("true_alerts")]
    zero_alerts = [case_id for case_id, oracle in oracles.items() if oracle.get("zero_alert_run")]
    no_current_exposure = [case_id for case_id, oracle in oracles.items() if oracle.get("zero_alert_run") and "exposure" in json.dumps(oracle).lower()]
    if len(oracles) < 4:
        failures.append("corpus has fewer than four cases")
    if len(positives) < 2:
        failures.append("corpus has fewer than two positive latent-risk cases")
    if not zero_alerts:
        failures.append("corpus has no zero-alert case")
    if not no_current_exposure:
        failures.append("corpus has no explicit no-current-exposure control")
    return failures


def _verify_workspaces(oracles: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for case_id in sorted(oracles):
        workspace = DATA_ROOT / "workspaces" / case_id
        if not workspace.exists():
            failures.append(f"{case_id}: workspace missing")
            continue

        files = {str(path.relative_to(workspace)) for path in workspace.rglob("*") if path.is_file()}
        missing = sorted(REQUIRED_WORKSPACE_FILES - files)
        if missing:
            failures.append(f"{case_id}: missing public workspace files {missing}")

        hidden = sorted(
            str(path.relative_to(workspace))
            for path in workspace.rglob("*")
            if path.is_file()
            and (
                "hidden" in path.name.lower()
                or "oracle" in path.name.lower()
                or "evaluation" in path.parts
                or path.name == "trial_input.json"
            )
        )
        if hidden:
            failures.append(f"{case_id}: hidden/evaluation files leaked into workspace {hidden}")

        headline_path = workspace / "events" / "headline-pack.md"
        if headline_path.exists():
            headline = headline_path.read_text(encoding="utf-8")
            items = re.findall(r"^## H[0-9]{2}$", headline, flags=re.MULTILINE)
            leaks = INTERNAL_ID_RE.findall(headline)
            if not 8 <= len(items) <= 12:
                failures.append(f"{case_id}: headline pack has {len(items)} items, expected 8-12")
            if leaks:
                failures.append(f"{case_id}: headline pack leaks internal ids {sorted(set(leaks))}")
        else:
            failures.append(f"{case_id}: headline pack missing")
    return failures


def _verify_run(run_dir: Path, oracles: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not run_dir.exists():
        return [f"run directory does not exist: {run_dir}"]

    schema = _load_json(DATA_ROOT / "workspaces" / sorted(oracles)[0] / "output" / "output_schema.json")
    validator = Draft202012Validator(schema)
    rows: list[dict[str, Any]] = []
    for trial in sorted((run_dir / "trials").glob("trial_*"), key=lambda path: int(path.name.split("_")[1])):
        summary = _load_json(trial / "summary.json")
        mapped = _load_json(trial / "grader" / "mapped_output.json")
        result = _load_json(trial / "agent" / "result.json")
        case_id = summary["ids"]["task_id"]
        variant = summary["ids"]["variant_id"]
        response = _parse_response(result, trial)
        scan = response.get("scan", {}) if isinstance(response, dict) else {}
        events_path = trial / "agent" / "events.jsonl"
        commands = _risk_tool_commands(events_path)
        trace_violations = _trace_access_violations(events_path)
        schema_errors = list(validator.iter_errors(response))
        rows.append(
            {
                "case_id": case_id,
                "variant": variant,
                "passed": bool(mapped["payload"]["pass_rate"]),
                "score": float(mapped["payload"]["score"]),
                "schema_valid": not schema_errors,
                "exit_status": str(summary["agent"]["exit_status"]),
                "error": result.get("error", {}).get("message") if isinstance(result.get("error"), dict) else None,
                "scan_result": scan.get("result"),
                "alerts": len(scan.get("alerts") or []) if isinstance(scan.get("alerts"), list) else None,
                "commands": commands,
                "trace_violations": trace_violations,
                "read_headline_pack": any(" read_headline_pack" in command for command in commands),
                "read_record": any(" read_record " in command for command in commands),
                "calculate_exposure": any(" calculate_exposure " in command for command in commands),
                "submit_risk_scan": any(" submit_risk_scan " in command for command in commands),
            }
        )

    expected_trials = len(oracles) * 2
    if len(rows) != expected_trials:
        failures.append(f"run has {len(rows)} trials, expected {expected_trials}")

    for row in rows:
        label = f"{row['case_id']}:{row['variant']}"
        if row["exit_status"] != "0" or row["error"]:
            failures.append(f"{label}: agent runtime failed: status={row['exit_status']} error={row['error']}")
        if row["trace_violations"]:
            failures.append(f"{label}: trace access violations {row['trace_violations']}")
        if not row["schema_valid"]:
            failures.append(f"{label}: final response is not schema-valid")
        if not row["read_headline_pack"] or not row["read_record"] or not row["submit_risk_scan"]:
            failures.append(f"{label}: trace does not use required workspace API/read/submit path")
        if row["variant"] == "evidence_first" and not row["calculate_exposure"]:
            failures.append(f"{label}: evidence-first trace did not call calculate_exposure")
        if row["variant"] == "evidence_first" and oracles[row["case_id"]].get("true_alerts"):
            expected_entity = str(oracles[row["case_id"]]["true_alerts"][0]["hidden_dependency_path"][0])
            exposure_commands = [command for command in row["commands"] if " calculate_exposure " in command]
            if not any(expected_entity in command for command in exposure_commands):
                failures.append(f"{label}: evidence-first exposure call did not target expected dependency {expected_entity}")

        is_zero_alert = bool(oracles[row["case_id"]].get("zero_alert_run"))
        if row["variant"] == "evidence_first" and not row["passed"]:
            failures.append(f"{label}: evidence-first did not pass hidden oracle")
        if row["variant"] == "baseline":
            expected_baseline_pass = is_zero_alert
            if bool(row["passed"]) != expected_baseline_pass:
                failures.append(f"{label}: baseline pass={row['passed']} expected {expected_baseline_pass}")

    return failures


def _risk_tool_commands(events_path: Path) -> list[str]:
    if not events_path.exists():
        return []
    commands: list[str] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event_type") != "tool_call_end":
            continue
        if event.get("tool", {}).get("name") != "Bash":
            continue
        command = event.get("input", {}).get("arguments", {}).get("command")
        if isinstance(command, str) and "tools/risk_tools.py" in command:
            commands.append(command)
    return commands


def _trace_access_violations(events_path: Path) -> list[str]:
    if not events_path.exists():
        return ["missing events.jsonl"]
    violations: list[str] = []
    hidden_markers = ("hidden", "oracle", "evaluation", "trial_input")
    public_record_markers = ("events/", "records/")
    for line_no, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event_type") != "tool_call_end":
            continue
        tool_name = event.get("tool", {}).get("name")
        arguments = event.get("input", {}).get("arguments", {})
        rendered = json.dumps(arguments, sort_keys=True)
        rendered_lower = rendered.lower()
        if any(marker in rendered_lower for marker in hidden_markers):
            violations.append(f"line {line_no}: tool input references hidden/evaluation material")
        if tool_name in {"Read", "Grep", "Glob"} and any(marker in rendered for marker in public_record_markers):
            violations.append(f"line {line_no}: {tool_name} accessed public records/headlines outside risk_tools API")
        if tool_name == "Bash":
            command = str(arguments.get("command", ""))
            if _bash_direct_public_access(command) and "tools/risk_tools.py" not in command:
                violations.append(f"line {line_no}: Bash command accessed public records/headlines outside risk_tools API")
    return violations


def _bash_direct_public_access(command: str) -> bool:
    direct_patterns = (
        r"\bcat\s+(?:events|records)/",
        r"\bsed\b[^\n]*(?:events|records)/",
        r"\bawk\b[^\n]*(?:events|records)/",
        r"\bgrep\b[^\n]*(?:events|records)/",
        r"\brg\b[^\n]*(?:events|records)/",
        r"\bhead\b[^\n]*(?:events|records)/",
        r"\btail\b[^\n]*(?:events|records)/",
        r"open\(['\"](?:events|records)/",
        r"Path\(['\"](?:events|records)/",
        r"read_text\([^)]*(?:events|records)/",
    )
    return any(re.search(pattern, command) for pattern in direct_patterns)


def _parse_response(result: dict[str, Any], trial: Path) -> dict[str, Any]:
    response = result.get("response")
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{trial}: response is not JSON: {exc}") from exc
    if isinstance(response, dict):
        return response
    return result


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_line(line: str) -> Any:
    return json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
