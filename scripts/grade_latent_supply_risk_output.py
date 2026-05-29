from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - separate grader images may only need JSON.
    yaml = None


WEIGHTS = {
    "result_correct": 20,
    "source_headline": 10,
    "latent_edge": 20,
    "enterprise_path": 20,
    "business_exposure": 15,
    "citations": 10,
    "calibrated_uncertainty": 5,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade a latent supply-risk scan.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracles", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    output = _unwrap_agent_output(_load_structured(Path(args.output)))
    oracles = json.loads(Path(args.oracles).read_text(encoding="utf-8"))
    oracle = oracles[args.case_id]
    report = grade(output, oracle)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["score"] >= 0 else 1


def grade(output: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    notes: list[str] = []
    scan = output.get("scan", {}) if isinstance(output, dict) else {}
    if not _schema_valid(scan):
        return {
            "case_id": oracle["run_id"],
            "passed": False,
            "score": 0.0,
            "checks": {"schema_valid": False},
            "hard_failures": {
                "invented_company_record": False,
                "unsupported_major_false_positive": False,
                "missing_scan_object": not bool(scan),
                "schema_violation": True,
            },
            "false_positive_count": 0,
            "scored_alerts": [],
            "notes": ["submitted scan does not match the required output schema"],
        }
    alerts = scan.get("alerts") if isinstance(scan.get("alerts"), list) else []
    true_alerts = oracle.get("true_alerts", [])
    expected_result = "no_alert" if oracle.get("zero_alert_run") else "alerts_found"

    checks["result_correct"] = scan.get("result") == expected_result
    if not checks["result_correct"]:
        notes.append(f"expected result {expected_result}, got {scan.get('result')!r}")

    false_positive_count = 0
    scored_alerts: list[dict[str, Any]] = []

    if oracle.get("zero_alert_run"):
        checks["source_headline"] = len(alerts) == 0
        checks["latent_edge"] = len(alerts) == 0
        checks["enterprise_path"] = len(alerts) == 0
        checks["business_exposure"] = len(alerts) == 0
        checks["citations"] = bool(scan.get("no_alert_rationale"))
        checks["calibrated_uncertainty"] = bool(scan.get("no_alert_rationale"))
        false_positive_count = len(alerts)
        if false_positive_count:
            notes.append(f"zero-alert run contained {false_positive_count} submitted alerts")
    else:
        aggregate = {
            "source_headline": False,
            "latent_edge": False,
            "enterprise_path": False,
            "business_exposure": False,
            "citations": False,
            "calibrated_uncertainty": False,
        }
        matched_alert_ids: set[str] = set()
        for expected in true_alerts:
            best = _best_alert_match(alerts, expected)
            scored_alerts.append(best)
            if best["matched"]:
                matched_alert_ids.add(best["submitted_alert_id"])
            for key in aggregate:
                aggregate[key] = aggregate[key] or best["checks"][key]
        for alert in alerts:
            if str(alert.get("id")) not in matched_alert_ids and not _matches_any_expected(alert, true_alerts):
                false_positive_count += 1
        checks.update(aggregate)
        if false_positive_count:
            notes.append(f"{false_positive_count} submitted alert(s) did not match the hidden answer key")

    hard_failures = {
        "invented_company_record": _contains_invented_record(scan),
        "unsupported_major_false_positive": false_positive_count > 0,
        "missing_scan_object": not bool(scan),
    }

    raw_points = 0
    for check_id, weight in WEIGHTS.items():
        if checks.get(check_id):
            raw_points += weight

    if hard_failures["invented_company_record"] or hard_failures["missing_scan_object"]:
        score = 0.0
        passed = False
    else:
        penalty = 20 * false_positive_count
        score = max(0.0, (raw_points - penalty) / sum(WEIGHTS.values()))
        passed = score >= 0.85 and checks.get("business_exposure", False) and not hard_failures["unsupported_major_false_positive"]

    return {
        "case_id": oracle["run_id"],
        "passed": passed,
        "score": round(score, 4),
        "checks": checks,
        "hard_failures": hard_failures,
        "false_positive_count": false_positive_count,
        "scored_alerts": scored_alerts,
        "notes": notes,
    }


def _best_alert_match(alerts: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for alert in alerts:
        checks = _score_alert(alert, expected)
        score = sum(1 for value in checks.values() if value)
        candidate = {
            "expected_alert_id": expected["id"],
            "submitted_alert_id": str(alert.get("id", "")),
            "matched": score >= 4,
            "match_score": score,
            "checks": checks,
        }
        if best is None or candidate["match_score"] > best["match_score"]:
            best = candidate
    if best is None:
        return {
            "expected_alert_id": expected["id"],
            "submitted_alert_id": None,
            "matched": False,
            "match_score": 0,
            "checks": {
                "source_headline": False,
                "latent_edge": False,
                "enterprise_path": False,
                "business_exposure": False,
                "citations": False,
                "calibrated_uncertainty": False,
            },
        }
    return best


def _score_alert(alert: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool]:
    source_headlines = {str(item) for item in alert.get("source_headlines", [])}
    expected_headlines = {str(item) for item in expected.get("source_headlines", [])}
    concepts = [str(item).lower() for item in expected["latent_edge"].get("required_concepts", [])]
    latent_text = " ".join(
        str(alert.get(field, ""))
        for field in ["latent_edge_hypothesis", "real_world_prior", "reasoning"]
    ).lower()
    enterprise_path = [str(item) for item in alert.get("enterprise_path", [])]
    enterprise_text = " ".join(enterprise_path).lower()
    expected_path = [str(item) for item in expected.get("hidden_dependency_path", [])]
    exposure = alert.get("business_exposure", {}) if isinstance(alert.get("business_exposure"), dict) else {}
    expected_exposure = expected.get("business_exposure", {})
    citations = alert.get("citations", {}) if isinstance(alert.get("citations"), dict) else {}
    uncertainty = alert.get("uncertainty", [])

    return {
        "source_headline": bool(source_headlines.intersection(expected_headlines)),
        "latent_edge": sum(1 for concept in concepts if concept in latent_text) >= max(2, len(concepts) - 1),
        "enterprise_path": all(item in enterprise_path or item.lower() in enterprise_text for item in expected_path),
        "business_exposure": _exposure_matches(exposure, expected_exposure),
        "citations": bool(citations.get("headlines")) and bool(citations.get("records")),
        "calibrated_uncertainty": bool(uncertainty) and _has_hypothesis_language(latent_text + " " + " ".join(map(str, uncertainty)).lower()),
    }


def _schema_valid(scan: Any) -> bool:
    if not isinstance(scan, dict):
        return False
    if not all(key in scan for key in ("run_id", "result", "alerts", "no_alert_rationale")):
        return False
    if scan.get("result") not in {"alerts_found", "no_alert"} or not isinstance(scan.get("alerts"), list):
        return False
    for alert in scan.get("alerts", []):
        if not isinstance(alert, dict):
            return False
        required = (
            "id",
            "confidence",
            "source_headlines",
            "latent_edge_hypothesis",
            "real_world_prior",
            "enterprise_path",
            "business_exposure",
            "reasoning",
            "citations",
            "uncertainty",
        )
        if not all(key in alert for key in required):
            return False
        exposure = alert.get("business_exposure")
        citations = alert.get("citations")
        if not isinstance(exposure, dict) or not all(
            key in exposure
            for key in ("revenue_at_risk", "affected_orders", "affected_customers", "constrained_inventory")
        ):
            return False
        if not isinstance(citations, dict) or not isinstance(citations.get("headlines"), list) or not isinstance(citations.get("records"), list):
            return False
        if not isinstance(alert.get("source_headlines"), list) or not isinstance(alert.get("enterprise_path"), list) or not isinstance(alert.get("uncertainty"), list):
            return False
    return True


def _exposure_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    actual_orders = {str(item) for item in actual.get("affected_orders", [])}
    actual_customers = {str(item) for item in actual.get("affected_customers", [])}
    expected_orders = {str(item) for item in expected.get("affected_orders", [])}
    expected_customers = {str(item) for item in expected.get("affected_customers", [])}
    try:
        revenue_ok = int(actual.get("revenue_at_risk", -1)) == int(expected.get("revenue_at_risk", -2))
    except (TypeError, ValueError):
        revenue_ok = False
    constrained_ok = bool(actual.get("constrained_inventory")) == bool(expected.get("constrained_inventory"))
    return revenue_ok and constrained_ok and expected_orders.issubset(actual_orders) and expected_customers.issubset(actual_customers)


def _matches_any_expected(alert: dict[str, Any], expected_alerts: list[dict[str, Any]]) -> bool:
    return any(sum(_score_alert(alert, expected).values()) >= 4 for expected in expected_alerts)


def _has_hypothesis_language(text: str) -> bool:
    markers = [
        "may",
        "might",
        "could",
        "plausible",
        "hypothesis",
        "does not prove",
        "uncertain",
        "review",
        "not stated",
        "not confirmed",
        "not yet known",
        "does not quantify",
    ]
    return any(marker in text for marker in markers)


def _contains_invented_record(scan: dict[str, Any]) -> bool:
    known_prefixes = (
        "records/",
        "/workspace/task/",
        "bom.yaml",
        "suppliers.yaml",
        "orders.yaml",
        "inventory.yaml",
        "products.yaml",
        "company-profile.md",
        "events/",
        "headline",
        "H01",
        "H02",
        "H03",
        "H04",
        "H05",
        "H06",
        "H07",
        "H08",
        "H09",
        "H10",
        "H11",
        "H12",
    )
    for alert in scan.get("alerts", []) if isinstance(scan.get("alerts"), list) else []:
        citations = alert.get("citations", {})
        if not isinstance(citations, dict):
            continue
        for record in citations.get("records", []):
            if not str(record).startswith(known_prefixes):
                return True
    return False


def _load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise SystemExit(f"{path} is not JSON and PyYAML is not installed")
    loaded = yaml.safe_load(text)
    return loaded or {}


def _unwrap_agent_output(output: dict[str, Any]) -> dict[str, Any]:
    if isinstance(output, dict) and isinstance(output.get("response"), str):
        try:
            parsed = json.loads(output["response"])
        except json.JSONDecodeError:
            return output
        if isinstance(parsed, dict):
            return parsed
    return output


if __name__ == "__main__":
    raise SystemExit(main())
