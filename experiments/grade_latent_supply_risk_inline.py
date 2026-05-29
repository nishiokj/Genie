from __future__ import annotations

import json
from pathlib import Path


ORACLES = {
    "LSR-001": {
        "zero_alert_run": False,
        "true_alerts": [
            {
                "source_headlines": ["H03"],
                "latent_edge": {"required_concepts": ["gujarat", "castor", "polyol", "coating"]},
                "hidden_dependency_path": ["castor_derived_polyol", "CR-88", "CMC-410", "SO-7731"],
                "business_exposure": {
                    "revenue_at_risk": 4200000,
                    "affected_orders": ["SO-7731"],
                    "affected_customers": ["Atlantic Energy"],
                    "constrained_inventory": True,
                },
            }
        ],
    },
    "LSR-014": {
        "zero_alert_run": False,
        "true_alerts": [
            {
                "source_headlines": ["H01"],
                "latent_edge": {"required_concepts": ["resin", "rosin", "flux", "electronics"]},
                "hidden_dependency_path": ["rosin_ester_flux", "ASM-44", "MTG-220", "SO-7844"],
                "business_exposure": {
                    "revenue_at_risk": 3692000,
                    "affected_orders": ["SO-7844"],
                    "affected_customers": ["Mercy North Hospitals"],
                    "constrained_inventory": True,
                },
            }
        ],
    },
    "LSR-027": {"zero_alert_run": True, "true_alerts": []},
    "LSR-033": {
        "zero_alert_run": False,
        "true_alerts": [
            {
                "source_headlines": ["H01"],
                "latent_edge": {"required_concepts": ["silicon metal", "silicone", "elastomer", "potting"]},
                "hidden_dependency_path": ["silicone_elastomer", "POT-72", "RSG-700", "SO-7991"],
                "business_exposure": {
                    "revenue_at_risk": 3330000,
                    "affected_orders": ["SO-7991"],
                    "affected_customers": ["Canal Rail"],
                    "constrained_inventory": True,
                },
            }
        ],
    },
    "LSR-052": {
        "zero_alert_run": False,
        "true_alerts": [
            {
                "source_headlines": ["H01"],
                "latent_edge": {"required_concepts": ["antimony", "trioxide", "flame", "cable"]},
                "hidden_dependency_path": ["antimony_trioxide_synergist", "FJ-31", "EVB-330", "SO-8207"],
                "business_exposure": {
                    "revenue_at_risk": 3360000,
                    "affected_orders": ["SO-8207"],
                    "affected_customers": ["Metro Tunnel Authority"],
                    "constrained_inventory": True,
                },
            }
        ],
    },
    "LSR-041": {"zero_alert_run": True, "true_alerts": []},
}


def main() -> int:
    case_id = Path("/grader/in/case_id.txt").read_text(encoding="utf-8").strip()
    result = json.loads(Path("/grader/in/result.json").read_text(encoding="utf-8"))
    if isinstance(result, dict) and isinstance(result.get("response"), str):
        try:
            result = json.loads(result["response"])
        except json.JSONDecodeError:
            pass
    report = grade(case_id, result)
    Path("/grader/out").mkdir(parents=True, exist_ok=True)
    Path("/grader/out/latent-supply-risk-grade.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def grade(case_id: str, result: dict) -> dict:
    oracle = ORACLES[case_id]
    scan = result.get("scan", {}) if isinstance(result, dict) else {}
    alerts = scan.get("alerts") if isinstance(scan.get("alerts"), list) else []
    expected_result = "no_alert" if oracle["zero_alert_run"] else "alerts_found"
    checks = {"result_correct": scan.get("result") == expected_result}
    false_positive_count = 0
    if oracle["zero_alert_run"]:
        checks.update(
            {
                "source_headline": len(alerts) == 0,
                "latent_edge": len(alerts) == 0,
                "enterprise_path": len(alerts) == 0,
                "business_exposure": len(alerts) == 0,
                "citations": bool(scan.get("no_alert_rationale")),
                "calibrated_uncertainty": bool(scan.get("no_alert_rationale")),
            }
        )
        false_positive_count = len(alerts)
    else:
        expected = oracle["true_alerts"][0]
        alert = alerts[0] if alerts else {}
        text = " ".join(str(alert.get(key, "")) for key in ["latent_edge_hypothesis", "real_world_prior", "reasoning"]).lower()
        path = [str(item) for item in alert.get("enterprise_path", [])]
        path_text = " ".join(path).lower()
        exposure = alert.get("business_exposure", {}) if isinstance(alert.get("business_exposure"), dict) else {}
        citations = alert.get("citations", {}) if isinstance(alert.get("citations"), dict) else {}
        concepts = expected["latent_edge"]["required_concepts"]
        exp = expected["business_exposure"]
        checks.update(
            {
                "source_headline": bool(set(alert.get("source_headlines", [])).intersection(expected["source_headlines"])),
                "latent_edge": sum(1 for concept in concepts if concept.lower() in text) >= max(2, len(concepts) - 1),
                "enterprise_path": all(item in path or item.lower() in path_text for item in expected["hidden_dependency_path"]),
                "business_exposure": _exposure_matches(exposure, exp),
                "citations": bool(citations.get("headlines")) and bool(citations.get("records")),
                "calibrated_uncertainty": bool(alert.get("uncertainty")),
            }
        )
        false_positive_count = max(0, len(alerts) - 1)
    weights = {
        "result_correct": 20,
        "source_headline": 10,
        "latent_edge": 20,
        "enterprise_path": 20,
        "business_exposure": 15,
        "citations": 10,
        "calibrated_uncertainty": 5,
    }
    points = sum(weights[key] for key, value in checks.items() if value)
    score = max(0.0, (points - 20 * false_positive_count) / sum(weights.values()))
    return {
        "case_id": case_id,
        "passed": score >= 0.85 and false_positive_count == 0 and bool(checks.get("business_exposure")),
        "score": round(score, 4),
        "checks": checks,
        "false_positive_count": false_positive_count,
    }


def _exposure_matches(actual: dict, expected: dict) -> bool:
    return (
        _safe_int(actual.get("revenue_at_risk")) == int(expected["revenue_at_risk"])
        and set(expected["affected_orders"]).issubset(set(actual.get("affected_orders", [])))
        and set(expected["affected_customers"]).issubset(set(actual.get("affected_customers", [])))
        and bool(actual.get("constrained_inventory")) == bool(expected["constrained_inventory"])
    )


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
