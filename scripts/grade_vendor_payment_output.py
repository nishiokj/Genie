from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "primary_reason",
        "required_next_step",
        "evidence",
        "confidence",
        "memo_summary",
    ],
    "properties": {
        "decision": {"type": "string", "enum": ["approve", "hold", "escalate"]},
        "primary_reason": {"type": "string", "minLength": 12},
        "required_next_step": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 6},
        },
        "evidence": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key", "source", "value"],
                "properties": {
                    "key": {"type": "string"},
                    "source": {"type": "string"},
                    "value": {},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "memo_summary": {"type": "string", "minLength": 20},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade vendor payment exception agent output.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracles", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    oracles = json.loads(Path(args.oracles).read_text(encoding="utf-8"))
    oracle = oracles[args.case_id]["hidden_oracle"]
    raw = json.loads(Path(args.output).read_text(encoding="utf-8"))
    answer = _extract_answer(raw)

    report = grade_answer(args.case_id, answer, oracle)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


def grade_answer(case_id: str, answer: Any, oracle: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    schema_errors = sorted(Draft202012Validator(OUTPUT_SCHEMA).iter_errors(answer), key=lambda err: list(err.path))
    errors.extend(f"schema:{list(err.path)}:{err.message}" for err in schema_errors)
    if schema_errors:
        return _report(case_id, passed=False, score=0.0, errors=errors, details={})

    expected = oracle["expected_decision"]
    decision_ok = str(answer["decision"]).lower() == expected
    if not decision_ok:
        errors.append(f"decision: expected {expected}, got {answer['decision']}")

    cited = _canonical_evidence_keys(answer["evidence"])
    required = set(oracle["must_cite"])
    missing = sorted(required - cited)
    if missing:
        errors.append("missing_evidence:" + ",".join(missing))

    text = " ".join(
        [
            str(answer["primary_reason"]),
            str(answer["memo_summary"]),
            " ".join(str(step) for step in answer["required_next_step"]),
        ]
    ).lower()
    forbidden_hits = [phrase for phrase in oracle["forbidden_reasoning"] if phrase.lower() in text]
    if forbidden_hits:
        errors.append("forbidden_reasoning:" + ";".join(forbidden_hits))

    decision_score = 0.45 if decision_ok else 0.0
    evidence_score = 0.35 * ((len(required) - len(missing)) / max(1, len(required)))
    next_step_score = 0.15 if _next_step_is_nonempty(answer) else 0.0
    confidence_score = 0.05 if 0 <= float(answer["confidence"]) <= 1 else 0.0
    score = round(decision_score + evidence_score + next_step_score + confidence_score, 4)
    passed = decision_ok and not missing and not forbidden_hits and score >= 0.8
    return _report(
        case_id,
        passed=passed,
        score=score,
        errors=errors,
        details={
            "expected_decision": expected,
            "actual_decision": answer["decision"],
            "cited_evidence": sorted(cited),
            "missing_evidence": missing,
            "forbidden_hits": forbidden_hits,
        },
    )


def _extract_answer(raw: Any) -> Any:
    if isinstance(raw, dict):
        if "decision" in raw:
            return raw
        for key in ("decision", "answer", "output", "result", "response"):
            value = raw.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    continue
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _canonical_evidence_keys(evidence: list[dict[str, Any]]) -> set[str]:
    cited: set[str] = set()
    for item in evidence:
        raw_key = str(item["key"])
        raw_source = str(item.get("source", ""))
        cited.add(raw_key)

        source = raw_source
        if source.startswith("records/"):
            source = source.removeprefix("records/")
        source = source.removesuffix(".json").removesuffix(".md")

        key = raw_key.removeprefix("result.")
        if source:
            cited.add(f"{source}.{key}")

        audit_key = key
        if audit_key.startswith("audit."):
            audit_key = "audit_vendor." + audit_key.removeprefix("audit.")
        audit_key = audit_key.replace(".result.", ".")
        if audit_key.startswith("audit_vendor."):
            cited.add(audit_key)
        if source.startswith("audit_vendor ") and "." in key:
            cited.add("audit_vendor." + key)
    return cited


def _next_step_is_nonempty(answer: dict[str, Any]) -> bool:
    steps = answer.get("required_next_step")
    return isinstance(steps, list) and any(str(step).strip() for step in steps)


def _report(case_id: str, *, passed: bool, score: float, errors: list[str], details: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "passed": passed,
        "score": score,
        "errors": errors,
        "details": details,
    }


if __name__ == "__main__":
    raise SystemExit(main())
