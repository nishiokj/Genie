"""Post-parse normalization for Nova-authored draft.yaml.

The model produces *intent*; the schema (`config.DomainConfig`) enforces
*validity*. We sit between them: if YAML parses but the document violates
the schema in known small ways (e.g. an unquoted colon in a list item gets
parsed as a single-key dict instead of a string), we repair it in place and
re-validate. Anything we can't repair becomes a structured error the
interview loop surfaces to the user (and, later, to Nova for self-correct).

The repair set is intentionally narrow — we only normalize patterns we have
seen in real Nova output. Don't reach for generic YAML coercion; let the
schema fail loudly for things we don't recognize.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import yaml


# Fields in DomainConfig that MUST be list[str]. Any dict items appearing here
# are almost always the "unquoted colon → single-key dict" pattern; we flatten
# them back to "key: value" strings.
LIST_STR_FIELDS: set[str] = {
    "case_types",
    "scenarios",
    "abilities",
    "environments",
    "diagnostic_pressure_types",
    "scoring_methods",
    "route_codes",
    "subcodes",
    "semantic_rules",
    "anti_overfit_policy",
    "quality_gate_rules",
    "rubric_gate_rules",
}

# generator_guidance has a nested list[str] under `common_rejection_patterns`.
NESTED_LIST_STR_PATHS: list[tuple[str, ...]] = [
    ("generator_guidance", "common_rejection_patterns"),
]


@dataclass
class NormalizeResult:
    yaml_text: str
    parsed: dict[str, Any] | None
    repairs: list[str]
    parse_error: str | None
    validation_error: str | None

    @property
    def ok(self) -> bool:
        return self.parse_error is None and self.validation_error is None

    @property
    def changed(self) -> bool:
        return bool(self.repairs)


def _flatten_dict_item(item: dict[str, Any]) -> str:
    """`{'Excellent patches are surgical': 'single-file ...'}`
    → `'Excellent patches are surgical: single-file ...'`.
    """
    return ", ".join(f"{k}: {v}" for k, v in item.items())


def _coerce_list_str(items: Any) -> tuple[list[str] | dict, int]:
    """Coerce a `list[str]` field into a flat unique list of strings.

    Handles two real-world miss-patterns:
    - list of dicts (unquoted-colon trap): each dict flattens to "k: v".
    - whole field is a dict (model nested `subcodes` under `route_codes`):
      flatten all values to a unique ordered list of strings.
    """
    if isinstance(items, dict):
        flat: list[str] = []
        seen: set[str] = set()
        for v in items.values():
            if isinstance(v, list):
                for sub in v:
                    s = sub if isinstance(sub, str) else str(sub)
                    if s not in seen:
                        seen.add(s)
                        flat.append(s)
            elif isinstance(v, str):
                if v not in seen:
                    seen.add(v)
                    flat.append(v)
        # also include the dict keys themselves if they're not already there
        # (the original keys are usually meaningful — e.g. route codes — but for
        # subcodes specifically we want the values not the route names; skip
        # keys to avoid polluting). Comment kept to explain the deliberate skip.
        return flat, 1
    if not isinstance(items, list):
        return items, 0
    out: list[str] = []
    changes = 0
    for i in items:
        if isinstance(i, str):
            out.append(i)
        elif isinstance(i, dict):
            out.append(_flatten_dict_item(i))
            changes += 1
        else:
            out.append(str(i))
            changes += 1
    return out, changes


def normalize(yaml_text: str) -> NormalizeResult:
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return NormalizeResult(
            yaml_text=yaml_text,
            parsed=None,
            repairs=[],
            parse_error=str(e),
            validation_error=None,
        )

    if not isinstance(parsed, dict):
        return NormalizeResult(
            yaml_text=yaml_text,
            parsed=None,
            repairs=[],
            parse_error="top-level must be a mapping",
            validation_error=None,
        )

    repairs: list[str] = []
    for field in LIST_STR_FIELDS:
        if field in parsed:
            new_items, n = _coerce_list_str(parsed[field])
            if n:
                parsed[field] = new_items
                repairs.append(f"{field}: coerced {n} non-string item(s) to string")

    for path in NESTED_LIST_STR_PATHS:
        cursor: Any = parsed
        for segment in path[:-1]:
            cursor = cursor.get(segment) if isinstance(cursor, dict) else None
            if cursor is None:
                break
        if isinstance(cursor, dict) and path[-1] in cursor:
            new_items, n = _coerce_list_str(cursor[path[-1]])
            if n:
                cursor[path[-1]] = new_items
                repairs.append(f"{'.'.join(path)}: coerced {n} non-string item(s) to string")

    if repairs:
        buf = io.StringIO()
        yaml.safe_dump(parsed, buf, sort_keys=False, allow_unicode=True, default_flow_style=False)
        yaml_text = buf.getvalue()

    return NormalizeResult(
        yaml_text=yaml_text,
        parsed=parsed,
        repairs=repairs,
        parse_error=None,
        validation_error=None,
    )


def validate(parsed: dict[str, Any]) -> str | None:
    """Run DomainConfig.model_validate. Returns the stringified error on
    failure, None on success. Lazy-import so we don't pull in the pipeline's
    deps at agent-service boot.
    """
    try:
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from config import DomainConfig  # type: ignore
    except Exception as e:  # pragma: no cover
        return f"validator unavailable: {e}"

    try:
        DomainConfig.model_validate(parsed)
    except Exception as e:
        return str(e)
    return None


def normalize_and_validate(yaml_text: str) -> NormalizeResult:
    result = normalize(yaml_text)
    if result.parsed is None:
        return result
    err = validate(result.parsed)
    if err:
        result.validation_error = err
    return result


# ---------- field-fill status (UI signal) ----------

# Load-bearing fields a user actually has to think about. (case_types, scenarios,
# difficulties, route_codes, etc. are templated — not worth surfacing as chips.)
LOAD_BEARING_FIELDS: list[tuple[str, str]] = [
    ("domain_id", "domain_id"),
    ("abilities", "abilities"),
    ("environments", "environments"),
    ("diagnostic_pressure_types", "diagnostic pressure"),
    ("semantic_rules", "semantic rules"),
    ("generator_guidance", "generator guidance"),
    ("benchmark_case_schema", "case schema"),
    ("quality_gate_rules", "quality gate"),
    ("rubric_gate_rules", "rubric gate"),
]


def _classify(field: str, value: Any) -> str:
    """Return 'missing' | 'thin' | 'filled' for a single field."""
    if value is None:
        return "missing"
    if isinstance(value, str):
        return "filled" if value.strip() else "missing"
    if isinstance(value, list):
        if not value:
            return "missing"
        # 'thin' heuristic: 1 item AND that item is shorter than ~20 chars
        if len(value) == 1 and isinstance(value[0], str) and len(value[0].strip()) < 20:
            return "thin"
        return "filled"
    if isinstance(value, dict):
        if not value:
            return "missing"
        # for generator_guidance specifically, want the four nested keys
        if field == "generator_guidance":
            need = {"goal", "scoring_contract_bar", "proxy_claim_bar", "common_rejection_patterns"}
            have = {k for k, v in value.items() if v}
            if not (need & have):
                return "missing"
            if len(need - have) >= 2:
                return "thin"
            crp = value.get("common_rejection_patterns")
            if isinstance(crp, list) and crp and all(isinstance(x, str) for x in crp):
                return "filled"
            return "thin"
        return "filled"
    return "filled"


def field_status(parsed: dict[str, Any] | None) -> list[dict[str, str]]:
    """Returns [{field, label, state}, …] for the UI chip row.
    state ∈ {missing, thin, filled}.
    """
    out: list[dict[str, str]] = []
    if not isinstance(parsed, dict):
        return [{"field": f, "label": label, "state": "missing"} for f, label in LOAD_BEARING_FIELDS]
    for f, label in LOAD_BEARING_FIELDS:
        out.append({"field": f, "label": label, "state": _classify(f, parsed.get(f))})
    return out
