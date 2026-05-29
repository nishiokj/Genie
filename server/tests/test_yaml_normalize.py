"""Regression tests for the draft.yaml normalizer.

Every test is anchored on a real failure pattern we hit during interviews,
plus the inverse "don't break valid documents." Schema validation (against
the pipeline's `config.DomainConfig`) is exercised end-to-end so a future
schema change that breaks our coercions surfaces here.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from yaml_normalize import (
    LIST_STR_FIELDS,
    NormalizeResult,
    field_status,
    normalize,
    normalize_and_validate,
    validate,
)


# ---------------------------------------------------------------------------
# Minimum-valid baseline used as the substrate for repair tests.
# ---------------------------------------------------------------------------

_MINIMAL_VALID = """\
domain_id: probe_domain
case_types: ["proxy_strong"]
difficulties: [1, 2, 3, 4, 5]
scenarios: ["nominal", "edge", "adversarial"]
abilities:
  - "Detect causal relations between latent variables."
  - "Distinguish confounded from non-confounded paths."
environments:
  - "Tabular observational datasets with hidden confounders."
  - "Time-series with intervention markers."
diagnostic_pressure_types:
  - interacting_constraints
  - forbidden_obvious_solution
scoring_methods: ["rubric", "hard_checks_plus_rubric"]
route_codes:
  - accept
  - reject_criteria_mismatch
  - reject_schema
  - retry_infra
  - drop_retry_exhausted
subcodes:
  - accept_complete
  - missing_required_field
  - leakage_detected
  - transient_generation_failure
  - retries_exhausted
novelty_threshold: 0.08
max_design_retries: 2
max_generation_retries: 2
deterministic_rules:
  require_negative_control: true
  min_proxy_claim_chars: 80
  min_diagnostic_pressure_items: 2
  min_leakage_risk_items: 1
  min_known_limit_items: 1
semantic_rules:
  - "Patches must address the underlying causal mechanism, not symptoms."
generator_guidance:
  goal: "Probe ability X in environment Y."
  scoring_contract_bar: "A non-trivial bar."
  proxy_claim_bar: "Argue why score_x discriminates ability_z."
  common_rejection_patterns:
    - "Surface-level keyword matching without mechanism."
quality_gate_rules:
  - "Accept when criteria are concrete and falsifiable."
rubric_gate_rules:
  - "Reject vibes-only scoring contracts."
benchmark_case_schema:
  type: object
  additionalProperties: true
  required: [prompt]
  properties:
    prompt:
      type: string
output_schema_path: "schemas/benchmark_output.schema.json"
"""


def _replace_field(yaml_text: str, field: str, raw_replacement: str) -> str:
    """Swap `<field>: …` (block-scalar value through the next top-level key)
    with the supplied raw YAML block. Caller is responsible for the indentation
    and trailing newline.
    """
    lines = yaml_text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"{field}:"):
            start = i
            break
    if start is None:
        raise AssertionError(f"field {field!r} not found in baseline")
    # Find the next top-level key (column-0, non-blank).
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j] and not lines[j].startswith((" ", "\t", "#")):
            end = j
            break
    return "".join(lines[:start]) + raw_replacement + "".join(lines[end:])


# ---------------------------------------------------------------------------
# Negative space: the baseline parses + validates cleanly with no repairs.
# ---------------------------------------------------------------------------


def test_baseline_is_valid_and_untouched():
    r = normalize_and_validate(_MINIMAL_VALID)
    assert r.parse_error is None
    assert r.validation_error is None, r.validation_error
    assert r.repairs == []
    assert r.ok is True
    assert r.changed is False
    # round-trip preserved bytes
    assert r.yaml_text == _MINIMAL_VALID


# ---------------------------------------------------------------------------
# Pattern 1: unquoted colon in a list[str] item → YAML parses it as a
# single-key mapping. Hit by Nova on semantic_rules in prod.
# ---------------------------------------------------------------------------


def test_colon_trap_in_top_level_list_field_is_repaired():
    bad = _replace_field(
        _MINIMAL_VALID,
        "semantic_rules",
        textwrap.dedent(
            """\
            semantic_rules:
              - "Plain string."
              - Excellent patches are surgical: single-file fixes only.
              - "Another plain string."
            """
        ),
    )
    r = normalize_and_validate(bad)
    assert r.parse_error is None
    assert r.changed is True
    assert any("semantic_rules" in msg for msg in r.repairs)
    parsed = yaml.safe_load(r.yaml_text)
    rules = parsed["semantic_rules"]
    assert all(isinstance(x, str) for x in rules)
    assert any("Excellent patches are surgical" in x for x in rules)
    assert r.validation_error is None, r.validation_error


def test_colon_trap_in_nested_common_rejection_patterns_is_repaired():
    bad = _replace_field(
        _MINIMAL_VALID,
        "generator_guidance",
        textwrap.dedent(
            """\
            generator_guidance:
              goal: "x"
              scoring_contract_bar: "y"
              proxy_claim_bar: "z"
              common_rejection_patterns:
                - "Plain rejection."
                - Bad rejection: with an unquoted colon trap.
            """
        ),
    )
    r = normalize_and_validate(bad)
    assert r.parse_error is None
    assert r.changed is True
    assert any("common_rejection_patterns" in msg for msg in r.repairs)
    parsed = yaml.safe_load(r.yaml_text)
    crp = parsed["generator_guidance"]["common_rejection_patterns"]
    assert all(isinstance(x, str) for x in crp)


# ---------------------------------------------------------------------------
# Pattern 2: whole list[str] field authored as a dict.
# Hit on `subcodes` — model nested subcodes under their route_codes.
# ---------------------------------------------------------------------------


def test_subcodes_authored_as_dict_is_flattened():
    bad = _replace_field(
        _MINIMAL_VALID,
        "subcodes",
        textwrap.dedent(
            """\
            subcodes:
              accept:
                - accept_complete
              reject_criteria_mismatch:
                - ratio_not_met
                - leakage_detected
              reject_schema:
                - missing_required_field
              retry_infra:
                - transient_generation_failure
              drop_retry_exhausted:
                - retries_exhausted
            """
        ),
    )
    r = normalize_and_validate(bad)
    assert r.parse_error is None, r.parse_error
    assert r.changed is True
    assert any("subcodes" in msg for msg in r.repairs)
    parsed = yaml.safe_load(r.yaml_text)
    subs = parsed["subcodes"]
    assert isinstance(subs, list)
    assert all(isinstance(x, str) for x in subs)
    assert "accept_complete" in subs
    assert "leakage_detected" in subs
    assert "retries_exhausted" in subs
    # dedup preserved
    assert len(subs) == len(set(subs))
    assert r.validation_error is None, r.validation_error


def test_dict_flatten_deduplicates_repeated_values():
    bad = _replace_field(
        _MINIMAL_VALID,
        "subcodes",
        textwrap.dedent(
            """\
            subcodes:
              group_a:
                - shared_code
                - unique_a
              group_b:
                - shared_code
                - unique_b
            """
        ),
    )
    r = normalize_and_validate(bad)
    parsed = yaml.safe_load(r.yaml_text)
    assert parsed["subcodes"].count("shared_code") == 1
    assert "unique_a" in parsed["subcodes"]
    assert "unique_b" in parsed["subcodes"]


# ---------------------------------------------------------------------------
# Pattern 3: hard parse error (not a coercion problem) — we surface it.
# ---------------------------------------------------------------------------


def test_unrecoverable_yaml_parse_error_is_surfaced():
    # Mismatched quote — yaml will hard-error here.
    bad = 'domain_id: "unclosed string\nabilities:\n  - foo\n'
    r = normalize_and_validate(bad)
    assert r.parse_error is not None
    assert r.parsed is None
    # We don't try to repair things we don't understand.
    assert r.repairs == []
    assert r.ok is False


def test_top_level_must_be_mapping():
    bad = "- a\n- b\n"
    r = normalize_and_validate(bad)
    assert r.parse_error == "top-level must be a mapping"
    assert r.parsed is None


# ---------------------------------------------------------------------------
# Pattern 4: missing schema fields surface as validation_error
# without us trying to invent values.
# ---------------------------------------------------------------------------


def test_missing_required_field_surfaces_as_validation_error():
    parsed = yaml.safe_load(_MINIMAL_VALID)
    parsed.pop("route_codes")
    bad = yaml.safe_dump(parsed, sort_keys=False)
    r = normalize_and_validate(bad)
    assert r.parse_error is None
    assert r.validation_error is not None
    assert "route_codes" in r.validation_error
    assert r.ok is False


# ---------------------------------------------------------------------------
# field_status — the chip-row signal for the UI. Anchors:
# - all required load-bearing fields must appear in LOAD_BEARING_FIELDS
# - filled / thin / missing classification matches reality
# ---------------------------------------------------------------------------


def test_field_status_marks_complete_baseline_all_filled():
    parsed = yaml.safe_load(_MINIMAL_VALID)
    status = field_status(parsed)
    assert {entry["state"] for entry in status} == {"filled"}


def test_field_status_marks_empty_doc_all_missing():
    status = field_status({})
    assert {entry["state"] for entry in status} == {"missing"}


def test_field_status_classifies_thin_list_as_thin():
    parsed = yaml.safe_load(_MINIMAL_VALID)
    parsed["abilities"] = ["x"]  # single short item → thin
    status = field_status(parsed)
    abilities = next(s for s in status if s["field"] == "abilities")
    assert abilities["state"] == "thin"


def test_field_status_returns_none_input_as_all_missing():
    status = field_status(None)
    assert status, "expected non-empty status list even when input is None"
    assert {entry["state"] for entry in status} == {"missing"}


def test_generator_guidance_thin_when_only_one_nested_key():
    parsed = yaml.safe_load(_MINIMAL_VALID)
    parsed["generator_guidance"] = {"goal": "x"}
    status = field_status(parsed)
    gg = next(s for s in status if s["field"] == "generator_guidance")
    assert gg["state"] == "thin"


# ---------------------------------------------------------------------------
# Misuse / round-trip robustness
# ---------------------------------------------------------------------------


def test_normalize_does_not_mutate_input_string():
    bad = _replace_field(
        _MINIMAL_VALID,
        "semantic_rules",
        "semantic_rules:\n  - Mixing: this is bad.\n",
    )
    original = str(bad)
    normalize_and_validate(bad)
    assert bad == original, "input string was mutated"


def test_list_str_fields_are_documented_in_one_place():
    """If you add a list[str] field to DomainConfig, add it to
    LIST_STR_FIELDS too — otherwise the dict-flatten repair won't fire.
    This test is a tripwire, not a complete schema check.
    """
    expected_minimum = {
        "case_types",
        "scenarios",
        "abilities",
        "environments",
        "diagnostic_pressure_types",
        "route_codes",
        "subcodes",
        "semantic_rules",
    }
    assert expected_minimum.issubset(LIST_STR_FIELDS)


def test_repairs_message_format_is_stable():
    """The interview surfaces repair messages to the UI; keep the prefix
    stable so the UI can group them.
    """
    bad = _replace_field(
        _MINIMAL_VALID,
        "subcodes",
        "subcodes:\n  k1: [x]\n",
    )
    r = normalize_and_validate(bad)
    assert any(msg.startswith("subcodes:") for msg in r.repairs)
