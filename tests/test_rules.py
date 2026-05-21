from __future__ import annotations

import tempfile
from pathlib import Path

from config import load_domain
from models import CandidateSample, DesignBrief, TaxonomyCell, Verdict
from rules import deterministic_sample_verdict as _deterministic_sample_verdict, validate_design_batch

_TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []

def deterministic_sample_verdict(candidate: CandidateSample, domain):
    return _deterministic_sample_verdict(candidate, domain)


def _code_runtime_requirements(**overrides) -> dict:
    values = {
        "kind": "filesystem_task",
        "execution": {"mode": "task_image", "base_image": "python:3.11-slim", "os": "linux", "arch": "amd64"},
        "language": {"name": "python", "version": "3.11+"},
        "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
        "commands": {"test": "python -m pytest -q"},
        "network": "disabled_during_eval",
    }
    values.update(overrides)
    return values


def _candidate(**overrides) -> CandidateSample:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=3, scenario="adversarial")
    benchmark_case = overrides.pop(
        "benchmark_case",
        {"prompt": "Write a haiku that evokes layoffs as late autumn without saying work, loss, leaves, cold, or endings."},
    )
    runtime_requirements = overrides.pop("runtime_requirements", "__missing__")
    environment_artifact = overrides.pop("environment_artifact", None)
    score_x = overrides.pop(
        "score_x",
        {
            "score_type": "hard_checks_plus_rubric",
            "dimensions": [{"name": "constraint_adherence", "weight": 0.4, "high_score_criterion": "Output contains none of the forbidden terms and uses indirect imagery.", "low_score_criterion": "Output uses one or more forbidden terms directly."}],
        },
    )
    judge_artifact = {
        "score_x": score_x,
        "private_root_cause": overrides.pop("private_root_cause", "The hidden cause belongs in private judge context, not candidate-facing text."),
        "expected_fix_properties": overrides.pop("expected_fix_properties", ["preserves the intended invariant"]),
        "hidden_failure_modes": overrides.pop("hidden_failure_modes", ["a shallow patch fails a hidden edge case"]),
        "shallow_solution_traps": overrides.pop("shallow_solution_traps", ["patching the visible symptom only"]),
        "candidate_visibility_boundaries": overrides.pop(
            "candidate_visibility_boundaries",
            ["candidate-facing material must show symptoms without naming the root cause or fix"],
        ),
        "proxy_claim": overrides.pop("proxy_claim", "A model that succeeds here is showing more than line-count compliance because it must preserve emotional intent while avoiding the obvious vocabulary and imagery that would make a template answer pass superficially."),
        "diagnostic_pressure": overrides.pop("diagnostic_pressure", ["forbids obvious imagery", "requires emotional transfer"]),
        "scoring_contract": overrides.pop(
            "scoring_contract",
            {
                "credit": ["preserves emotional intent", "obeys forbidden-word constraints"],
                "penalties": ["generic seasonal template", "mentions the source domain directly"],
                "uncertainty_policy": "Mark uncertainty when taste and constraint adherence conflict.",
            },
        ),
        "leakage_risks": overrides.pop("leakage_risks", ["A compliant but lifeless template may receive too much credit."]),
        "known_limits": overrides.pop("known_limits", ["The case does not prove broad poetic taste."]),
        "coverage_tags": overrides.pop("coverage_tags", ["anti_template", "emotional_transfer"]),
        "negative_controls": overrides.pop("negative_controls", [{"output": "dead leaves fall at work", "should_fail_because": "uses forbidden imagery and source domain"}]),
    }
    agent_artifact = {"benchmark_case": benchmark_case}
    if runtime_requirements == "__missing__" and environment_artifact is not None:
        runtime_requirements = _code_runtime_requirements()
    if runtime_requirements != "__missing__" and runtime_requirements is not None:
        agent_artifact["runtime_requirements"] = runtime_requirements
    if environment_artifact is not None:
        agent_artifact["environment_artifact"] = environment_artifact
    values = {
        "id": "candidate-1",
        "design_id": "design-1",
        "content_hash": "abc",
        "cell": cell,
        "agent_artifact": agent_artifact,
        "judge_artifact": judge_artifact,
        "ability_z": {"name": "constrained_poetic_generation", "sub_abilities": ["metaphorical_transfer"]},
        "environment_y": {"name": "single_turn_creative_writing", "assumptions": ["No tools"]},
        "difficulty": 3,
        "case_type": "proxy_strong",
    }
    values.update(overrides)
    return CandidateSample(**values)


def _code_design(**overrides) -> DesignBrief:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=4, scenario="edge")
    values = {
        "design_id": "code-design-1",
        "cell": cell,
        "target_ability": "fault_localization",
        "target_environment": "single_turn_debug_with_test",
        "design_intent": "Create a compact benchmark around a realistic stateful debugging failure.",
        "environment_premise": {
            "product_context": "billing reconciliation worker for subscription invoices",
            "codebase_shape": "parser, reconciliation engine, export adapter, and focused tests",
            "state_model": "invoice rows move through parsed, matched, adjusted, and exported states",
            "core_invariant": "recognized revenue totals must remain stable per billing period after adjustments",
            "failure_surface": "monthly summary is wrong only for partial refunds near timezone boundaries",
            "tempting_wrong_fix": "round the exported total or patch the summary formatter",
            "actual_causal_region": "refund normalization before grouping by billing period",
            "required_depth": "requires tracing a transformed value across parser, normalizer, and summarizer",
        },
        "runtime_requirements": _code_runtime_requirements(),
        "failure_mode_family": "misleading downstream aggregate caused by upstream normalization",
        "diagnostic_pressure": ["misleading output error with upstream normalization cause"],
        "why_weak_agents_fail": ["they patch the formatter or rounded total without preserving the period invariant"],
        "tempting_shallow_solutions": ["formatter-only patch passes visible symptom without preserving invariant"],
        "success_evidence_required": ["trace transformed values across parser, normalizer, and summarizer"],
        "minimum_depth_requirements": ["requires at least two modules and a state invariant"],
        "forbidden_shortcuts": ["one-line arithmetic boundary patch", "test edit", "formatter mask"],
        "non_goals": ["typo", "missing import", "one-line loop patch"],
    }
    values.update(overrides)
    return DesignBrief.create(**values)


def _code_benchmark_case(**overrides) -> dict:
    values = {
        "prompt": "Debug the reconciliation worker and provide an invariant-preserving patch with a short explanation.",
        "setup": "You are working in a small Python service. Run the tests after patching.",
        "inputs": {},
        "environment": {"runtime": "python"},
    }
    values.update(overrides)
    return values


def _code_environment_artifact(**payload_overrides) -> dict:
    payload = {
        "files": [
            {
                "path": "billing/parser.py",
                "content": "def parse_row(row):\n    return dict(row)\n",
            },
            {
                "path": "billing/reconcile.py",
                "content": "from billing.parser import parse_row\n\n\ndef summarize(rows):\n    totals = {}\n    for row in rows:\n        parsed = parse_row(row)\n        amount = abs(parsed['amount'])\n        totals[parsed['period']] = totals.get(parsed['period'], 0) + amount\n    return totals\n",
            },
            {
                "path": "tests/test_reconcile.py",
                "content": "from billing.reconcile import summarize\n\n\ndef test_partial_refund_stays_in_original_period():\n    rows = [{'period': '2026-03', 'amount': 100}, {'period': '2026-03', 'amount': -25}]\n    assert summarize(rows) == {'2026-03': 75}\n",
            },
        ],
        "commands": {"test": "python -m pytest -q"},
    }
    payload.update(payload_overrides)
    root = tempfile.TemporaryDirectory(prefix="test-executioner-workspace-")
    workspace_root = Path(root.name)
    for item in payload.get("files", []):
        if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("content"), str):
            target = workspace_root / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item["content"], encoding="utf-8")
    commands = {str(k): str(v) for k, v in payload.get("commands", {}).items()} if isinstance(payload.get("commands"), dict) else {}
    artifact_payload = {
        "session_id": "test-session",
        "logical_root": "/workspace",
        "workspace_root": str(workspace_root),
        "commands": commands,
        "files": [{"path": item["path"]} for item in payload.get("files", []) if isinstance(item, dict) and isinstance(item.get("path"), str)],
    }
    _TEMP_DIRS.append(root)
    return {"kind": "executioner_workspace", "payload": artifact_payload}


def test_benchmark_candidate_passes_deterministic_rules() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    verdict, _ = deterministic_sample_verdict(_candidate(), domain)

    assert verdict.verdict == Verdict.ACCEPT


def test_missing_negative_control_rejected() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    verdict, _ = deterministic_sample_verdict(_candidate(negative_controls=[]), domain)

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["schema_violation"]


def test_output_schema_violation_rejected() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    verdict, checks = deterministic_sample_verdict(_candidate(output={"benchmark_case": {"prompt": "too thin"}}), domain)

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["schema_violation"]
    assert checks[1].check_id == "output_schema"


def test_output_schema_rejects_versioned_environment_artifact() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    environment_artifact = _code_environment_artifact()
    output_environment_artifact = dict(environment_artifact)
    output_environment_artifact["version"] = "vws-1"
    output = {
        "agent_artifact": {
            "benchmark_case": _code_benchmark_case(),
            "environment_artifact": output_environment_artifact,
        },
        "judge_artifact": {
            "score_x": {
                "score_type": "hard_checks_plus_rubric",
                "dimensions": [
                    {
                        "name": "causal_fix",
                        "weight": 1.0,
                        "high_score_criterion": "The solution identifies the upstream cause and preserves invariants.",
                        "low_score_criterion": "The solution only masks the downstream symptom.",
                    }
                ],
            },
            "proxy_claim": "A strong score indicates debugging ability because the candidate must connect a misleading downstream symptom to upstream state normalization while preserving an invariant.",
            "diagnostic_pressure": ["misleading downstream symptom", "upstream state invariant"],
            "scoring_contract": {"credit": ["causal fix preserves invariant"], "penalties": ["formatter-only patch"]},
            "leakage_risks": ["Visible symptom may encourage formatter masking."],
            "known_limits": ["Single case does not prove broad debugging skill."],
            "coverage_tags": ["stateful_debugging"],
            "negative_controls": [{"output": "Round the summary total.", "should_fail_because": "Masks the symptom without fixing the cause."}],
        },
        "ability_z": {"name": "fault_localization"},
        "environment_y": {"name": "single_turn_debug_with_test"},
    }

    verdict, checks = deterministic_sample_verdict(
        _candidate(benchmark_case=_code_benchmark_case(), environment_artifact=environment_artifact, output=output),
        domain,
    )

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["schema_violation"]
    assert checks[1].check_id == "output_schema"


def test_output_schema_rejects_unknown_top_level_fields() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")
    candidate = _candidate()
    output = dict(candidate.output)
    output["score_x"] = {"score_type": "unknown_top_level"}

    verdict, checks = deterministic_sample_verdict(_candidate(output=output), domain)

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["schema_violation"]
    assert checks[1].check_id == "output_schema"


def test_vague_scoring_contract_rejected() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    verdict, _ = deterministic_sample_verdict(_candidate(scoring_contract={"credit": ["good output"]}), domain)

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["schema_violation"]


def test_malformed_control_text_rejected_before_quality_gate() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    verdict, checks = deterministic_sample_verdict(
        _candidate(leakage_risks=["Rare-word stuffing that appears clich\u0000e9 but empty."]),
        domain,
    )

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["malformed_text"]
    assert checks[0].check_id == "text_hygiene"


def test_code_domain_does_not_require_oracle_during_generation() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")

    verdict, checks = deterministic_sample_verdict(
        _candidate(benchmark_case=_code_benchmark_case(), environment_artifact=_code_environment_artifact()),
        domain,
    )

    assert verdict.verdict == Verdict.ACCEPT
    assert checks[-1].check_id == "benchmark_oracle"


def test_code_domain_accepts_candidate_with_oracle() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    benchmark_case = _code_benchmark_case(
        oracle={
            "expected_repair_characteristics": ["producer-side minimal fix", "preserves public API"],
            "hidden_tests": [
                {"name": "edge_one", "input": "case A", "expected": "passes only with causal fix"},
                {"name": "edge_two", "input": "case B", "expected": "rejects shallow guard"},
            ],
            "shallow_fix_failures": [
                {"fix": "consumer-side masking", "fails_because": "does not repair producer invariant"},
                {"fix": "broad try/except", "fails_because": "swallows unrelated failures"},
            ],
        },
    )

    verdict, checks = deterministic_sample_verdict(
        _candidate(benchmark_case=benchmark_case, environment_artifact=_code_environment_artifact()),
        domain,
    )

    assert verdict.verdict == Verdict.ACCEPT
    assert checks[-1].check_id == "benchmark_oracle"


def test_code_domain_requires_executioner_workspace() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")

    verdict, checks = deterministic_sample_verdict(_candidate(), domain)

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["missing_workspace"]
    assert [check.check_id for check in checks] == [
        "text_hygiene",
        "output_schema",
        "benchmark_case_schema",
        "environment_artifact",
        "runtime_requirements",
        "answer_leakage",
        "taxonomy_cell",
        "benchmark_contract",
        "benchmark_oracle",
    ]


def test_code_domain_requires_runtime_requirements_for_executable_workspace() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")

    verdict, checks = deterministic_sample_verdict(
        _candidate(
            benchmark_case=_code_benchmark_case(),
            environment_artifact=_code_environment_artifact(),
            runtime_requirements=None,
        ),
        domain,
    )

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["missing_runtime_requirements"]
    assert checks[4].check_id == "runtime_requirements"


def test_code_domain_rejects_runtime_command_mismatch() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")

    verdict, checks = deterministic_sample_verdict(
        _candidate(
            benchmark_case=_code_benchmark_case(),
            environment_artifact=_code_environment_artifact(),
            runtime_requirements=_code_runtime_requirements(commands={"test": "pytest -q"}),
        ),
        domain,
    )

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["runtime_contract_mismatch"]
    assert checks[4].check_id == "runtime_requirements"


def test_code_domain_rejects_placeholder_workspace_files() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    benchmark_case = _code_benchmark_case()
    environment_artifact = _code_environment_artifact(
        files=[
            {"path": "app.py", "content": "..."},
            {"path": "tests/test_app.py", "content": "def test_app():\n    assert True\n"},
            {"path": "README.md", "content": "Run pytest."},
        ],
        commands={"test": "python -m pytest -q"},
    )

    verdict, _ = deterministic_sample_verdict(_candidate(benchmark_case=benchmark_case, environment_artifact=environment_artifact), domain)

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["placeholder_workspace_file"]


def test_code_domain_rejects_workspace_test_collection_errors() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    benchmark_case = _code_benchmark_case()
    environment_artifact = _code_environment_artifact(
        files=[
            {"path": "token_manager.py", "content": "class TokenManager:\n    pass\n"},
            {"path": "cache.py", "content": "class SimpleCache:\n    pass\n"},
            {"path": "tests/test_import.py", "content": "from missing_module import nope\n\n\ndef test_nope():\n    assert nope\n"},
        ],
        commands={"test": "python -m pytest -q"},
    )

    verdict, checks = deterministic_sample_verdict(_candidate(benchmark_case=benchmark_case, environment_artifact=environment_artifact), domain)

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["workspace_test_command_failed"]
    assert checks[3].check_id == "environment_artifact"


def test_code_domain_rejects_workspace_tests_that_do_not_reproduce_failure() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    benchmark_case = _code_benchmark_case()
    environment_artifact = _code_environment_artifact(
        files=[
            {"path": "billing/parser.py", "content": "def parse_row(row):\n    return dict(row)\n"},
            {
                "path": "billing/reconcile.py",
                "content": "from billing.parser import parse_row\n\n\ndef summarize(rows):\n    totals = {}\n    for row in rows:\n        parsed = parse_row(row)\n        totals[parsed['period']] = totals.get(parsed['period'], 0) + parsed['amount']\n    return totals\n",
            },
            {
                "path": "tests/test_reconcile.py",
                "content": "from billing.reconcile import summarize\n\n\ndef test_partial_refund_stays_in_original_period():\n    rows = [{'period': '2026-03', 'amount': 100}, {'period': '2026-03', 'amount': -25}]\n    assert summarize(rows) == {'2026-03': 75}\n",
            },
        ],
        commands={"test": "python -m pytest -q"},
    )

    verdict, _ = deterministic_sample_verdict(_candidate(benchmark_case=benchmark_case, environment_artifact=environment_artifact), domain)

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["workspace_tests_do_not_reproduce_failure"]


def test_code_domain_rejects_noisy_workspace_failures_across_many_files() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    benchmark_case = _code_benchmark_case()
    environment_artifact = _code_environment_artifact(
        files=[
            {"path": "app.py", "content": "def value():\n    return 1\n"},
            {"path": "tests/test_one.py", "content": "def test_one():\n    assert False\n"},
            {"path": "tests/test_two.py", "content": "def test_two():\n    assert False\n"},
            {"path": "tests/test_three.py", "content": "def test_three():\n    assert False\n"},
        ],
        commands={"test": "python -m pytest -q"},
    )

    verdict, checks = deterministic_sample_verdict(
        _candidate(benchmark_case=benchmark_case, environment_artifact=environment_artifact),
        domain,
    )

    assert verdict.verdict == Verdict.REJECT
    assert verdict.subcodes == ["workspace_test_command_failed"]
    assert checks[3].check_id == "environment_artifact"


def test_code_domain_rejects_candidate_facing_answer_leaks() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    benchmark_case = _code_benchmark_case()
    environment_artifact = _code_environment_artifact(
        files=[
            {"path": "billing/parser.py", "content": "def parse_row(row):\n    return dict(row)\n"},
            {
                "path": "billing/reconcile.py",
                "content": "from billing.parser import parse_row\n\n\ndef summarize(rows):\n    totals = {}\n    for row in rows:\n        parsed = parse_row(row)\n        # BUG: current code incorrectly treats refunds as positive revenue.\n        amount = abs(parsed['amount'])\n        totals[parsed['period']] = totals.get(parsed['period'], 0) + amount\n    return totals\n",
            },
            {
                "path": "tests/test_reconcile.py",
                "content": "from billing.reconcile import summarize\n\n\ndef test_partial_refund_stays_in_original_period():\n    rows = [{'period': '2026-03', 'amount': 100}, {'period': '2026-03', 'amount': -25}]\n    assert summarize(rows) == {'2026-03': 75}\n",
            },
        ],
        commands={"test": "python -m pytest -q"},
    )

    verdict, checks = deterministic_sample_verdict(
        _candidate(benchmark_case=benchmark_case, environment_artifact=environment_artifact),
        domain,
    )

    assert verdict.verdict == Verdict.REJECT
    assert verdict.route_code.value == "reject_leakage"
    assert verdict.subcodes == ["answer_leak_explicit_bug_label"]
    assert checks[5].check_id == "answer_leakage"


def test_code_domain_classifies_candidate_facing_answer_leak_type() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    benchmark_case = _code_benchmark_case()
    environment_artifact = _code_environment_artifact(
        files=[
            {
                "path": "billing/reconcile.py",
                "content": "def summarize(rows):\n    # current code incorrectly treats refunds as positive revenue.\n    return {}\n",
            },
            {"path": "billing/parser.py", "content": "def parse_row(row):\n    return dict(row)\n"},
            {
                "path": "tests/test_reconcile.py",
                "content": "from billing.reconcile import summarize\n\n\ndef test_refund_total():\n    assert summarize([]) == {'total': 0}\n",
            },
        ],
        commands={"test": "python -m pytest -q"},
    )

    verdict, checks = deterministic_sample_verdict(
        _candidate(benchmark_case=benchmark_case, environment_artifact=environment_artifact),
        domain,
    )

    assert verdict.verdict == Verdict.REJECT
    assert verdict.route_code.value == "reject_leakage"
    assert verdict.subcodes == ["answer_leak_fault_behavior_description"]
    assert checks[5].check_id == "answer_leakage"


def test_code_domain_allows_benign_candidate_facing_should_comments() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    benchmark_case = _code_benchmark_case()
    environment_artifact = _code_environment_artifact(
        files=[
            {"path": "billing/parser.py", "content": "# Import adapters should provide mapping-like rows.\n\ndef parse_row(row):\n    return dict(row)\n"},
            {
                "path": "billing/reconcile.py",
                "content": "from billing.parser import parse_row\n\n\ndef summarize(rows):\n    totals = {}\n    for row in rows:\n        parsed = parse_row(row)\n        amount = abs(parsed['amount'])\n        totals[parsed['period']] = totals.get(parsed['period'], 0) + amount\n    return totals\n",
            },
            {
                "path": "tests/test_reconcile.py",
                "content": "from billing.reconcile import summarize\n\n\ndef test_partial_refund_stays_in_original_period():\n    rows = [{'period': '2026-03', 'amount': 100}, {'period': '2026-03', 'amount': -25}]\n    assert summarize(rows) == {'2026-03': 75}\n",
            },
        ],
        commands={"test": "python -m pytest -q"},
    )

    verdict, checks = deterministic_sample_verdict(
        _candidate(benchmark_case=benchmark_case, environment_artifact=environment_artifact),
        domain,
    )

    assert verdict.verdict == Verdict.ACCEPT
    assert checks[5].check_id == "answer_leakage"


def test_code_domain_allows_generic_root_cause_prompt() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    benchmark_case = _code_benchmark_case(
        prompt="Run the failing tests, diagnose the root cause, and implement a minimal patch.",
    )

    verdict, checks = deterministic_sample_verdict(
        _candidate(benchmark_case=benchmark_case, environment_artifact=_code_environment_artifact()),
        domain,
    )

    assert verdict.verdict == Verdict.ACCEPT
    assert checks[5].check_id == "answer_leakage"


def test_code_design_requires_environment_premise_depth() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    design = _code_design(environment_premise={})

    verdict, route_code, subcodes = validate_design_batch([design], domain)

    assert verdict == Verdict.REJECT
    assert route_code.value == "reject_criteria_mismatch"
    assert subcodes == ["weak_diagnostic_pressure"]


def test_code_design_requires_runtime_requirements() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    design = _code_design(runtime_requirements={})

    verdict, route_code, subcodes = validate_design_batch([design], domain)

    assert verdict == Verdict.REJECT
    assert route_code.value == "reject_criteria_mismatch"
    assert subcodes == ["missing_runtime_requirements"]


def test_code_design_rejects_toy_core_blueprint() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    design = _code_design(
        environment_premise={
            "product_context": "small list utility",
            "codebase_shape": "one module and one test file",
            "state_model": "single list input and output",
            "core_invariant": "return all items in order",
            "failure_surface": "one visible test fails",
            "tempting_wrong_fix": "change one loop bound",
            "actual_causal_region": "off-by-one in a single loop",
            "required_depth": "one-line patch",
            "non_goals": ["larger system debugging"],
        }
    )

    verdict, route_code, subcodes = validate_design_batch([design], domain)

    assert verdict == Verdict.REJECT
    assert route_code.value == "reject_criteria_mismatch"
    assert subcodes == ["weak_diagnostic_pressure"]


def test_code_design_accepts_environment_premise_blueprint() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")

    verdict, route_code, subcodes = validate_design_batch([_code_design()], domain)

    assert verdict == Verdict.ACCEPT
    assert route_code.value == "accept"
    assert subcodes == []
