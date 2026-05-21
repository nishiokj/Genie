from __future__ import annotations

import unicodedata
from typing import Any

from jsonschema import Draft202012Validator

from config import DomainConfig
from models import (
    CandidateSample,
    CheckResult,
    EvidenceRef,
    RouteCode,
    SampleVerdict,
    Verdict,
)
from services.environment_validation import validate_environment_artifact
from services.execution_workspace import ExecutionWorkspace, ExecutionWorkspaceError
from services.runtime_requirements import validate_supported_container_runtime
from text_hygiene import find_disallowed_text


def validate_design_batch(designs: list[Any], domain: DomainConfig) -> tuple[Verdict, RouteCode, list[str]]:
    seen: set[str] = set()
    for design in designs:
        cell = design.cell
        if cell.case_type not in domain.case_types:
            return Verdict.REJECT, RouteCode.REJECT_COVERAGE_MISMATCH, ["unknown_case_type"]
        if cell.difficulty not in domain.difficulties:
            return Verdict.REJECT, RouteCode.REJECT_COVERAGE_MISMATCH, ["unknown_difficulty"]
        if cell.scenario not in domain.scenarios:
            return Verdict.REJECT, RouteCode.REJECT_COVERAGE_MISMATCH, ["unknown_scenario"]
        if design.content_hash in seen:
            return Verdict.REJECT, RouteCode.REJECT_DUPLICATE, ["duplicate_design"]
        design_environment_verdict = _validate_design_environment(design, domain)
        if design_environment_verdict is not None:
            return design_environment_verdict
        seen.add(design.content_hash)
    return Verdict.ACCEPT, RouteCode.ACCEPT, []


_CODE_ENV_DESIGN_REQUIRED_FIELDS = {
    "product_context",
    "codebase_shape",
    "state_model",
    "core_invariant",
    "failure_surface",
    "tempting_wrong_fix",
    "actual_causal_region",
    "required_depth",
}

_TOY_CODE_DESIGN_MARKERS = {
    "typo",
    "missing import",
    "off-by-one",
    "one-line",
    "single loop",
    "wrong operator",
}


def _validate_design_environment(design: Any, domain: DomainConfig) -> tuple[Verdict, RouteCode, list[str]] | None:
    if domain.domain_id != "benchmark_code_debug":
        return None

    env_design = getattr(design, "environment_premise", {}) or {}
    if not isinstance(env_design, dict):
        return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"]

    runtime = getattr(design, "runtime_requirements", {}) or {}
    if not isinstance(runtime, dict) or not runtime.get("kind"):
        return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["missing_runtime_requirements"]
    if runtime.get("kind") == "filesystem_task":
        execution = runtime.get("execution")
        if not isinstance(execution, dict) or execution.get("mode") not in {"task_image", "container"} or not execution.get("base_image"):
            return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["unsupported_runtime_requirements"]
        commands = runtime.get("commands")
        if not isinstance(commands, dict) or not commands.get("test"):
            return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["missing_runtime_requirements"]

    missing = [field for field in sorted(_CODE_ENV_DESIGN_REQUIRED_FIELDS) if not env_design.get(field)]
    if missing:
        return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"]
    list_fields = (
        "diagnostic_pressure",
        "why_weak_agents_fail",
        "tempting_shallow_solutions",
        "success_evidence_required",
        "minimum_depth_requirements",
        "forbidden_shortcuts",
        "non_goals",
    )
    if any(not getattr(design, field, []) for field in list_fields):
        return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"]

    core_text = " ".join(str(env_design.get(field, "")) for field in _CODE_ENV_DESIGN_REQUIRED_FIELDS).lower()
    if any(marker in core_text for marker in _TOY_CODE_DESIGN_MARKERS):
        return Verdict.REJECT, RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"]

    return None


def deterministic_sample_verdict(
    candidate: CandidateSample,
    domain: DomainConfig,
    *,
    execution_workspace: ExecutionWorkspace | None = None,
) -> tuple[SampleVerdict, list[CheckResult]]:
    checks = [
        _text_hygiene_check(candidate),
        _output_schema_check(candidate, domain),
        _schema_check(candidate, domain),
        _taxonomy_check(candidate, domain),
        _benchmark_contract_check(candidate, domain),
        _benchmark_oracle_check(candidate, domain),
    ]
    if domain.domain_id == "benchmark_code_debug":
        checks.insert(3, _runtime_requirements_check(candidate, domain))
        checks.insert(
            3,
            validate_environment_artifact(
                candidate,
                domain,
                execution_workspace=execution_workspace,
            ),
        )
        checks.insert(5, _candidate_facing_answer_leak_check(candidate, execution_workspace=execution_workspace))
    failed = [check for check in checks if not check.passed]
    if not failed:
        return (
            SampleVerdict(
                candidate_id=candidate.id,
                check_kind="deterministic",
                verdict=Verdict.ACCEPT,
                route_code=RouteCode.ACCEPT,
            ),
            checks,
        )

    first = failed[0]
    return (
        SampleVerdict(
            candidate_id=candidate.id,
            check_kind="deterministic",
            verdict=Verdict.REJECT,
            route_code=first.route_code,
            subcodes=[code for code in [first.subcode] if code],
            evidence=first.evidence,
        ),
        checks,
    )


def _schema_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    validator = Draft202012Validator(domain.benchmark_case_schema)
    errors = sorted(validator.iter_errors(candidate.agent_artifact.benchmark_case), key=lambda err: err.path)
    if errors:
        error = errors[0]
        return CheckResult(
            check_id="benchmark_case_schema",
            passed=False,
            route_code=RouteCode.REJECT_SCHEMA,
            subcode="schema_violation",
            evidence=[EvidenceRef(source="jsonschema", path=".".join(map(str, error.path)), value=error.message)],
        )
    return CheckResult(check_id="benchmark_case_schema", passed=True)


def _runtime_requirements_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    if not bool(domain.deterministic_rules.get("require_runtime_requirements", False)):
        return CheckResult(check_id="runtime_requirements", passed=True)

    runtime = candidate.agent_artifact.runtime_requirements
    if not isinstance(runtime, dict) or not runtime.get("kind"):
        return CheckResult(
            check_id="runtime_requirements",
            passed=False,
            route_code=RouteCode.REJECT_SCHEMA,
            subcode="missing_runtime_requirements",
            evidence=[
                EvidenceRef(
                    source="deterministic_rule",
                    path="agent_artifact.runtime_requirements",
                    value="runtime_requirements.kind is required when a benchmark environment must execute",
                )
            ],
        )

    if runtime.get("kind") != "filesystem_task":
        return CheckResult(check_id="runtime_requirements", passed=True)

    runtime_error = validate_supported_container_runtime(runtime)
    if runtime_error is not None:
        return CheckResult(
            check_id="runtime_requirements",
            passed=False,
            route_code=RouteCode.REJECT_SCHEMA,
            subcode="unsupported_runtime_requirements",
            evidence=[
                EvidenceRef(
                    source="deterministic_rule",
                    path="agent_artifact.runtime_requirements.execution",
                    value=runtime_error,
                )
            ],
        )

    commands = runtime.get("commands")
    if not isinstance(commands, dict) or not isinstance(commands.get("test"), str) or not commands["test"].strip():
        return CheckResult(
            check_id="runtime_requirements",
            passed=False,
            route_code=RouteCode.REJECT_SCHEMA,
            subcode="missing_runtime_requirements",
            evidence=[
                EvidenceRef(
                    source="deterministic_rule",
                    path="agent_artifact.runtime_requirements.commands.test",
                    value="filesystem_task runtime_requirements must declare the visible test command",
                )
            ],
        )

    artifact = candidate.agent_artifact.environment_artifact
    artifact_command = None
    if artifact is not None and artifact.kind == "executioner_workspace":
        payload_commands = artifact.payload.get("commands")
        if isinstance(payload_commands, dict):
            artifact_command = payload_commands.get("test")
    if artifact_command is not None and artifact_command != commands["test"]:
        return CheckResult(
            check_id="runtime_requirements",
            passed=False,
            route_code=RouteCode.REJECT_SCHEMA,
            subcode="runtime_contract_mismatch",
            evidence=[
                EvidenceRef(
                    source="deterministic_rule",
                    path="agent_artifact.environment_artifact.payload.commands.test",
                    value="workspace test command does not match runtime_requirements.commands.test",
                )
            ],
        )

    return CheckResult(check_id="runtime_requirements", passed=True)


def _output_schema_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    if not domain.output_schema:
        return CheckResult(check_id="output_schema", passed=True)
    validator = Draft202012Validator(domain.output_schema)
    errors = sorted(validator.iter_errors(candidate.output), key=lambda err: err.path)
    if errors:
        error = errors[0]
        return CheckResult(
            check_id="output_schema",
            passed=False,
            route_code=RouteCode.REJECT_SCHEMA,
            subcode="schema_violation",
            evidence=[
                EvidenceRef(
                    source="jsonschema",
                    path="output." + ".".join(map(str, error.path)),
                    value=error.message,
                )
            ],
        )
    return CheckResult(check_id="output_schema", passed=True)


def _text_hygiene_check(candidate: CandidateSample) -> CheckResult:
    issue = find_disallowed_text(candidate.model_dump(mode="json"))
    if issue is None:
        return CheckResult(check_id="text_hygiene", passed=True)
    path, char = issue
    codepoint = f"U+{ord(char):04X}"
    name = unicodedata.name(char, "UNKNOWN")
    return CheckResult(
        check_id="text_hygiene",
        passed=False,
        route_code=RouteCode.REJECT_SCHEMA,
        subcode="malformed_text",
        evidence=[EvidenceRef(source="deterministic_rule", path=path, value=f"{codepoint} {name}")],
    )


_ANSWER_LEAK_PATTERNS = (
    ("answer_leak_explicit_bug_label", "bug:"),
    ("answer_leak_explicit_bug_label", "bug note"),
    ("answer_leak_explicit_bug_label", "bug intentionally"),
    ("answer_leak_explicit_bug_label", "intentionally buggy"),
    ("answer_leak_explicit_bug_label", "intentional bug"),
    ("answer_leak_root_cause_disclosure", "root cause:"),
    ("answer_leak_root_cause_disclosure", "source of the bug"),
    ("answer_leak_root_cause_disclosure", "source of this bug"),
    ("answer_leak_fix_instruction", "intended fix"),
    ("answer_leak_fix_instruction", "exact fix"),
    ("answer_leak_fix_instruction", "your fix should"),
    ("answer_leak_fault_location", "faulty line"),
    ("answer_leak_fault_behavior_description", "current code incorrectly"),
    ("answer_leak_fault_behavior_description", "currently incorrectly"),
    ("answer_leak_intended_behavior", "intended invariant"),
    ("answer_leak_intended_behavior", "intended behavior"),
    ("answer_leak_scaffold_label", "starter code"),
    ("answer_leak_scaffold_label", "starter workspace"),
)


def _candidate_facing_answer_leak_check(
    candidate: CandidateSample,
    *,
    execution_workspace: ExecutionWorkspace | None = None,
) -> CheckResult:
    for path, value in _candidate_facing_texts(candidate, execution_workspace=execution_workspace):
        lowered = value.lower()
        for subcode, pattern in _ANSWER_LEAK_PATTERNS:
            if pattern in lowered:
                return CheckResult(
                    check_id="answer_leakage",
                    passed=False,
                    route_code=RouteCode.REJECT_LEAKAGE,
                    subcode=subcode,
                    evidence=[
                        EvidenceRef(
                            source="deterministic_rule",
                            path=path,
                            value=f"candidate-facing text contains answer-leak marker: {pattern}",
                        )
                    ],
                )
    return CheckResult(check_id="answer_leakage", passed=True)


def _candidate_facing_texts(
    candidate: CandidateSample,
    *,
    execution_workspace: ExecutionWorkspace | None = None,
) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    benchmark_case = candidate.agent_artifact.benchmark_case
    for key in ("prompt", "setup", "inputs", "environment"):
        texts.extend(_string_values(f"benchmark_case.{key}", benchmark_case.get(key)))

    artifact = candidate.agent_artifact.environment_artifact
    if artifact is not None and artifact.kind == "executioner_workspace":
        workspace: ExecutionWorkspace | None = None
        close_workspace = False
        try:
            if _workspace_matches_artifact(execution_workspace, artifact.payload):
                workspace = execution_workspace
            else:
                workspace = ExecutionWorkspace.from_artifact(artifact.payload)
                close_workspace = True
            for index, path in enumerate(workspace.list_files()):
                texts.append((f"environment_artifact.payload.files.{index}.path", path))
                texts.append((f"environment_artifact.workspace.{path}", workspace.read_file(path)))
        except ExecutionWorkspaceError:
            pass
        finally:
            if close_workspace and workspace is not None:
                workspace.close()
    return texts


def _workspace_matches_artifact(workspace: ExecutionWorkspace | None, payload: dict[str, Any]) -> bool:
    if workspace is None:
        return False
    root_value = payload.get("workspace_root")
    if not isinstance(root_value, str) or not root_value.strip():
        return False
    try:
        from pathlib import Path

        return workspace.active_root.resolve() == Path(root_value).resolve()
    except OSError:
        return False


def _string_values(path: str, value: Any) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        texts: list[tuple[str, str]] = []
        for key, nested in value.items():
            texts.extend(_string_values(f"{path}.{key}", nested))
        return texts
    if isinstance(value, list):
        texts = []
        for index, nested in enumerate(value):
            texts.extend(_string_values(f"{path}.{index}", nested))
        return texts
    return []


def _taxonomy_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    if candidate.cell.case_type not in domain.case_types:
        return CheckResult(
            check_id="taxonomy_cell",
            passed=False,
            route_code=RouteCode.REJECT_COVERAGE_MISMATCH,
            subcode="unknown_case_type",
        )
    if candidate.cell.scenario not in domain.scenarios:
        return CheckResult(
            check_id="taxonomy_cell",
            passed=False,
            route_code=RouteCode.REJECT_COVERAGE_MISMATCH,
            subcode="unknown_scenario",
        )
    return CheckResult(check_id="taxonomy_cell", passed=True)


def _benchmark_contract_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    judge = candidate.judge_artifact
    if len(judge.proxy_claim.strip()) < int(domain.deterministic_rules.get("min_proxy_claim_chars", 1)):
        return _failed_contract("weak_proxy_validity", "proxy_claim is missing or too short")
    if len(judge.diagnostic_pressure) < int(domain.deterministic_rules.get("min_diagnostic_pressure_items", 1)):
        return _failed_contract("weak_diagnostic_pressure", "diagnostic_pressure has too few items")
    if len(judge.leakage_risks) < int(domain.deterministic_rules.get("min_leakage_risk_items", 0)):
        return _failed_contract("shortcut_leakage", "leakage_risks is missing")
    if len(judge.known_limits) < int(domain.deterministic_rules.get("min_known_limit_items", 0)):
        return _failed_contract("missing_known_limits", "known_limits is missing")
    if bool(domain.deterministic_rules.get("require_negative_control", False)) and not judge.negative_controls:
        return _failed_contract("missing_negative_control", "negative_controls is required")
    if not judge.score_x.get("score_type"):
        return _failed_contract("unreliable_score", "score_x.score_type is missing")
    if not judge.score_x.get("dimensions"):
        return _failed_contract("unreliable_score", "score_x.dimensions is missing")
    for dim in judge.score_x.get("dimensions", []):
        if not dim.get("high_score_criterion") or not dim.get("low_score_criterion"):
            return _failed_contract("vague_scoring_contract", f"dimension '{dim.get('name', '?')}' is missing high_score_criterion or low_score_criterion")
    if not judge.scoring_contract.get("credit") or not judge.scoring_contract.get("penalties"):
        return _failed_contract("vague_scoring_contract", "scoring_contract needs credit and penalties")
    if not candidate.ability_z.get("name"):
        return _failed_contract("weak_proxy_validity", "ability_z.name is missing")
    if not candidate.environment_y.get("name"):
        return _failed_contract("irrelevant_environment", "environment_y.name is missing")
    return CheckResult(check_id="benchmark_contract", passed=True)


def _benchmark_oracle_check(candidate: CandidateSample, domain: DomainConfig) -> CheckResult:
    if not bool(domain.deterministic_rules.get("require_benchmark_oracle", False)):
        return CheckResult(check_id="benchmark_oracle", passed=True)
    oracle = candidate.agent_artifact.benchmark_case.get("oracle")
    if not isinstance(oracle, dict):
        return _failed_oracle("benchmark_case.oracle is required")
    expected = oracle.get("expected_repair_characteristics")
    if not expected:
        return _failed_oracle("oracle.expected_repair_characteristics is required")
    hidden_tests = oracle.get("hidden_tests", [])
    if not isinstance(hidden_tests, list) or len(hidden_tests) < int(domain.deterministic_rules.get("min_hidden_tests", 1)):
        return _failed_oracle("oracle.hidden_tests has too few items")
    shallow_fix_failures = oracle.get("shallow_fix_failures", [])
    if not isinstance(shallow_fix_failures, list) or len(shallow_fix_failures) < int(domain.deterministic_rules.get("min_shallow_fix_failures", 1)):
        return _failed_oracle("oracle.shallow_fix_failures has too few items")
    return CheckResult(check_id="benchmark_oracle", passed=True)


def _failed_contract(subcode: str, value: str) -> CheckResult:
    route = RouteCode.REJECT_LEAKAGE if subcode == "shortcut_leakage" else RouteCode.REJECT_CRITERIA_MISMATCH
    return CheckResult(
        check_id="benchmark_contract",
        passed=False,
        route_code=route,
        subcode=subcode,
        evidence=[EvidenceRef(source="deterministic_rule", path="candidate", value=value)],
    )


def _failed_oracle(value: str) -> CheckResult:
    return CheckResult(
        check_id="benchmark_oracle",
        passed=False,
        route_code=RouteCode.REJECT_CRITERIA_MISMATCH,
        subcode="missing_oracle",
        evidence=[EvidenceRef(source="deterministic_rule", path="benchmark_case.oracle", value=value)],
    )
