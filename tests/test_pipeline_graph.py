from __future__ import annotations

import json

from langgraph.graph import END

from config import ModelConfig, build_runtime_config
from provider_errors import ProviderError, ProviderStructuredOutputError
from models import (
    AdversaryReport,
    CandidateSample,
    ContextPolicy,
    GenerationEnvelope,
    GenerationPipelineInput,
    RouteCode,
    RoutingDecision,
    SampleVerdict,
    DesignBrief,
    StageKind,
    TaxonomyCell,
    Verdict,
)
from pipeline import (
    PipelineRunner,
    after_adversary,
    after_audit_design,
    after_curate,
    after_generate,
    after_validate_det,
    after_select_next_design,
    after_terminal_design,
    after_gate_join,
    _graph_recursion_limit,
    _provider_error_route_code,
    route_from_decision,
)
from tests.test_pipeline_smoke import FakeModelClient


def test_revision_tool_schema_errors_route_as_parse_retry() -> None:
    route = _provider_error_route_code(
        ProviderError("revision patch environment_ops.0.old_text did not match file")
    )

    assert route == RouteCode.RETRY_PARSE


def _code_design(design_id: str, *, environment_premise: dict | None = None) -> DesignBrief:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=4, scenario="edge")
    return DesignBrief.create(
        design_id=design_id,
        cell=cell,
        target_ability="fault_localization",
        target_environment="single_turn_debug_with_test",
        design_intent="Create a compact benchmark around a realistic stateful debugging failure.",
        environment_premise=environment_premise
        if environment_premise is not None
        else {
            "product_context": "billing reconciliation worker for subscription invoices",
            "codebase_shape": "parser, reconciliation engine, export adapter, and focused tests",
            "state_model": "invoice rows move through parsed, matched, adjusted, and exported states",
            "core_invariant": "recognized revenue totals must remain stable per billing period after adjustments",
            "failure_surface": "monthly summary is wrong only for partial refunds near timezone boundaries",
            "tempting_wrong_fix": "round the exported total or patch the summary formatter",
            "actual_causal_region": "refund normalization before grouping by billing period",
            "required_depth": "requires tracing a transformed value across parser, normalizer, and summarizer",
        },
        runtime_requirements={
            "kind": "filesystem_task",
            "execution": {"mode": "task_image", "base_image": "python:3.11-slim", "os": "linux", "arch": "amd64"},
            "language": {"name": "python", "version": "3.11+"},
            "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
            "commands": {"test": "python -m pytest -q"},
            "network": "disabled_during_eval",
        },
        failure_mode_family="misleading downstream aggregate caused by upstream normalization",
        diagnostic_pressure=["misleading output error with upstream normalization cause"],
        why_weak_agents_fail=["they patch the formatter or rounded total without preserving the period invariant"],
        tempting_shallow_solutions=["formatter-only patch passes visible symptom without preserving invariant"],
        success_evidence_required=["trace transformed values across parser, normalizer, and summarizer"],
        minimum_depth_requirements=["requires at least two modules and a state invariant"],
        forbidden_shortcuts=["one-line arithmetic boundary patch", "test edit", "formatter mask"],
        non_goals=["typo", "missing import", "one-line loop patch"],
    )


def _haiku_design(design_id: str) -> DesignBrief:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=3, scenario="adversarial")
    return DesignBrief.create(
        design_id=design_id,
        cell=cell,
        target_ability="constrained_poetic_generation",
        target_environment="single_turn_creative_writing",
        design_intent="Generate a haiku benchmark that pressures metaphor and anti-template behavior.",
        environment_premise={"mode": "single turn", "tools": "none"},
        failure_mode_family="template compliance without emotional transfer",
        diagnostic_pressure=["forbid obvious imagery while preserving emotional intent"],
        why_weak_agents_fail=["they produce a format-valid seasonal template"],
        tempting_shallow_solutions=["generic haiku about autumn sadness"],
        success_evidence_required=["indirect metaphor", "constraint adherence"],
        minimum_depth_requirements=["balance form, lexical avoidance, and emotional intent"],
        forbidden_shortcuts=["format-only haiku"],
        non_goals=["broad literary greatness"],
    )


def _decision(
    *,
    verdict: Verdict = Verdict.ACCEPT,
    next_stage: StageKind | None = StageKind.GENERATION,
    terminal: bool = False,
) -> RoutingDecision:
    return RoutingDecision(
        run_id="test",
        from_stage=StageKind.DESIGN_AUDIT,
        verdict=verdict,
        route_code=RouteCode.ACCEPT,
        next_stage=next_stage,
        context_policy=ContextPolicy.FRESH,
        retry_index=0,
        terminal=terminal,
    )


def _candidate(design: DesignBrief) -> CandidateSample:
    return CandidateSample(
        id=f"candidate-{design.id}",
        design_id=design.id,
        content_hash="abc",
        cell=design.cell,
        agent_artifact={"benchmark_case": {"prompt": "Debug the reconciliation worker and explain the invariant-preserving fix."}},
        judge_artifact={
            "score_x": {
                "score_type": "hard_checks_plus_rubric",
                "dimensions": [
                    {
                        "name": "causal_fix",
                        "weight": 1.0,
                        "high_score_criterion": "Identifies the upstream normalization cause and preserves period invariants.",
                        "low_score_criterion": "Patches only the displayed summary or visible assertion.",
                    }
                ],
            },
            "private_root_cause": "The hidden causal issue is upstream normalization corrupting the period-level state before summary formatting.",
            "expected_fix_properties": ["Repair normalization while preserving period invariants."],
            "hidden_failure_modes": ["Formatter-only patches should fail stronger invariant checks."],
            "shallow_solution_traps": ["Round or rewrite only the displayed summary total."],
            "candidate_visibility_boundaries": ["Do not reveal the upstream normalization cause in candidate-facing materials."],
            "proxy_claim": "A strong score indicates debugging ability because the candidate must connect a misleading downstream symptom to upstream state normalization while preserving an invariant.",
            "diagnostic_pressure": ["misleading downstream symptom", "upstream state invariant"],
            "scoring_contract": {
                "credit": ["causal fix preserves invariant"],
                "penalties": ["formatter-only patch"],
            },
            "leakage_risks": ["Visible symptom may encourage formatter masking."],
            "known_limits": ["Single case does not prove broad debugging skill."],
            "coverage_tags": ["stateful_debugging"],
            "negative_controls": [{"output": "Round the summary total.", "should_fail_because": "Masks the symptom without fixing the cause."}],
        },
        ability_z={"name": "fault_localization"},
        environment_y={"name": "single_turn_debug_with_test"},
        difficulty=design.cell.difficulty,
        case_type=design.cell.case_type,
    )


def test_compiled_graph_contains_pipeline_nodes(monkeypatch) -> None:
    monkeypatch.setattr("pipeline.ModelClient", FakeModelClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="graph",
    )

    graph = PipelineRunner(config).graph.get_graph()

    assert set(graph.nodes) - {"__start__", "__end__"} == {
        "design",
        "validate_design_batch_det",
        "select_next_design",
        "audit_design",
        "generate",
        "validate_det",
        "adversary",
        "revise_from_adversary",
        "quality_gate",
        "rubric_gate",
        "join_gates",
        "curate",
    }


def test_generation_entrypoint_graph_starts_at_generate(monkeypatch) -> None:
    monkeypatch.setattr("pipeline.ModelClient", FakeModelClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="generation-graph",
    )

    graph = PipelineRunner(config).generation_graph.get_graph()

    assert set(graph.nodes) - {"__start__", "__end__"} == {
        "generate",
        "validate_det",
        "adversary",
        "revise_from_adversary",
        "quality_gate",
        "rubric_gate",
        "join_gates",
        "curate",
    }
    assert "design" not in graph.nodes
    assert "audit_design" not in graph.nodes


def test_graph_recursion_limit_scales_with_run_policy() -> None:
    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=5,
        seed=42,
        run_id="graph-limit",
    )

    assert _graph_recursion_limit(config) > 25


def test_pipeline_runner_uses_role_specific_model_configs(monkeypatch) -> None:
    monkeypatch.setattr("pipeline.ModelClient", FakeModelClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="role-models",
        models=ModelConfig(provider="openai", model="default-model"),
        generator_model=ModelConfig(provider="openai", model="generator-model"),
        adversary_model=ModelConfig(provider="gemini", model="adversary-model"),
        revisor_model=ModelConfig(provider="gemini", model="revisor-model"),
        quality_gate_model=ModelConfig(provider="gemini", model="quality-model"),
    )

    runner = PipelineRunner(config)

    assert runner.generator.client.config.model == "generator-model"
    assert runner.adversary.client.config.model == "adversary-model"
    assert runner.revisor.client.config.model == "revisor-model"
    assert runner.quality_gate.client.config.model == "quality-model"
    assert runner.rubric_gate.client.config.model == "default-model"


def test_design_batch_partition_keeps_valid_designs_from_mixed_batch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.ModelClient", FakeModelClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_code_debug.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="partition",
    )
    config.data_dir = tmp_path / "data"
    config.logs_dir = tmp_path / "logs"
    runner = PipelineRunner(config)
    valid_design = _code_design("valid-design")
    toy_design = _code_design(
        "toy-design",
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
        },
    )

    accepted, rejected = runner._partition_design_batch([toy_design, valid_design])

    assert [design.id for design in accepted] == ["valid-design"]
    assert [(design.id, route_code, subcodes) for design, route_code, subcodes in rejected] == [
        ("toy-design", RouteCode.REJECT_CRITERIA_MISMATCH, ["weak_diagnostic_pressure"])
    ]


def test_route_from_decision_dispatches_by_next_stage() -> None:
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.DESIGN)}) == "design"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.DESIGN_AUDIT)}) == "audit_design"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.GENERATION)}) == "generate"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.VALIDATION)}) == "validate_det"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.CURATION)}) == "curate"
    assert route_from_decision({"last_decision": _decision(next_stage=StageKind.GENERATION, terminal=True)}) == END


def test_after_curate_branches() -> None:
    assert (
        after_curate({"committed_count": 2, "target_n": 2, "designs_queue": [], "design_round": 1, "max_design_retries": 2})
        == END
    )
    assert (
        after_curate(
            {"committed_count": 1, "target_n": 2, "designs_queue": [object()], "design_round": 1, "max_design_retries": 2}
        )
        == "select_next_design"
    )
    assert (
        after_curate({"committed_count": 1, "target_n": 2, "designs_queue": [], "design_round": 1, "max_design_retries": 2})
        == "design"
    )
    assert (
        after_curate({"committed_count": 1, "target_n": 2, "designs_queue": [], "design_round": 3, "max_design_retries": 2})
        == END
    )


def test_after_terminal_design_branches() -> None:
    assert (
        after_terminal_design({"committed_count": 2, "target_n": 2, "designs_queue": [], "design_round": 1, "max_design_retries": 2})
        == END
    )
    assert (
        after_terminal_design(
            {"committed_count": 1, "target_n": 2, "designs_queue": [object()], "design_round": 1, "max_design_retries": 2}
        )
        == "select_next_design"
    )
    assert (
        after_terminal_design({"committed_count": 1, "target_n": 2, "designs_queue": [], "design_round": 1, "max_design_retries": 2})
        == "design"
    )
    assert (
        after_terminal_design({"committed_count": 1, "target_n": 2, "designs_queue": [], "design_round": 3, "max_design_retries": 2})
        == END
    )


def test_after_generate_keeps_design_terminal_from_ending_run() -> None:
    state = {
        "last_decision": _decision(next_stage=None, terminal=True),
        "committed_count": 0,
        "target_n": 2,
        "designs_queue": [object()],
        "design_round": 1,
        "max_design_retries": 2,
    }

    assert after_generate(state) == "select_next_design"


def test_after_select_next_design_branches() -> None:
    assert after_select_next_design({"design": object()}) == "audit_design"
    assert after_select_next_design({"design": None}) == "design"


def test_after_audit_design_branches() -> None:
    assert (
        after_audit_design(
            {"last_decision": _decision(verdict=Verdict.ACCEPT), "designs_queue": [], "design_round": 1, "max_design_retries": 2}
        )
        == "generate"
    )
    assert (
        after_audit_design(
            {
                "last_decision": _decision(verdict=Verdict.REJECT),
                "designs_queue": [object()],
                "design_round": 1,
                "max_design_retries": 2,
            }
        )
        == "select_next_design"
    )
    assert (
        after_audit_design(
            {
                "last_decision": _decision(verdict=Verdict.REJECT),
                "designs_queue": [],
                "design_round": 1,
                "max_design_retries": 2,
            }
        )
        == "design"
    )
    assert (
        after_audit_design(
            {
                "last_decision": _decision(verdict=Verdict.REJECT),
                "designs_queue": [],
                "design_round": 3,
                "max_design_retries": 2,
            }
        )
        == END
    )


def test_after_adversary_revise_still_routes_to_revision() -> None:
    assert after_adversary({"last_decision": None, "adversary_done": False}) == "revise_from_adversary"


def test_adversary_pass_skips_revision_and_routes_to_gates_as_caveats(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.ModelClient", FakeModelClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_code_debug.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="adv-pass",
    )
    config.data_dir = tmp_path / "data"
    config.logs_dir = tmp_path / "logs"
    runner = PipelineRunner(config)
    design = _code_design("design-pass")
    candidate = _candidate(design)
    monkeypatch.setattr(
        runner.adversary,
        "attack",
        lambda candidate, design: (
            AdversaryReport(
                candidate_id=candidate.id,
                revision_disposition="pass",
                disposition_rationale="No blocking attack found.",
            ),
            {"provider": "test", "model": "fake", "prompt_hash": "fake"},
        ),
    )

    update = runner.node_adversary({"design": design, "candidate": candidate, "gen_attempt": 0})

    assert update["adversary_done"] is True
    assert update["last_decision"] is None
    assert after_adversary(update) == ["quality_gate", "rubric_gate"]
    assert after_validate_det({"det_accepted": True, "adversary_done": update["adversary_done"]}) == ["quality_gate", "rubric_gate"]


def test_join_gates_records_rejects_as_caveats_without_rerouting(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.ModelClient", FakeModelClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_code_debug.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="gate-caveat",
    )
    config.data_dir = tmp_path / "data"
    config.logs_dir = tmp_path / "logs"
    runner = PipelineRunner(config)
    design = _code_design("design-caveat")
    candidate = _candidate(design)
    quality = SampleVerdict(
        candidate_id=candidate.id,
        check_kind="quality",
        verdict=Verdict.REJECT,
        route_code=RouteCode.REJECT_SEMANTIC_MISMATCH,
        subcodes=["answer_leak_fix_instruction"],
    )
    rubric = SampleVerdict(
        candidate_id=candidate.id,
        check_kind="rubric",
        verdict=Verdict.ACCEPT,
        route_code=RouteCode.ACCEPT,
    )

    update = runner.node_join_gates(
        {
            "candidate": candidate,
            "quality_verdict": quality,
            "rubric_verdict": rubric,
            "gen_attempt": 0,
        }
    )

    decision = update["last_decision"]
    assert decision.verdict == Verdict.ACCEPT
    assert decision.route_code == RouteCode.ACCEPT
    assert decision.subcodes == ["quality_gate_rejected", "answer_leak_fix_instruction"]
    assert after_gate_join(update) == "curate"


def test_run_from_generation_uses_envelope_and_writes_single_run_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.ModelClient", FakeModelClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="from-generation",
    )
    config.data_dir = tmp_path / "data"
    config.logs_dir = tmp_path / "logs"
    runner = PipelineRunner(config)
    envelope = GenerationEnvelope.from_design(
        _haiku_design("haiku-seed-1"),
        envelope_id="haiku-envelope-1",
        seed_context={"fixture": "static-seed"},
    )

    result = runner.run_from_generation(
        GenerationPipelineInput(
            envelope=envelope,
            output_dir=tmp_path / "entrypoint-output",
        )
    )

    assert result.run_id == "from-generation"
    assert result.envelope_id == "haiku-envelope-1"
    assert result.final_status == "committed"
    assert result.committed == 1
    assert result.dropped == 0
    assert result.candidate_id
    assert result.result_path is not None
    written = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert written["envelope_id"] == "haiku-envelope-1"
    stage_records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "from-generation" / "stage_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    roles = [record["role"] for record in stage_records]
    assert roles[0] == "generate_candidate_sample"
    assert "design_batch" not in roles
    assert "audit_design" not in roles
    assert stage_records[0]["trace_ref"] == "stage_io.jsonl"
    stage_io = [
        json.loads(line)
        for line in (tmp_path / "logs" / "from-generation" / "stage_io.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert stage_io[0]["input"]["envelope"]["id"] == "haiku-envelope-1"
    envelopes = [
        json.loads(line)
        for line in (tmp_path / "logs" / "from-generation" / "generation_envelopes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert envelopes[0]["envelope"]["id"] == "haiku-envelope-1"
    assert envelopes[0]["envelope"]["seed_context"] == {"fixture": "static-seed"}
    candidates = [
        json.loads(line)
        for line in (tmp_path / "logs" / "from-generation" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert candidates[0]["candidate"]["provenance"]["generation_envelope_id"] == "haiku-envelope-1"


def test_quality_gate_invalid_json_records_raw_output_and_routes_parse_retry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.ModelClient", FakeModelClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="quality-invalid-json",
    )
    config.data_dir = tmp_path / "data"
    config.logs_dir = tmp_path / "logs"
    runner = PipelineRunner(config)
    candidate = _candidate(_haiku_design("bad-quality-json"))
    raw_output = '{"verdict": "reject", "rationale": "bad\nnewline"}'

    class BadQualityGate:
        def validate(self, candidate):
            try:
                json.loads(raw_output)
            except json.JSONDecodeError as exc:
                raise ProviderStructuredOutputError(
                    "provider returned invalid structured output: bad newline",
                    raw_content=raw_output,
                    parsing_error=exc,
                )
            raise AssertionError("expected invalid JSON")

    runner.quality_gate = BadQualityGate()
    update = runner.node_quality_gate(
        {
            "candidate": candidate,
            "gen_attempt": 0,
        }
    )

    verdict = update["quality_verdict"]
    assert verdict.verdict == Verdict.REJECT
    assert verdict.route_code == RouteCode.RETRY_PARSE
    assert verdict.subcodes == ["provider_error"]

    traces = [
        json.loads(line)
        for line in (tmp_path / "logs" / "quality-invalid-json" / "stage_io.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    quality_error = next(trace for trace in traces if trace["role"] == "quality_gate_candidate")
    assert quality_error["output"]["raw_provider_output"] == raw_output
    assert quality_error["output"]["parse_error"]["line"] == 1
