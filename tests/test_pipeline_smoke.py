from __future__ import annotations

import json
from contextlib import contextmanager

from config import build_runtime_config
from pipeline import PipelineRunner


class FakeModelClient:
    def __init__(self, config, **kwargs) -> None:
        self.config = config

    @contextmanager
    def stream_context(self, context):
        yield

    def complete_json(self, *, system: str, user: str, schema: dict, temperature: float = 0.4):
        meta = {
            "provider": "test",
            "model": "fake",
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 1,
            "cost_usd": 0.0,
            "prompt_hash": "fake",
        }
        if "Designer" in system:
            return {
                "designs": [
                    {
                        "case_type": "proxy_strong",
                        "difficulty": 3,
                        "scenario": "adversarial",
                        "target_ability": "constrained_poetic_generation",
                        "target_environment": "single_turn_creative_writing",
                        "design_intent": "Generate a haiku benchmark that pressures metaphor and anti-template behavior.",
                        "environment_premise": {"mode": "single turn", "tools": "none"},
                        "failure_mode_family": "template compliance without emotional transfer",
                        "diagnostic_pressure": ["forbid obvious imagery while preserving emotional intent"],
                        "why_weak_agents_fail": ["they produce a format-valid seasonal template"],
                        "tempting_shallow_solutions": ["generic haiku about autumn sadness"],
                        "success_evidence_required": ["indirect metaphor", "constraint adherence"],
                        "minimum_depth_requirements": ["balance form, lexical avoidance, and emotional intent"],
                        "forbidden_shortcuts": ["format-only haiku"],
                        "non_goals": ["broad literary greatness"],
                    }
                ]
            }, meta
        if "Design Auditor" in system:
            return {
                "verdict": "accept",
                "route_code": "accept",
                "subcodes": [],
                "evidence": [],
                "rationale": "The design matches the allowed taxonomy and describes a concrete benchmark pressure.",
            }, meta
        if "REVISION MODE" in system or "RevisionGenerator" in system:
            return {
                "benchmark_case_updates": {
                    "prompt": "Write a haiku that evokes a layoff as late autumn without saying work, loss, leaves, cold, endings, empty, office, or silence."
                },
                "metadata_updates": {
                    "proxy_claim": "A model that succeeds here must preserve emotional transfer while avoiding both the source-domain vocabulary and the easiest seasonal-template vocabulary."
                },
                "environment_ops": [],
                "revision_rationale": "Tighten the lexical trap after the adversary found obvious template paths.",
            }, meta
        if "Benchmark Case Generator" in system:
            return {
                "agent_artifact": {
                    "benchmark_case": {
                        "prompt": "Write a haiku that evokes a layoff as late autumn without saying work, loss, leaves, cold, or endings.",
                        "setup": "Single-turn creative-writing benchmark.",
                        "inputs": {},
                        "environment": {},
                    },
                },
                "judge_artifact": {
                    "score_x": {
                        "score_type": "hard_checks_plus_rubric",
                        "range": [0, 1],
                        "dimensions": [{"name": "constraint_adherence", "weight": 0.4, "high_score_criterion": "Output contains none of the forbidden terms and uses indirect imagery.", "low_score_criterion": "Output uses one or more forbidden terms directly."}],
                    },
                    "private_root_cause": "The hidden target is preserving layoff-related emotional transfer while avoiding direct workplace and seasonal-template vocabulary.",
                    "expected_fix_properties": ["Indirectly evoke the source emotion while respecting all visible lexical constraints."],
                    "hidden_failure_modes": ["Format-valid haiku that uses obvious autumn or workplace vocabulary."],
                    "shallow_solution_traps": ["Generic late-autumn template with no emotional transfer."],
                    "candidate_visibility_boundaries": ["Do not show the source-domain answer or judge-only forbidden template analysis to the evaluated agent."],
                    "proxy_claim": "A model that succeeds here is showing more than haiku formatting because it must preserve emotional intent while avoiding obvious lexical shortcuts and template seasonal imagery.",
                    "diagnostic_pressure": ["forbids obvious imagery", "requires emotional transfer"],
                    "scoring_contract": {
                        "credit": ["preserves emotional intent", "obeys forbidden-word constraints"],
                        "penalties": ["generic seasonal template", "mentions the source domain directly"],
                        "uncertainty_policy": "Mark uncertainty when taste and constraint adherence conflict.",
                    },
                    "leakage_risks": ["A compliant but lifeless template may receive too much credit."],
                    "known_limits": ["The case does not prove broad poetic taste."],
                    "coverage_tags": ["anti_template", "emotional_transfer"],
                    "negative_controls": [{"output": "dead leaves fall at work", "should_fail_because": "uses forbidden imagery and source domain"}],
                },
                "ability_z": {"name": "constrained_poetic_generation", "sub_abilities": ["metaphorical_transfer"]},
                "environment_y": {"name": "single_turn_creative_writing", "assumptions": ["No tools"]},
            }, meta
        return {
            "verdict": "accept",
            "route_code": "accept",
            "subcodes": [],
            "evidence": [],
            "rationale": "The candidate is sufficient for this fake smoke test.",
        }, meta

    def embed(self, text: str):
        return [1.0, 0.0, 0.0], {
            "provider": "test",
            "model": "fake-embedding",
            "input_tokens": 1,
            "output_tokens": 0,
            "latency_ms": 1,
            "cost_usd": 0.0,
            "prompt_hash": "fake-embed",
        }


def test_pipeline_smoke_uses_fenced_fake_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pipeline.ModelClient", FakeModelClient)
    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="smoke",
    )
    config.data_dir = tmp_path / "data"
    config.logs_dir = tmp_path / "logs"

    summary = PipelineRunner(config).run()

    assert summary["committed"] == 1
    corpus_path = tmp_path / "data" / "corpus" / "benchmark" / "smoke.jsonl"
    assert corpus_path.exists()
    assert (tmp_path / "logs" / "smoke" / "stage_records.jsonl").exists()
    stage_io_path = tmp_path / "logs" / "smoke" / "stage_io.jsonl"
    assert stage_io_path.exists()
    envelope_path = tmp_path / "logs" / "smoke" / "generation_envelopes.jsonl"
    assert envelope_path.exists()
    candidates_path = tmp_path / "logs" / "smoke" / "candidates.jsonl"
    assert candidates_path.exists()
    stage_events_path = tmp_path / "logs" / "smoke" / "stage_events.jsonl"
    assert stage_events_path.exists()

    committed = json.loads(corpus_path.read_text(encoding="utf-8").splitlines()[0])
    stage_records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "smoke" / "stage_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    stage_io = [json.loads(line) for line in stage_io_path.read_text(encoding="utf-8").splitlines()]
    envelopes = [json.loads(line) for line in envelope_path.read_text(encoding="utf-8").splitlines()]
    candidate_snapshots = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()]
    stage_events = [json.loads(line) for line in stage_events_path.read_text(encoding="utf-8").splitlines()]
    assert len(stage_io) == len(stage_records)
    assert {"stage": "quality_gate", "stage_event": "provider_start"} in [
        {"stage": event.get("stage"), "stage_event": event.get("stage_event")} for event in stage_events
    ]
    assert {"stage": "rubric_gate", "stage_event": "provider_start"} in [
        {"stage": event.get("stage"), "stage_event": event.get("stage_event")} for event in stage_events
    ]
    assert all("prompt" not in event and "proxy" not in event and "candidate" not in event for event in stage_events)
    assert all(record["trace_ref"] == "stage_io.jsonl" for record in stage_records)
    design_trace = next(trace for trace in stage_io if trace["role"] == "design_batch")
    assert design_trace["output"]["designs"][0]["target_ability"] == "constrained_poetic_generation"
    generation_trace = next(trace for trace in stage_io if trace["role"] == "generate_candidate_sample")
    assert generation_trace["input"]["envelope"]["design"]["id"] == envelopes[0]["envelope"]["design"]["id"]
    assert envelopes[0]["envelope"]["design"]["target_ability"] == "constrained_poetic_generation"
    assert [snapshot["phase"] for snapshot in candidate_snapshots] == ["generated", "adversary_revision"]
    assert candidate_snapshots[1]["parent_candidate_id"] == candidate_snapshots[0]["candidate_id"]
    assert candidate_snapshots[1]["adversary_report_id"] == f"{candidate_snapshots[0]['candidate_id']}-adversary-report"
    assert [check["check_id"] for check in committed["deterministic_checks"]] == [
        "text_hygiene",
        "output_schema",
        "benchmark_case_schema",
        "taxonomy_cell",
        "benchmark_contract",
        "benchmark_oracle",
    ]
    assert "output" not in committed["candidate"]
    assert committed["candidate"]["agent_artifact"]["benchmark_case"]["prompt"]
    assert "score_x" not in committed["candidate"]
    assert committed["candidate"]["judge_artifact"]["score_x"]["score_type"] == "hard_checks_plus_rubric"
    assert [check["check_kind"] for check in committed["semantic_checks"]] == ["quality", "rubric"]
    assert all(check["rationale"] for check in committed["semantic_checks"])
