from __future__ import annotations

from models import CandidateSample, CheckResult, CommittedSample, TaxonomyCell
from services.corpus_index import _committed_corpus_record


def test_committed_corpus_record_omits_raw_candidate_output() -> None:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=2, scenario="nominal")
    candidate = CandidateSample(
        id="candidate-1",
        design_id="design-1",
        content_hash="abc",
        cell=cell,
        output={"benchmark_case": {"prompt": "duplicated raw model payload"}},
        agent_artifact={
            "benchmark_case": {
                "prompt": "Normalized committed prompt.",
            },
            "environment_artifact": {
                "kind": "executioner_workspace",
                "payload": {
                    "workspace_root": "/tmp/example-workspace",
                    "logical_root": "/workspace",
                    "files": [{"path": "src/app.py"}, {"path": "tests/test_app.py"}, {"path": "README.md"}],
                    "commands": {"test": "python -m pytest -q"},
                },
            },
        },
        judge_artifact={
            "score_x": {
                "score_type": "rubric",
                "dimensions": [
                    {
                        "name": "quality",
                        "weight": 1.0,
                        "high_score_criterion": "The output satisfies the benchmark's intended behavior.",
                        "low_score_criterion": "The output misses the benchmark's intended behavior.",
                    }
                ],
            },
            "proxy_claim": "This benchmark is a proxy for the named ability because success requires the agent to satisfy the actual task constraints rather than a superficial pattern.",
            "diagnostic_pressure": ["requires reasoning", "punishes shallow compliance"],
            "scoring_contract": {"credit": ["correct behavior"], "penalties": ["shallow behavior"]},
            "leakage_risks": ["visible prompt may overconstrain the answer"],
            "known_limits": ["single case does not prove broad ability"],
            "coverage_tags": ["unit"],
            "negative_controls": [{"output": "bad", "should_fail_because": "It ignores the task."}],
        },
        ability_z={"name": "target_ability"},
        environment_y={"name": "target_environment"},
        difficulty=2,
        case_type="proxy_strong",
    )
    committed = CommittedSample(
        id="committed-1",
        certified_id="certified-1",
        content_hash="hash",
        candidate=candidate,
        deterministic_checks=[CheckResult(check_id="schema", passed=True)],
        semantic_checks=[],
        embedding_ref="embedding-1",
        nn_distance=None,
        taxonomy_cell=cell,
    )

    record = _committed_corpus_record(committed)

    assert "output" not in record["candidate"]
    assert "benchmark_case" not in record["candidate"]
    assert "environment_artifact" not in record["candidate"]
    assert "score_x" not in record["candidate"]
    assert "scoring_contract" not in record["candidate"]
    assert record["candidate"]["agent_artifact"]["benchmark_case"]["prompt"] == "Normalized committed prompt."
    assert record["candidate"]["agent_artifact"]["environment_artifact"]["kind"] == "executioner_workspace"
    assert "version" not in record["candidate"]["agent_artifact"]["environment_artifact"]
    assert record["candidate"]["judge_artifact"]["score_x"]["score_type"] == "rubric"
