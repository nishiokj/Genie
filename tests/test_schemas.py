from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import CandidateSample, DesignBrief, TaxonomyCell


def test_candidate_round_trip() -> None:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=2, scenario="nominal")
    design = DesignBrief.create(
        design_id="design-1",
        cell=cell,
        target_ability="constrained_poetic_generation",
        target_environment="single_turn_creative_writing",
        design_intent="Generate a constrained haiku benchmark case.",
        environment_premise={"mode": "single turn", "tools": "none"},
        failure_mode_family="template compliance without poetic transfer",
        diagnostic_pressure=["forbid obvious imagery while preserving intent"],
        why_weak_agents_fail=["they satisfy line count while using obvious imagery"],
        tempting_shallow_solutions=["generic autumn haiku template"],
        success_evidence_required=["indirect emotional transfer", "lexical constraint adherence"],
        minimum_depth_requirements=["must balance form, constraint, and metaphor"],
        forbidden_shortcuts=["format-only haiku"],
        non_goals=["broad poetry taste"],
    )
    candidate = CandidateSample(
        id="candidate-1",
        design_id=design.id,
        content_hash="abc",
        cell=cell,
        agent_artifact={"benchmark_case": {"prompt": "Write a haiku about autumn restraint without mentioning leaves."}},
        judge_artifact={
            "score_x": {"score_type": "hard_checks_plus_rubric", "dimensions": [{"name": "constraint_adherence", "weight": 0.5}]},
            "proxy_claim": "Success on this case is a proxy for preserving poetic intent while obeying concrete lexical constraints.",
            "diagnostic_pressure": ["forbids obvious imagery", "requires compact poetic form"],
            "scoring_contract": {"credit": ["obeys constraints"], "penalties": ["generic template"], "uncertainty_policy": "mark uncertainty on taste tradeoffs"},
            "leakage_risks": ["a generic template may satisfy the visible form"],
            "known_limits": ["aesthetic taste remains judge-dependent"],
            "coverage_tags": ["anti_template"],
            "negative_controls": [{"output": "old leaves fall now\ncold winds move through empty trees\nsad autumn is here", "should_fail_because": "uses forbidden obvious imagery"}],
        },
        ability_z={"name": "constrained_poetic_generation", "sub_abilities": ["constraint_following"]},
        environment_y={"name": "single_turn_creative_writing", "assumptions": ["no tools"]},
        difficulty=2,
        case_type="proxy_strong",
    )

    loaded = CandidateSample.model_validate(candidate.model_dump())

    assert loaded.design_id == "design-1"
    assert loaded.cell.key() == "proxy_strong|2|nominal"


def test_environment_artifact_rejects_version_field() -> None:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=2, scenario="nominal")

    with pytest.raises(ValidationError):
        CandidateSample(
            id="candidate-1",
            design_id="design-1",
            content_hash="abc",
            cell=cell,
            agent_artifact={
                "benchmark_case": {"prompt": "Debug the provided execution workspace and explain the fix."},
                "environment_artifact": {"kind": "executioner_workspace", "version": "do-not-accept", "payload": {}},
            },
            judge_artifact={
                "score_x": {"score_type": "rubric", "dimensions": [{"name": "quality", "weight": 1.0}]},
                "proxy_claim": "Success on this case is a proxy because it requires reading the artifact, localizing the defect, and proposing a causal fix rather than guessing from prompt prose.",
                "diagnostic_pressure": ["artifact reading", "causal debugging"],
                "scoring_contract": {"credit": ["causal fix"], "penalties": ["prompt-only guess"]},
                "leakage_risks": ["prompt may overconstrain the fix"],
                "known_limits": ["single case"],
                "coverage_tags": ["artifact"],
                "negative_controls": [{"output": "guess", "should_fail_because": "does not inspect the artifact"}],
            },
            ability_z={"name": "fault_localization"},
            environment_y={"name": "single_turn_debug_with_test"},
            difficulty=2,
            case_type="proxy_strong",
        )
