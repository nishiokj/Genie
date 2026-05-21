from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class StageKind(str, Enum):
    DESIGN = "design"
    DESIGN_AUDIT = "design_audit"
    GENERATION = "generation"
    VALIDATION = "validation"
    CURATION = "curation"


class AgentRole(str, Enum):
    DESIGNER = "designer"
    DESIGN_AUDITOR = "design_auditor"
    SAMPLE_GENERATOR = "sample_generator"
    ADVERSARY = "adversary"
    QUALITY_GATE = "quality_gate"
    RUBRIC_GATE = "rubric_gate"


class Verdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"


class RouteCode(str, Enum):
    ACCEPT = "accept"
    REJECT_CRITERIA_MISMATCH = "reject_criteria_mismatch"
    REJECT_SCHEMA = "reject_schema"
    REJECT_LEAKAGE = "reject_leakage"
    REJECT_DUPLICATE = "reject_duplicate"
    REJECT_COVERAGE_MISMATCH = "reject_coverage_mismatch"
    REJECT_SEMANTIC_MISMATCH = "reject_semantic_mismatch"
    REJECT_UPSTREAM_INVARIANT = "reject_upstream_invariant"
    RETRY_INFRA = "retry_infra"
    RETRY_PARSE = "retry_parse"
    RETRY_PROVIDER_EMPTY = "retry_provider_empty"
    DROP_RETRY_EXHAUSTED = "drop_retry_exhausted"


class ContextPolicy(str, Enum):
    FRESH = "fresh"
    SAME_INPUT_RETRY = "same_input_retry"
    CRITERIA_ONLY = "criteria_only"
    CRITERIA_PLUS_ROUTE_CODE = "criteria_plus_route_code"


class TaxonomyCell(BaseModel):
    case_type: str
    difficulty: int = Field(ge=1, le=5)
    scenario: str

    def key(self) -> str:
        return f"{self.case_type}|{self.difficulty}|{self.scenario}"


class EvidenceRef(BaseModel):
    source: str
    path: str
    value: Optional[str] = None


class CheckResult(BaseModel):
    check_id: str
    passed: bool
    route_code: RouteCode = RouteCode.ACCEPT
    subcode: Optional[str] = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class EnvironmentArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentArtifact(BaseModel):
    benchmark_case: dict[str, Any]
    runtime_requirements: Optional[dict[str, Any]] = None
    environment_artifact: Optional[EnvironmentArtifact] = None


class JudgeArtifact(BaseModel):
    score_x: dict[str, Any]
    private_root_cause: str = ""
    expected_fix_properties: list[str] = Field(default_factory=list)
    hidden_failure_modes: list[str] = Field(default_factory=list)
    shallow_solution_traps: list[str] = Field(default_factory=list)
    candidate_visibility_boundaries: list[str] = Field(default_factory=list)
    proxy_claim: str
    diagnostic_pressure: list[str] = Field(default_factory=list)
    scoring_contract: dict[str, Any]
    leakage_risks: list[str] = Field(default_factory=list)
    known_limits: list[str] = Field(default_factory=list)
    coverage_tags: list[str] = Field(default_factory=list)
    negative_controls: list[dict[str, Any]] = Field(default_factory=list)


class DesignBrief(BaseModel):
    id: str
    target_stage: Literal["benchmark"] = "benchmark"
    cell: TaxonomyCell
    target_ability: str
    target_environment: str
    design_intent: str
    environment_premise: dict[str, Any] = Field(default_factory=dict)
    runtime_requirements: dict[str, Any] = Field(default_factory=dict)
    environment_artifact_spec: dict[str, Any] = Field(default_factory=dict)
    failure_mode_family: str
    diagnostic_pressure: list[str] = Field(default_factory=list)
    why_weak_agents_fail: list[str] = Field(default_factory=list)
    tempting_shallow_solutions: list[str] = Field(default_factory=list)
    success_evidence_required: list[str] = Field(default_factory=list)
    minimum_depth_requirements: list[str] = Field(default_factory=list)
    forbidden_shortcuts: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    parent_design_batch_id: Optional[str] = None
    content_hash: str

    @classmethod
    def create(
        cls,
        *,
        design_id: str,
        cell: TaxonomyCell,
        target_ability: str,
        target_environment: str,
        design_intent: str,
        environment_premise: Optional[dict[str, Any]] = None,
        runtime_requirements: Optional[dict[str, Any]] = None,
        environment_artifact_spec: Optional[dict[str, Any]] = None,
        failure_mode_family: str,
        diagnostic_pressure: list[str],
        why_weak_agents_fail: list[str],
        tempting_shallow_solutions: list[str],
        success_evidence_required: list[str],
        minimum_depth_requirements: list[str],
        forbidden_shortcuts: list[str],
        non_goals: list[str],
        parent_design_batch_id: Optional[str] = None,
    ) -> "DesignBrief":
        content_hash = stable_hash(
            {
                "target_stage": "benchmark",
                "cell": cell.model_dump(),
                "target_ability": target_ability,
                "target_environment": target_environment,
                "design_intent": design_intent,
                "environment_premise": environment_premise or {},
                "runtime_requirements": runtime_requirements or {},
                "environment_artifact_spec": environment_artifact_spec or {},
                "failure_mode_family": failure_mode_family,
                "diagnostic_pressure": diagnostic_pressure,
                "why_weak_agents_fail": why_weak_agents_fail,
                "tempting_shallow_solutions": tempting_shallow_solutions,
                "success_evidence_required": success_evidence_required,
                "minimum_depth_requirements": minimum_depth_requirements,
                "forbidden_shortcuts": forbidden_shortcuts,
                "non_goals": non_goals,
                "parent_design_batch_id": parent_design_batch_id,
            }
        )
        return cls(
            id=design_id,
            cell=cell,
            target_ability=target_ability,
            target_environment=target_environment,
            design_intent=design_intent,
            environment_premise=environment_premise or {},
            runtime_requirements=runtime_requirements or {},
            environment_artifact_spec=environment_artifact_spec or {},
            failure_mode_family=failure_mode_family,
            diagnostic_pressure=diagnostic_pressure,
            why_weak_agents_fail=why_weak_agents_fail,
            tempting_shallow_solutions=tempting_shallow_solutions,
            success_evidence_required=success_evidence_required,
            minimum_depth_requirements=minimum_depth_requirements,
            forbidden_shortcuts=forbidden_shortcuts,
            non_goals=non_goals,
            parent_design_batch_id=parent_design_batch_id,
            content_hash=content_hash,
        )


class GenerationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    domain_ref: Optional[str] = None
    design: DesignBrief
    seed_context: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_design(
        cls,
        design: DesignBrief,
        *,
        envelope_id: Optional[str] = None,
        domain_ref: Optional[str] = None,
        seed_context: Optional[dict[str, Any]] = None,
    ) -> "GenerationEnvelope":
        return cls(
            id=envelope_id or design.id,
            domain_ref=domain_ref,
            design=design,
            seed_context=seed_context or {},
        )


class GenerationPipelineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    envelope: GenerationEnvelope
    output_dir: Optional[Path] = None


class GenerationPipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    envelope_id: str
    design_id: str
    final_status: Literal["committed", "dropped", "incomplete"]
    committed: int
    dropped: int
    candidate_id: Optional[str] = None
    route_code: Optional[RouteCode] = None
    subcodes: list[str] = Field(default_factory=list)
    logs_dir: Path
    corpus_path: Path
    result_path: Optional[Path] = None


class DesignVerdict(BaseModel):
    design_id: str
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    rationale: str = ""


class CandidateSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    design_id: str
    content_hash: str
    cell: TaxonomyCell
    output: dict[str, Any] = Field(default_factory=dict)
    agent_artifact: AgentArtifact
    judge_artifact: JudgeArtifact
    ability_z: dict[str, Any]
    environment_y: dict[str, Any]
    difficulty: int = Field(ge=1, le=5)
    case_type: str
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def difficulty_matches_cell(self) -> "CandidateSample":
        if self.difficulty != self.cell.difficulty:
            raise ValueError("difficulty must match taxonomy cell difficulty")
        if self.case_type != self.cell.case_type:
            raise ValueError("case_type must match taxonomy cell case_type")
        if not self.output:
            output = {
                "agent_artifact": self.agent_artifact.model_dump(mode="json", exclude_none=True),
                "judge_artifact": self.judge_artifact.model_dump(mode="json"),
                "ability_z": self.ability_z,
                "environment_y": self.environment_y,
            }
            self.output = output
        return self


class AdversaryReport(BaseModel):
    candidate_id: str
    revision_disposition: Literal["pass", "revise", "nuke"] = "revise"
    disposition_rationale: str = ""
    attack_summary: str = ""
    attacks: list[dict[str, Any]] = Field(default_factory=list)
    cheap_pass_strategy: str = ""
    proxy_damage: str = ""
    survival_requirements: list[str] = Field(default_factory=list)


class SampleVerdict(BaseModel):
    candidate_id: str
    check_kind: Literal["deterministic", "semantic", "quality", "rubric", "curation"]
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    rationale: str = ""


class CertifiedSample(BaseModel):
    id: str
    candidate_id: str
    content_hash: str
    candidate: CandidateSample
    deterministic_checks: list[CheckResult]
    semantic_checks: list[SampleVerdict]


class CommittedSample(BaseModel):
    id: str
    certified_id: str
    content_hash: str
    candidate: CandidateSample
    deterministic_checks: list[CheckResult]
    semantic_checks: list[SampleVerdict]
    embedding_ref: str
    nn_distance: Optional[float]
    taxonomy_cell: TaxonomyCell


class RoutingDecision(BaseModel):
    run_id: str
    from_stage: StageKind
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    next_stage: Optional[StageKind]
    context_policy: ContextPolicy
    retry_index: int
    attempt_of: Optional[str] = None
    terminal: bool = False


class StageRecord(BaseModel):
    run_id: str
    stage_id: str
    role: str
    stage_kind: StageKind
    agent_role: Optional[AgentRole] = None
    parent_artifact_id: Optional[str] = None
    artifact_id: str
    model: str
    provider: str
    prompt_hash: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    reasoning_effort: Optional[str] = None
    text_normalization_replacements: int = 0
    error: Optional[str] = None
    revision_op_count: int = 0
    revision_edit_file_count: int = 0
    revision_files_touched: int = 0
    revision_bytes_added: int = 0
    revision_bytes_removed: int = 0
    revision_bytes_changed: int = 0
    revision_full_rewrite_count: int = 0
    revision_create_file_count: int = 0
    revision_replace_all_count: int = 0
    revision_full_rewrite_ratio: float = 0.0
    verdict: Verdict
    route_code: RouteCode
    subcodes: list[str] = Field(default_factory=list)
    criteria_hash: str
    context_policy: ContextPolicy
    retry_index: int = 0
    attempt_of: Optional[str] = None
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    trace_ref: Optional[str] = None
    wallclock_ts: str = Field(default_factory=utc_now_iso)
