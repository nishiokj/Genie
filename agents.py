from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any

from agent_constants import (
    ADVERSARY_ATTACK_TYPE_TAXONOMY,
    DESIGN_RETRY_GUIDANCE,
    GENERATOR_IMPLEMENTATION_CONTRACT,
    GENERATOR_PRINCIPLES,
    GENERATOR_RETRY_CODE_MAP,
    GENERATOR_RETRY_GUIDANCE,
    REJECT_SIGNAL_CODES,
)
from config import DomainConfig
from models import (
    AdversaryReport,
    CandidateSample,
    EvidenceRef,
    DesignVerdict,
    GenerationEnvelope,
    RouteCode,
    SampleVerdict,
    DesignBrief,
    TaxonomyCell,
    Verdict,
    stable_hash,
)
from services.execution_workspace import ExecutionWorkspace
from structured_schemas import (
    ADVERSARY_REPORT_SCHEMA as _ADVERSARY_REPORT_SCHEMA,
    DESIGN_BATCH_SCHEMA as _DESIGN_BATCH_SCHEMA,
    REVISION_PATCH_SCHEMA as _REVISION_PATCH_SCHEMA,
    VERDICT_SCHEMA as _VERDICT_SCHEMA,
    generation_output_schema as _generation_output_schema,
)


from model_client import (
    ModelClient,
    _tool_call_arguments,
)
from model_helpers import _nonempty_string
from provider_errors import ProviderError
from generation_artifacts import (
    _apply_revision_patch,
    _candidate_from_generation_payload,
    _example_output_for_domain,
    _execute_workspace_tool,
    _finalize_candidate_tool_parameters,
    _finalize_payload_from_tool_args,
    _generation_payload_from_workspace_final,
    _normalize_tool_call_for_input,
    _revision_patch_shape,
    _test_command_from_design,
    _workspace_tool_final_shape,
    _workspace_tool_schemas,
    revision_patch_metrics,
)


class Designer:
    role_name = "Designer"

    def __init__(self, client: ModelClient, domain: DomainConfig) -> None:
        self.client = client
        self.domain = domain

    def design(
        self,
        *,
        run_id: str,
        target_n: int,
        coverage_snapshot: dict[str, int],
        retry_route_code: RouteCode | None = None,
        retry_subcodes: list[str] | None = None,
    ) -> tuple[list[DesignBrief], dict[str, Any]]:
        system = (
            "You are the Designer for a benchmark-generation pipeline. Produce diagnostic design briefs only. "
            "You separate benchmark design from implementation: define what the case must reveal and the runtime requirements the environment needs, but do not write final prompts, code files, tests, dependency manifests, rubrics, or answer keys. "
            "Return JSON only. A design brief is not a topic label; it must define the target ability, environment premise, failure family, diagnostic pressure, shallow paths, and evidence required for success. "
            "Use exact enum-like strings in runtime_requirements; for filesystem_task execution.mode must be exactly task_image or exactly container, never a combined value. "
            "Reject your own idea before returning it if it naturally becomes a one-line patch, obvious flag/default flip, comparator swap, off-by-one arithmetic repair, cache bypass, synchronous-vs-async toggle, test edit, sleep/timing hack, or direct traceback fix."
        )
        payload: dict[str, Any] = {
            "task": "Create diagnostic design briefs for benchmark cases where score_x should proxy ability_z in environment_y.",
            "target_count": target_n,
            "case_types": self.domain.case_types,
            "difficulties": self.domain.difficulties,
            "scenarios": self.domain.scenarios,
            "abilities": self.domain.abilities,
            "environments": self.domain.environments,
            "diagnostic_pressure_types": self.domain.diagnostic_pressure_types,
            "scoring_methods": self.domain.scoring_methods,
            "coverage_snapshot": coverage_snapshot,
            "design_quality_bar": {
                "must_have": [
                    "a concrete product/environment with state, lifecycle, and invariant-bearing objects",
                    "at least two interacting causes or constraints that make the shallow fix insufficient",
                    "a tempting wrong fix that is plausible and would pass weak visible checks but fail a stronger invariant",
                    "a causal region described as a subsystem or interaction, not the exact line/expression/default to flip",
                    "a design rationale for how the eventual benchmark can produce positive evidence of the target ability beyond table-stakes test passing",
                ],
                "must_not_have": [
                    "off-by-one pagination/binning/slicing as the central repair",
                    "single-line local arithmetic, comparator, typo, missing import, flag/default, or sync/async toggle as the central repair",
                    "an environment design that reveals the exact repair or names the faulty expression",
                    "difficulty that comes from extra files around an obvious local fix",
                    "a scoring idea that mostly means 'run the one visible test and read the explanation'",
                ],
            },
            "required_json_shape": {
                "designs": [
                    {
                        "case_type": "one allowed case type",
                        "difficulty": "integer 1..5",
                        "scenario": "one allowed scenario",
                        "target_ability": "one allowed ability or specific sub-ability",
                        "target_environment": "one allowed environment or environment slice",
                        "design_intent": "specific generation brief, 12-40 words",
                        "environment_premise": {
                            "product_context": "specific realistic software setting",
                            "codebase_shape": "compact description of modules/files/components",
                            "state_model": "objects, data flow, lifecycle, or invariant-bearing state",
                            "core_invariant": "behavior that must remain true",
                            "failure_surface": "observable symptom or user-facing failure",
                            "tempting_wrong_fix": "plausible shallow repair path",
                            "actual_causal_region": "where the real cause should live",
                            "required_depth": "why resolving it requires nontrivial reasoning",
                        },
                        "runtime_requirements": {
                            "kind": "none, text_only, filesystem_task, browser_task, or another explicit runtime class",
                            "execution": {
                                "mode": "exactly one of: task_image, container, none",
                                "base_image": "OCI base image tag or digest when mode is task_image or container",
                                "os": "linux, macos, windows, any, or none",
                                "arch": "amd64, arm64, any, or none",
                            },
                            "language": {
                                "name": "python, typescript, rust, shell, none, or another language/runtime",
                                "version": "runtime version or version range when relevant",
                            },
                            "dependencies": {
                                "policy": "none, stdlib_plus_runner, pinned_manifest, lockfile_required, system_packages, or domain-specific policy",
                                "packages": ["runtime/test dependencies the generated environment must provide"],
                            },
                            "commands": {
                                "test": "canonical visible evaluation command when the environment is executable",
                            },
                            "network": "disabled_during_eval, disabled, allowed, or not_applicable",
                        },
                        "environment_artifact_spec": {
                            "kind": "executioner_workspace when the design needs a bounded artifact environment",
                            "required_capabilities": ["Executioner workspace files needed by the eventual evaluated entity"],
                            "execution_expectation": "how a deterministic environment check should behave, if applicable",
                            "forbidden_environment_shortcuts": ["environment shapes that would make the benchmark too easy or leaky"],
                        },
                        "failure_mode_family": "class of failure the benchmark should instantiate",
                        "diagnostic_pressure": ["specific pressures this case should apply"],
                        "why_weak_agents_fail": ["why weaker evaluated agents should fail"],
                        "tempting_shallow_solutions": ["plausible cheap approaches that should not score well"],
                        "success_evidence_required": ["observable evidence a good implementation must elicit"],
                        "minimum_depth_requirements": ["minimum causal/state/tradeoff depth the implementation must preserve"],
                        "forbidden_shortcuts": ["benchmark shapes the generator must not collapse into"],
                        "non_goals": ["things this case is not trying to test"],
                    }
                ]
            },
        }
        if retry_route_code is not None:
            payload["prior_design_rejection"] = {
                "route_code": retry_route_code.value,
                "subcodes": retry_subcodes or [],
                "retry_guidance": _design_retry_guidance(retry_subcodes or []),
            }
        user = json.dumps(payload, sort_keys=True)
        payload, meta = self.client.complete_json(system=system, user=user, schema=_DESIGN_BATCH_SCHEMA, temperature=0.7)
        designs: list[DesignBrief] = []
        for index, raw in enumerate(payload.get("designs", [])):
            cell = TaxonomyCell(
                case_type=raw["case_type"],
                difficulty=int(raw["difficulty"]),
                scenario=raw["scenario"],
            )
            designs.append(
                DesignBrief.create(
                    design_id=f"{run_id}-design-{index + 1}",
                    cell=cell,
                    target_ability=str(raw["target_ability"]),
                    target_environment=str(raw["target_environment"]),
                    design_intent=str(raw["design_intent"]),
                    environment_premise=raw["environment_premise"] if isinstance(raw.get("environment_premise"), dict) else {},
                    runtime_requirements=raw["runtime_requirements"] if isinstance(raw.get("runtime_requirements"), dict) else {},
                    environment_artifact_spec=raw["environment_artifact_spec"] if isinstance(raw.get("environment_artifact_spec"), dict) else {},
                    failure_mode_family=str(raw["failure_mode_family"]),
                    diagnostic_pressure=_string_list(raw.get("diagnostic_pressure", [])),
                    why_weak_agents_fail=_string_list(raw.get("why_weak_agents_fail", [])),
                    tempting_shallow_solutions=_string_list(raw.get("tempting_shallow_solutions", [])),
                    success_evidence_required=_string_list(raw.get("success_evidence_required", [])),
                    minimum_depth_requirements=_string_list(raw.get("minimum_depth_requirements", [])),
                    forbidden_shortcuts=_string_list(raw.get("forbidden_shortcuts", [])),
                    non_goals=_string_list(raw.get("non_goals", [])),
                    parent_design_batch_id=f"{run_id}-design-batch-1",
                )
            )
        return designs, {**meta, "prompt_hash": stable_hash({"system": system, "user": user})}


class DesignAuditor:
    role_name = "DesignAuditor"

    def __init__(self, client: ModelClient, domain: DomainConfig) -> None:
        self.client = client
        self.domain = domain

    def audit(self, design: DesignBrief) -> tuple[DesignVerdict, dict[str, Any]]:
        system = (
            "You are a stateless Design Auditor. Judge the design against the domain criteria. "
            "Return verdict metadata plus a concise public rationale. "
            "Do not reveal hidden chain-of-thought. Do not rewrite, improve, or repair the design. Return JSON only."
        )
        user = json.dumps(
            {
                "design": design_prompt_view(design),
                "criteria": {
                    "allowed_case_types": self.domain.case_types,
                    "allowed_scenarios": self.domain.scenarios,
                    "allowed_abilities": self.domain.abilities,
                    "allowed_environments": self.domain.environments,
                    "diagnostic_pressure_types": self.domain.diagnostic_pressure_types,
                    "scoring_methods": self.domain.scoring_methods,
                    "difficulty_range": self.domain.difficulties,
                    "code_design_standard": {
                        "required_when_domain": "benchmark_code_debug",
                        "environment_premise_must_define": [
                            "product_context",
                            "codebase_shape",
                            "state_model",
                            "core_invariant",
                            "failure_surface",
                            "tempting_wrong_fix",
                            "actual_causal_region",
                            "required_depth",
                            "runtime_requirements",
                            "non_goals",
                        ],
                        "reject_if": [
                            "the design can naturally instantiate as a one-line patch",
                            "the environment is just a file/function label",
                            "the tempting wrong fix is not plausible",
                            "the failure has no state, invariant, or causal depth",
                            "the actual_causal_region names the exact expression, flag, comparator, default, or one-line repair",
                            "the natural solution is to make an async path synchronous, bypass a cache, edit a test, add a sleep, or change one arithmetic boundary",
                        ],
                    },
                    "route_codes": self.domain.route_codes,
                    "subcodes": self.domain.subcodes,
                },
                "required_json_shape": {
                    "verdict": "accept or reject",
                    "route_code": "accept or a reject route code",
                    "subcodes": ["descriptive labels only"],
                    "evidence": [{"source": "criteria", "path": "field", "value": "short quote"}],
                    "rationale": "2-5 sentence public justification for the verdict, citing concrete evidence; no hidden chain-of-thought",
                },
            },
            sort_keys=True,
        )
        payload, meta = self.client.complete_json(system=system, user=user, schema=_VERDICT_SCHEMA, temperature=0.2)
        verdict = _verdict(payload.get("verdict"))
        route_code = _route_code(payload.get("route_code"), default=RouteCode.ACCEPT if verdict == Verdict.ACCEPT else RouteCode.REJECT_CRITERIA_MISMATCH)
        design_verdict = DesignVerdict(
            design_id=design.id,
            verdict=verdict,
            route_code=route_code,
            subcodes=list(payload.get("subcodes", [])),
            evidence=_evidence(payload.get("evidence", [])),
            rationale=str(payload.get("rationale", "")),
        )
        return design_verdict, {**meta, "prompt_hash": stable_hash({"system": system, "user": user})}


def _format_generator_guidance(domain: DomainConfig) -> str:
    guidance = domain.generator_guidance
    if not guidance:
        return ""
    parts: list[str] = []
    parts.append("\nDOMAIN-SPECIFIC GENERATOR GUIDANCE")
    _section(parts, "Goal", guidance.get("goal"))
    _section(parts, "Scoring contract standard", guidance.get("scoring_contract_bar"))
    _section(parts, "Proxy claim standard", guidance.get("proxy_claim_bar"))
    _section(parts, "Diagnostic pressure in this domain", guidance.get("diagnostic_pressure_notes"))
    return "\n".join(parts)


def _format_gate_guidance(domain: DomainConfig, rules_attr: str) -> str:
    parts: list[str] = []
    rules = getattr(domain, rules_attr)
    if rules:
        parts.append("\nDOMAIN GATE RULES")
        for rule in rules:
            parts.append(f"  - {rule}")
    return "\n".join(parts)


def _format_probe_principles(parts: list[str], principles: dict[str, Any]) -> None:
    if not principles:
        return
    parts.append("\nGENERAL PROBE PRINCIPLES")
    for name, value in principles.items():
        if not isinstance(value, dict):
            parts.append(f"\n{name}:\n{value}")
            continue
        parts.append(f"\n{name}:")
        for key in ("definition", "test_question", "bad_example", "good_example"):
            if value.get(key):
                parts.append(f"  {key}: {str(value[key]).strip()}")
        shortcuts = value.get("shortcuts", [])
        if shortcuts:
            parts.append("  shortcuts:")
            for shortcut in shortcuts:
                parts.append(f"    - {shortcut}")


def _format_anti_overfit_policy(parts: list[str], policy: list[str]) -> None:
    if not policy:
        return
    parts.append("\nANTI-OVERFIT POLICY")
    for item in policy:
        parts.append(f"  - {item}")


def _section(parts: list[str], title: str, body: Any) -> None:
    if body:
        parts.append(f"\n{title}:\n{str(body).strip()}")


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


class SampleGenerator:
    role_name = "SampleGenerator"

    def __init__(
        self,
        client: ModelClient,
        domain: DomainConfig,
        *,
        execution_workspace: ExecutionWorkspace | None = None,
    ) -> None:
        self.client = client
        self.domain = domain
        self.execution_workspace = execution_workspace
        self._system = GENERATOR_PRINCIPLES + _format_generator_guidance(domain) + GENERATOR_IMPLEMENTATION_CONTRACT

    def generate(
        self,
        *,
        run_id: str,
        design: DesignBrief,
        attempt: int,
        retry_route_code: RouteCode | None = None,
        retry_subcodes: list[str] | None = None,
        execution_workspace: ExecutionWorkspace | None = None,
    ) -> tuple[CandidateSample, dict[str, Any]]:
        envelope = GenerationEnvelope.from_design(design)
        return self.generate_from_envelope(
            run_id=run_id,
            envelope=envelope,
            attempt=attempt,
            retry_route_code=retry_route_code,
            retry_subcodes=retry_subcodes,
            execution_workspace=execution_workspace,
        )

    def generate_from_envelope(
        self,
        *,
        run_id: str,
        envelope: GenerationEnvelope,
        attempt: int,
        retry_route_code: RouteCode | None = None,
        retry_subcodes: list[str] | None = None,
        execution_workspace: ExecutionWorkspace | None = None,
    ) -> tuple[CandidateSample, dict[str, Any]]:
        design = envelope.design
        system = self._system
        supports_tools = getattr(self.client, "supports_function_tools", lambda: False)
        if self.domain.domain_id == "benchmark_code_debug" and supports_tools():
            return self._generate_from_envelope_with_workspace_tools(
                run_id=run_id,
                envelope=envelope,
                attempt=attempt,
                retry_route_code=retry_route_code,
                retry_subcodes=retry_subcodes,
                system=system,
                execution_workspace=execution_workspace,
            )
        payload: dict[str, Any] = {
            "generation_envelope": _generation_envelope_prompt_view(envelope),
            "design_brief": design_prompt_view(design),
            "domain": {
                "output_schema": self.domain.output_schema,
                "benchmark_case_schema": self.domain.benchmark_case_schema,
                "abilities": self.domain.abilities,
                "environments": self.domain.environments,
                "diagnostic_pressure_types": self.domain.diagnostic_pressure_types,
                "scoring_methods": self.domain.scoring_methods,
            },
            "required_json_schema": self.domain.output_schema,
            "example_output": _example_output_for_domain(self.domain),
        }
        if retry_route_code is not None:
            safe_subcodes = _generator_safe_retry_subcodes(retry_subcodes or [])
            payload["prior_generation_rejection"] = {
                "route_code": retry_route_code.value,
                "subcodes": safe_subcodes,
                "retry_guidance": _generator_retry_guidance(safe_subcodes),
            }
        user = json.dumps(payload, sort_keys=True)
        payload, meta = self.client.complete_json(
            system=system,
            user=user,
            schema=_generation_output_schema(self.domain),
            temperature=0.8,
        )
        candidate = _candidate_from_generation_payload(
            run_id=run_id,
            envelope=envelope,
            design=design,
            attempt=attempt,
            role_name=self.role_name,
            payload=payload,
        )
        return candidate, {**meta, "prompt_hash": stable_hash({"system": system, "user": user})}

    def _generate_from_envelope_with_workspace_tools(
        self,
        *,
        run_id: str,
        envelope: GenerationEnvelope,
        attempt: int,
        retry_route_code: RouteCode | None,
        retry_subcodes: list[str] | None,
        system: str,
        execution_workspace: ExecutionWorkspace | None = None,
    ) -> tuple[CandidateSample, dict[str, Any]]:
        design = envelope.design
        workspace = execution_workspace or self.execution_workspace or ExecutionWorkspace()
        workspace.reset(_candidate_workspace_subdir(run_id, design, attempt))
        workspace.commands = {"test": _test_command_from_design(design)}
        user_payload: dict[str, Any] = {
            "generation_envelope": _generation_envelope_prompt_view(envelope),
            "design_brief": design_prompt_view(design),
            "domain": {
                "benchmark_case_schema": self.domain.benchmark_case_schema,
                "abilities": self.domain.abilities,
                "environments": self.domain.environments,
                "diagnostic_pressure_types": self.domain.diagnostic_pressure_types,
                "scoring_methods": self.domain.scoring_methods,
            },
            "workspace_authoring": {
                "available_tools": ["write_file", "read_file", "list_files", "finalize_candidate"],
                "rules": [
                    "Author the execution workspace by calling write_file for every file the evaluated agent should receive.",
                    "Use read_file and list_files only to inspect the workspace you have already authored.",
                    "Do not put placeholder content in files.",
                    "Call finalize_candidate only after the workspace contains source files, tests, and README/setup material.",
                    "Call finalize_candidate with structured fields matching finalize_candidate_json_schema. Do not pass JSON strings and do not include file contents; the runner records the Executioner workspace reference.",
                    "The finalize_candidate arguments must include these six top-level keys directly: benchmark_case, runtime_requirements, workspace_commands, judge_artifact, ability_z, environment_y.",
                ],
            },
            "required_final_candidate_shape": _workspace_tool_final_shape(self.domain),
            "finalize_candidate_json_schema": _finalize_candidate_tool_parameters(self.domain),
        }
        if retry_route_code is not None:
            safe_subcodes = _generator_safe_retry_subcodes(retry_subcodes or [])
            user_payload["prior_generation_rejection"] = {
                "route_code": retry_route_code.value,
                "subcodes": safe_subcodes,
                "retry_guidance": _generator_retry_guidance(safe_subcodes),
            }
        user = json.dumps(user_payload, sort_keys=True)
        input_items: list[dict[str, Any]] = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user}],
            }
        ]
        tools = _workspace_tool_schemas(self.domain)
        input_tokens = 0
        output_tokens = 0
        started = time.perf_counter()
        finalized: dict[str, Any] | None = None
        tool_call_count = 0
        emit_stream_event = getattr(self.client, "emit_stream_event", None)
        for step in range(1, 25):
            with self.client.stream_context({"stage_event": "workspace_tool_loop", "tool_loop_step": step}):
                response = self.client.complete_with_tools(system=system, input_items=input_items, tools=tools)
            input_tokens += int(response.usage_metadata.get("input_tokens") or 0)
            output_tokens += int(response.usage_metadata.get("output_tokens") or 0)
            output_items = list(response.output_items or [])
            tool_call_items = [item for item in output_items if item.get("type") in {"function_call", "tool_call", "function_tool_call"}]
            if not output_items:
                raise ProviderError("workspace tool loop returned no output items")
            # Normalize to "function_call" input format — codex emits "function_tool_call" in output
            # but the API only accepts "function_call" items when they appear in the input array.
            input_items.extend(_normalize_tool_call_for_input(item) for item in tool_call_items)
            tool_outputs: list[dict[str, Any]] = []
            for item in tool_call_items:
                tool_call_count += 1
                name = str(item.get("name") or "")
                call_id = _nonempty_string(item.get("call_id")) or _nonempty_string(item.get("id"))
                if not call_id:
                    raise ProviderError(f"workspace tool call {name or '<unknown>'} missing call_id")
                args = _tool_call_arguments(item)
                try:
                    result = _execute_workspace_tool(name, args, workspace, self.domain)
                    if name == "finalize_candidate":
                        finalized = _finalize_payload_from_tool_args(args, self.domain)
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
                if callable(emit_stream_event):
                    emit_stream_event(
                        {
                            "stream_event": "workspace_tool_result",
                            "stage": "model",
                            "tool_loop_step": step,
                            "tool_name": name,
                            "argument_keys": sorted(args.keys()),
                            "ok": bool(result.get("ok")),
                            "error": result.get("error") if not result.get("ok") else None,
                            "workspace_file_count": len(workspace.list_files()),
                        }
                    )
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, sort_keys=True),
                    }
                )
            if finalized is not None:
                break
            if not tool_outputs:
                raise ProviderError("workspace tool loop ended without finalize_candidate")
            input_items.extend(tool_outputs)
        if finalized is None:
            raise ProviderError("workspace tool loop exhausted without finalize_candidate")
        payload = _generation_payload_from_workspace_final(finalized, workspace, design)
        candidate = _candidate_from_generation_payload(
            run_id=run_id,
            envelope=envelope,
            design=design,
            attempt=attempt,
            role_name=self.role_name,
            payload=payload,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        meta = {
            "provider": self.client.config.provider,
            "model": self.client.config.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
            "reasoning_effort": self.client.config.reasoning_effort,
            "structured_output": False,
            "workspace_tool_loop": True,
            "workspace_tool_calls": tool_call_count,
            "workspace_file_count": len(workspace.list_files()),
            "text_normalization_replacements": 0,
        }
        return candidate, {**meta, "prompt_hash": stable_hash({"system": system, "user": user})}

    def revise_from_attack(
        self,
        *,
        run_id: str,
        design: DesignBrief,
        candidate: CandidateSample,
        report: AdversaryReport,
        attempt: int,
        execution_workspace: ExecutionWorkspace | None = None,
    ) -> tuple[CandidateSample, dict[str, Any]]:
        system = (
            "You are RevisionGenerator for a benchmark-generation pipeline. "
            "You revise benchmark artifacts, not solver submissions. "
            "Use the adversary report as falsification evidence about weaknesses in the prior benchmark candidate. "
            "Return a revision patch JSON object only. Do not return a complete candidate. "
            "Preserve the design's intended ability, environment, runtime contract, and benchmark invariants. "
            "For code-debug benchmarks, the unmodified starter workspace must still fail the declared test command before an evaluated agent edits it. "
            "Do not solve the agent-facing bug, do not replace buggy starter implementation with the correct solution, and do not make the starter tests pass. "
            "You may reduce leakage, broaden tests or fixtures, rename misleading candidate-facing material, strengthen metadata, or relocate the intentional defect only when the revised starter still fails for the intended benchmark reason. "
            "Use benchmark_case_updates for benchmark_case replacements, metadata_updates for judge metadata, and environment_ops for Executioner workspace file changes. "
            "environment_ops exposes exactly one tool-shaped operation: edit_file. Prefer exact old_text to new_text replacements. Set create_if_missing=true only for genuinely new files. Set replace_all=true only when every exact occurrence should change. "
            "Keep the patch small and surgical: at most four environment_ops, no full-file rewrites, and short exact text spans only. If the adversary critique would require broad reconstruction, make the smallest metadata or visible-text correction that preserves the starter failure. "
            "Do not include meta-discussion about the adversary or the revision process in candidate-facing materials. Return JSON only."
        )
        user_payload = {
            "design_brief": design_prompt_view(design),
            "prior_candidate": candidate_prompt_view(candidate),
            "adversary_attack_report": adversary_report_prompt_view(report),
            "domain": {
                "output_schema": self.domain.output_schema,
                "benchmark_case_schema": self.domain.benchmark_case_schema,
                "abilities": self.domain.abilities,
                "environments": self.domain.environments,
                "diagnostic_pressure_types": self.domain.diagnostic_pressure_types,
                "scoring_methods": self.domain.scoring_methods,
            },
            "required_revision_patch_shape": _revision_patch_shape(self.domain),
            "benchmark_invariants": [
                "This is benchmark authoring, not task solving.",
                "The revised candidate must preserve the intended evaluated-agent task.",
                "For code-debug workspaces, the unmodified starter code must fail the declared test command for the intended bug before the evaluated agent edits anything.",
                "Do not patch the starter implementation into the reference solution.",
                "Environment file changes are allowed only if they preserve or strengthen the benchmark while keeping the required pre-repair failure.",
            ],
            "revision_patch_rules": [
                "Return only keys from required_revision_patch_shape.",
                "Do not return benchmark_case, environment_artifact, score_x, ability_z, environment_y, or any complete candidate object at the top level.",
                "Do not return runtime_requirements; the seed runtime contract is preserved automatically unless the adversary should have nuked the candidate.",
                "Do not change ability_z or environment_y; if the candidate cannot be rescued within those, the adversary should have nuked it.",
                "Every environment_ops item must use op=edit_file. Do not emit write_file, delete_file, insert_after, remove, shell commands, diffs, or multiple operation names.",
                "Use at most four environment_ops. Each old_text/new_text span must be short and targeted; do not rewrite complete files.",
                "For existing files, old_text must be an exact non-empty string from the current file and new_text must be the replacement. If old_text matches more than once, the patch fails unless replace_all=true.",
                "For new files only, omit old_text and set create_if_missing=true with new_text as the complete file content.",
                "For code-debug candidates, every environment_ops patch must preserve a failing starter workspace; do not solve the bug under test.",
                "Do not include raw output, ids, provenance, route codes, or stage metadata.",
            ],
        }
        user = json.dumps(user_payload, sort_keys=True)
        payload, meta = self.client.complete_json(system=system, user=user, schema=_REVISION_PATCH_SCHEMA, temperature=0.2)
        meta = {**meta, **revision_patch_metrics(payload)}
        revised_candidate_id = f"{run_id}-candidate-{design.id}-{attempt}-rev"
        revised_fields = _apply_revision_patch(
            candidate,
            payload,
            execution_workspace=execution_workspace or self.execution_workspace,
            workspace_subdir=f"candidates/{revised_candidate_id}",
        )
        content = {
            "design_id": design.id,
            "cell": design.cell.model_dump(),
            "revision_patch": payload,
            "agent_artifact": {
                "benchmark_case": revised_fields["benchmark_case"],
                "runtime_requirements": revised_fields["runtime_requirements"],
                "environment_artifact": revised_fields["environment_artifact"],
            },
            "judge_artifact": {
                "score_x": revised_fields["score_x"],
                "private_root_cause": revised_fields["private_root_cause"],
                "expected_fix_properties": list(revised_fields["expected_fix_properties"]),
                "hidden_failure_modes": list(revised_fields["hidden_failure_modes"]),
                "shallow_solution_traps": list(revised_fields["shallow_solution_traps"]),
                "candidate_visibility_boundaries": list(revised_fields["candidate_visibility_boundaries"]),
                "proxy_claim": revised_fields["proxy_claim"],
                "diagnostic_pressure": list(revised_fields["diagnostic_pressure"]),
                "scoring_contract": revised_fields["scoring_contract"],
                "leakage_risks": list(revised_fields["leakage_risks"]),
                "known_limits": list(revised_fields["known_limits"]),
                "coverage_tags": list(revised_fields["coverage_tags"]),
                "negative_controls": list(revised_fields["negative_controls"]),
            },
            "ability_z": revised_fields["ability_z"],
            "environment_y": revised_fields["environment_y"],
            "revision_of": candidate.id,
            "adversary_report": report.model_dump(mode="json"),
        }
        revised = CandidateSample(
            id=revised_candidate_id,
            design_id=design.id,
            content_hash=stable_hash(content),
            cell=design.cell,
            agent_artifact=content["agent_artifact"],
            judge_artifact=content["judge_artifact"],
            ability_z=dict(revised_fields["ability_z"]),
            environment_y=dict(revised_fields["environment_y"]),
            difficulty=design.cell.difficulty,
            case_type=design.cell.case_type,
            provenance={"design_id": design.id, "generator": self.role_name, "revision_of": candidate.id},
        )
        return revised, {**meta, "prompt_hash": stable_hash({"system": system, "user": user})}


def _candidate_workspace_subdir(run_id: str, design: DesignBrief, attempt: int) -> str:
    return f"candidates/{run_id}-candidate-{design.id}-{attempt}"


def design_prompt_view(design: DesignBrief) -> dict[str, Any]:
    """Return the model-facing design brief without transport or lineage metadata."""
    return {
        "target_stage": design.target_stage,
        "cell": design.cell.model_dump(mode="json"),
        "target_ability": design.target_ability,
        "target_environment": design.target_environment,
        "design_intent": design.design_intent,
        "environment_premise": design.environment_premise,
        "runtime_requirements": design.runtime_requirements,
        "environment_artifact_spec": design.environment_artifact_spec,
        "failure_mode_family": design.failure_mode_family,
        "diagnostic_pressure": design.diagnostic_pressure,
        "why_weak_agents_fail": design.why_weak_agents_fail,
        "tempting_shallow_solutions": design.tempting_shallow_solutions,
        "success_evidence_required": design.success_evidence_required,
        "minimum_depth_requirements": design.minimum_depth_requirements,
        "forbidden_shortcuts": design.forbidden_shortcuts,
        "non_goals": design.non_goals,
    }


def _generation_envelope_prompt_view(envelope: GenerationEnvelope) -> dict[str, Any]:
    """Return the generator-facing envelope without transport metadata."""
    return {"design": design_prompt_view(envelope.design)}


def adversary_report_prompt_view(report: AdversaryReport) -> dict[str, Any]:
    """Return adversary findings without candidate identity metadata."""
    return {
        "revision_disposition": report.revision_disposition,
        "disposition_rationale": report.disposition_rationale,
        "attack_summary": report.attack_summary,
        "attacks": report.attacks,
        "cheap_pass_strategy": report.cheap_pass_strategy,
        "proxy_damage": report.proxy_damage,
        "survival_requirements": report.survival_requirements,
    }


def _adversary_report_schema_for_mode(disposition_mode: str) -> dict[str, Any]:
    schema = deepcopy(_ADVERSARY_REPORT_SCHEMA)
    if disposition_mode == "binary":
        properties = schema.get("properties", {})
        disposition = properties.get("revision_disposition")
        if isinstance(disposition, dict):
            disposition["enum"] = ["pass", "revise"]
    return schema


class Adversary:
    role_name = "Adversary"
    base_system_prompt = (
        "You are the Adversary for a benchmark-generation pipeline. "
        "You are not a gate, not a reviewer, not a design compliance checker, and not an improver. Your only job is to attack the benchmark candidate as an independent third party. "
        "The design brief is a claim to attack, not an authority to obey. You may reject the design premise, the implementation, the scoring setup, the proxy claim, or all of them. "
        "Find how the benchmark can be passed cheaply, gamed, leaked, misread, overclaimed, or made meaningless. "
        "Do not rewrite the benchmark and do not offer helpful edits. You may state survival requirements: conditions that would have to become true for your attack to fail. "
        "Be explicit and concrete: name the exact one-line patch, file edit, test edit, flag/default flip, cache bypass, synchronous toggle, sleep/logging hack, or grading loophole a weak solver would use. "
        "If a cheap pass exists, call it out even when the benchmark text says it is forbidden; prose constraints do not stop adversarial solvers. "
        "Decide whether the candidate should pass onward, receive one revision attempt, or be nuked. Choose pass only when you find no blocking attack that would materially damage the proxy claim. Choose revise when the artifact has a substantive core worth preserving but has concrete fixable weaknesses. Choose nuke when the core task is inherently toy-shaped, leaked, fake-hard, or reducible to local patching such that revision would mostly decorate a bad premise. "
        "The prompt separates agent_visible_artifact from evaluator_private_context. Hidden evaluator-private context may reveal intended scoring or proxy claims; never count that private context as answer leakage. "
        "Only label answer_leakage when the leak appears in agent_visible_artifact: candidate-facing prompt, files, comments, tests, fixture names, visible outputs, or setup. "
        "Attack candidate-facing prompt, files, comments, tests, fixture names, visible outputs, scoring criteria, negative controls, proxy claims, and known limits. "
        "Prioritize answer leakage, trivial core repairs, fake difficulty, vague scoring, missing operational checks, overbroad proxy claims, and shallow pass strategies. "
        "Return JSON only. Do not reveal hidden chain-of-thought."
    )
    ternary_disposition_prompt = (
        "Decide whether the candidate should pass onward, receive one revision attempt, or be nuked. Choose pass only when you find no blocking attack that would materially damage the proxy claim. Choose revise when the artifact has a substantive core worth preserving but has concrete fixable weaknesses. Choose nuke when the core task is inherently toy-shaped, leaked, fake-hard, or reducible to local patching such that revision would mostly decorate a bad premise. "
    )
    binary_disposition_prompt = (
        "Decide whether the candidate should pass onward or receive one revision attempt. Choose pass only when you find no blocking attack that would materially damage the proxy claim. Choose revise for every concrete weakness, including severe, leaked, toy-shaped, fake-hard, or local-patching failures; preserve severity and reason codes in attacks instead of using a terminal discard state. "
    )

    def __init__(self, client: ModelClient, domain: DomainConfig, *, disposition_mode: str = "ternary") -> None:
        self.client = client
        self.domain = domain
        if disposition_mode not in {"ternary", "binary"}:
            disposition_mode = "ternary"
        self.disposition_mode = disposition_mode
        disposition_prompt = (
            self.binary_disposition_prompt if disposition_mode == "binary" else self.ternary_disposition_prompt
        )
        self.system_prompt = self.base_system_prompt.replace(
            self.ternary_disposition_prompt,
            disposition_prompt,
        )
        self._schema = _adversary_report_schema_for_mode(disposition_mode)

    def attack(self, candidate: CandidateSample, design: DesignBrief | None = None) -> tuple[AdversaryReport, dict[str, Any]]:
        payload = {
            "design_brief": design_prompt_view(design) if design is not None else None,
            "candidate": candidate_prompt_view(candidate),
            "attack_surface": {
                "domain_id": self.domain.domain_id,
                "general_probe_principles": self.domain.general_probe_principles,
                "anti_overfit_policy": self.domain.anti_overfit_policy,
                "common_rejection_patterns": self.domain.generator_guidance.get("common_rejection_patterns", []),
                "attack_type_taxonomy": ADVERSARY_ATTACK_TYPE_TAXONOMY,
            },
            "required_json_shape": {
                "revision_disposition": "pass or revise"
                if self.disposition_mode == "binary"
                else "pass, revise, or nuke",
                "disposition_rationale": "why this candidate can proceed or deserves revision"
                if self.disposition_mode == "binary"
                else "why this candidate can proceed, deserves revision, or should be discarded instead",
                "attack_summary": "short hostile summary of how this benchmark can be defeated",
                "attacks": [
                    {
                        "attack_target": "design_premise | implementation | scoring | proxy_claim | leakage | other",
                        "attack_type": "one attack_type label from attack_surface.attack_type_taxonomy, or other if none fit",
                        "exploit_path": "how a weak evaluated agent or bad grader can exploit the benchmark",
                        "evidence": "specific candidate-facing field/path/span",
                        "severity": "critical | high | medium | low",
                        "why_it_invalidates_proxy": "why this damages score_x as evidence of ability_z",
                    }
                ],
                "cheap_pass_strategy": "most likely cheap strategy for getting a high score without the target ability",
                "proxy_damage": "how badly these attacks damage the benchmark's proxy claim",
                "survival_requirements": ["conditions that would have to hold for the attack to fail"],
            },
        }
        user = json.dumps(payload, sort_keys=True)
        raw, meta = self.client.complete_json(system=self.system_prompt, user=user, schema=self._schema, temperature=0.2)
        disposition = str(raw.get("revision_disposition", "revise")).lower()
        allowed_dispositions = {"pass", "revise"} if self.disposition_mode == "binary" else {"pass", "revise", "nuke"}
        if disposition not in allowed_dispositions:
            disposition = "revise"
        report = AdversaryReport(
            candidate_id=candidate.id,
            revision_disposition=disposition,
            disposition_rationale=str(raw.get("disposition_rationale", "")),
            attack_summary=str(raw.get("attack_summary", "")),
            attacks=_sanitize_adversary_attacks(list(raw.get("attacks", []))),
            cheap_pass_strategy=str(raw.get("cheap_pass_strategy", "")),
            proxy_damage=str(raw.get("proxy_damage", "")),
            survival_requirements=list(raw.get("survival_requirements", [])),
        )
        return report, {**meta, "prompt_hash": stable_hash({"system": self.system_prompt, "user": user})}


class _GateValidator:
    role_name = "GateValidator"
    check_kind = "semantic"
    system_prompt = ""
    rules_attr = "semantic_rules"

    def __init__(self, client: ModelClient, domain: DomainConfig) -> None:
        self.client = client
        self.domain = domain
        self._system = self.system_prompt + _format_gate_guidance(domain, self.rules_attr)
        self._schema = _verdict_schema_for_domain(domain)

    def _criteria(self, rules: list[str]) -> dict[str, Any]:
        return {
            "gate_rules": rules,
            "route_codes": self.domain.route_codes,
            "subcodes": self.domain.subcodes,
        }

    def _candidate_view(self, candidate: CandidateSample) -> dict[str, Any]:
        return candidate_prompt_view(candidate)

    def validate(self, candidate: CandidateSample) -> tuple[SampleVerdict, dict[str, Any]]:
        system = self._system
        rules = getattr(self.domain, self.rules_attr)
        user = json.dumps(
            {
                "candidate": self._candidate_view(candidate),
                "criteria": self._criteria(rules),
                "required_json_shape": {
                    "verdict": "accept or reject",
                    "route_code": "accept or reject_semantic_mismatch",
                    "subcodes": ["descriptive labels only"],
                    "evidence": [{"source": "candidate", "path": "field", "value": "short span"}],
                    "rationale": "2-5 sentence public justification for the verdict, citing concrete candidate fields; no hidden chain-of-thought",
                },
            },
            sort_keys=True,
        )
        payload, meta = self.client.complete_json(system=system, user=user, schema=self._schema, temperature=0.2)
        verdict = _verdict(payload.get("verdict"))
        subcodes = list(payload.get("subcodes", []))
        evidence = _evidence(payload.get("evidence", []))
        subcodes = _sanitize_gate_leakage_subcodes(subcodes, evidence)
        route_code = _route_code(
            payload.get("route_code"),
            default=RouteCode.ACCEPT if verdict == Verdict.ACCEPT else RouteCode.REJECT_SEMANTIC_MISMATCH,
        )
        if route_code == RouteCode.REJECT_LEAKAGE and not any(_is_answer_leak_subcode(code) for code in subcodes):
            route_code = RouteCode.REJECT_SEMANTIC_MISMATCH
        verdict, route_code, subcodes = _coerce_gate_verdict(
            verdict=verdict,
            route_code=route_code,
            subcodes=subcodes,
        )
        sample_verdict = SampleVerdict(
            candidate_id=candidate.id,
            check_kind=self.check_kind,
            verdict=verdict,
            route_code=route_code,
            subcodes=subcodes,
            evidence=evidence,
            rationale=str(payload.get("rationale", "")),
        )
        return sample_verdict, {**meta, "prompt_hash": stable_hash({"system": system, "user": user})}


def candidate_prompt_view(candidate: CandidateSample, *, include_evaluator_private: bool = True) -> dict[str, Any]:
    """Return the model-facing benchmark artifact without pipeline bookkeeping.

    Agent-visible material is structurally separated from evaluator-private
    context so judges cannot confuse hidden scoring/rubric context for leaked
    benchmark content.
    """
    view: dict[str, Any] = {
        "agent_visible_artifact": candidate.agent_artifact.model_dump(mode="json", exclude_none=True),
        "benchmark_context": {
            "ability_z": candidate.ability_z,
            "environment_y": candidate.environment_y,
            "difficulty": candidate.difficulty,
            "case_type": candidate.case_type,
            "cell": candidate.cell.model_dump(mode="json"),
        },
    }
    if include_evaluator_private:
        view["evaluator_private_context"] = {
            "judge_artifact": candidate.judge_artifact.model_dump(mode="json"),
        }
    return view


def candidate_quality_prompt_view(candidate: CandidateSample) -> dict[str, Any]:
    """Return only the artifact and target context QualityGate may judge."""
    return candidate_prompt_view(candidate, include_evaluator_private=False)


def _sanitize_gate_leakage_subcodes(subcodes: list[str], evidence: list[EvidenceRef]) -> list[str]:
    if not any(_is_answer_leak_subcode(code) for code in subcodes):
        return subcodes
    if any(_evidence_points_to_agent_visible_material(item) for item in evidence):
        return subcodes
    return [code for code in subcodes if not _is_answer_leak_subcode(code)]


def _is_answer_leak_subcode(code: str) -> bool:
    return code.startswith("answer_leak_")


def _verdict_schema_for_domain(domain: DomainConfig) -> dict[str, Any]:
    schema = deepcopy(_VERDICT_SCHEMA)
    subcodes = schema.get("properties", {}).get("subcodes")
    if isinstance(subcodes, dict):
        items = subcodes.setdefault("items", {})
        if isinstance(items, dict):
            items["enum"] = list(domain.subcodes)
    route_code = schema.get("properties", {}).get("route_code")
    if isinstance(route_code, dict):
        route_code["enum"] = list(domain.route_codes)
    return schema


def _evidence_points_to_agent_visible_material(evidence: EvidenceRef) -> bool:
    source = evidence.source.strip().lower()
    path = evidence.path.strip().lower()
    if "evaluator_private_context" in path or "judge_artifact" in path:
        return False
    if source in {"evaluator_private_context", "judge_artifact", "rubric", "private"}:
        return False
    return (
        source in {"candidate", "agent_visible_artifact", "agent_artifact", "visible"}
        or path.startswith("agent_visible_artifact.")
        or path.startswith("agent_artifact.")
        or path.startswith("benchmark_case.")
        or path.startswith("environment_artifact.")
    )


def _sanitize_adversary_attacks(attacks: list[Any]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for attack in attacks:
        if not isinstance(attack, dict):
            continue
        item = dict(attack)
        if str(item.get("attack_type") or "").strip().lower() == "answer_leakage" and not _adversary_attack_points_to_visible_leak(item):
            item["attack_type"] = _non_leakage_attack_type(item)
        sanitized.append(item)
    return sanitized


def _adversary_attack_points_to_visible_leak(attack: dict[str, Any]) -> bool:
    evidence = str(attack.get("evidence") or "").lower()
    exploit = str(attack.get("exploit_path") or "").lower()
    text = f"{evidence} {exploit}"
    private_markers = {
        "evaluator_private_context",
        "judge_artifact",
        "scoring_contract",
        "score_x",
        "proxy_claim",
        "negative_controls",
        "known_limits",
        "leakage_risks",
    }
    if any(marker in text for marker in private_markers):
        return False
    visible_markers = {
        "agent_visible_artifact",
        "agent_artifact",
        "benchmark_case",
        "environment_artifact",
        "readme",
        "prompt",
        "setup",
        "source comment",
        "comment",
        "test name",
        "fixture",
        "visible output",
        "visible test",
        "tests/",
        ".py",
        ".ts",
        ".js",
        ".java",
        ".go",
        ".rs",
    }
    return any(marker in text for marker in visible_markers)


def _non_leakage_attack_type(attack: dict[str, Any]) -> str:
    text = " ".join(str(attack.get(key) or "").lower() for key in ("attack_target", "evidence", "exploit_path"))
    if "scoring" in text or "score" in text or "rubric" in text:
        return "scoring_ambiguity"
    if "proxy" in text:
        return "proxy_overclaim"
    return "other"


class QualityGate(_GateValidator):
    role_name = "QualityGate"
    check_kind = "quality"
    rules_attr = "quality_gate_rules"
    system_prompt = (
        "You are QualityGate for a benchmark-generation pipeline. "
        "Judge benchmark quality independently of the rubric's self-description. "
        "Ask whether the concrete candidate artifact is worth grading as a proxy for ability_z in environment_y. "
        "Accept only when the artifact itself creates meaningful diagnostic pressure and would make success evidence of the claimed ability. "
        "Reject if the task is toy-shaped, leaked, fake-hard, too local, mostly compliance theater, or passable by pattern matching, visible clues, fixture-specific edits, or generic defensive guards. "
        "You only receive the evaluated agent's visible artifact and target benchmark context; do not infer hidden rubric validity or hidden answer keys. "
        "Only use answer_leak_* subcodes when the leak appears in agent_visible_artifact: prompt, setup, README, source comments, test names, fixture names, visible test assertions, or visible outputs. Prefer the most specific answer_leak_* subcode. "
        "Do not decide whether the rubric precisely grades outputs except when rubric dependence hides a weak artifact. RubricGate handles rubric validity. "
        "Cite concrete candidate fields in evidence and rationale. "
        "Do not rewrite or repair anything. Return JSON only."
    )

    def _candidate_view(self, candidate: CandidateSample) -> dict[str, Any]:
        return candidate_quality_prompt_view(candidate)

    def _criteria(self, rules: list[str]) -> dict[str, Any]:
        criteria = super()._criteria(rules)
        criteria["general_probe_principles"] = self.domain.general_probe_principles
        criteria["anti_overfit_policy"] = self.domain.anti_overfit_policy
        return criteria


class RubricGate(_GateValidator):
    role_name = "RubricGate"
    check_kind = "rubric"
    rules_attr = "rubric_gate_rules"
    system_prompt = (
        "You are RubricGate for a benchmark-generation pipeline. "
        "Validate the rubric for this exact benchmark artifact. "
        "Ask whether score_x and scoring_contract match what the candidate actually asks an evaluated agent to do. "
        "Accept only when the rubric would reward the intended successful behavior and penalize known-bad, shallow, overfit, or non-causal outputs. "
        "Reject if scoring dimensions are vague, permissive, inconsistent with the artifact, reliant on invented ground truth, or likely to reward visible-test passing, explanation polish, or rubric compliance without the target ability. "
        "The prompt separates agent_visible_artifact from evaluator_private_context. Hidden evaluator-private context is allowed rubric context; never count it as candidate-facing answer leakage. "
        "Do not decide whether the benchmark is worth including except when the rubric cannot grade it reliably. QualityGate handles benchmark quality outside the rubric. "
        "Cite concrete rubric, scoring_contract, negative_controls, tests, or artifact fields in evidence and rationale. "
        "Do not rewrite or repair anything. Return JSON only."
    )


def _verdict(value: Any) -> Verdict:
    try:
        return Verdict(str(value).lower())
    except ValueError:
        return Verdict.REJECT


def _route_code(value: Any, *, default: RouteCode) -> RouteCode:
    try:
        return RouteCode(str(value))
    except ValueError:
        return default


def _generator_safe_retry_subcodes(subcodes: list[str]) -> list[str]:
    safe: list[str] = []
    for code in subcodes:
        mapped = GENERATOR_RETRY_CODE_MAP.get(code, code)
        if mapped not in safe:
            safe.append(mapped)
    return safe


def _generator_retry_guidance(subcodes: list[str]) -> list[str]:
    guidance: list[str] = []
    for code in subcodes:
        item = GENERATOR_RETRY_GUIDANCE.get(code)
        if item and item not in guidance:
            guidance.append(item)
    return guidance


def _design_retry_guidance(subcodes: list[str]) -> list[str]:
    guidance: list[str] = []
    for code in subcodes:
        item = DESIGN_RETRY_GUIDANCE.get(code)
        if item and item not in guidance:
            guidance.append(item)
    return guidance


def _coerce_gate_verdict(
    *,
    verdict: Verdict,
    route_code: RouteCode,
    subcodes: list[str],
) -> tuple[Verdict, RouteCode, list[str]]:
    labels = {str(code) for code in subcodes}
    reject_labels = sorted(labels & REJECT_SIGNAL_CODES)
    if verdict == Verdict.ACCEPT and reject_labels:
        merged_subcodes = _dedupe([*subcodes, *reject_labels])
        route = (
            RouteCode.REJECT_LEAKAGE
            if "shortcut_leakage" in reject_labels or any(_is_answer_leak_subcode(code) for code in reject_labels)
            else RouteCode.REJECT_SEMANTIC_MISMATCH
        )
        return Verdict.REJECT, route, merged_subcodes
    return verdict, route_code, subcodes


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _evidence(values: Any) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    if not isinstance(values, list):
        return refs
    for value in values:
        if isinstance(value, dict):
            refs.append(
                EvidenceRef(
                    source=str(value.get("source", "llm")),
                    path=str(value.get("path", "")),
                    value=None if value.get("value") is None else str(value.get("value")),
                )
            )
    return refs
