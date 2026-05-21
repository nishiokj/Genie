from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from config import DomainConfig
from models import CandidateSample, DesignBrief, GenerationEnvelope, stable_hash
from model_helpers import _nonempty_string
from provider_errors import ProviderError
from services.execution_workspace import ExecutionWorkspace, ExecutionWorkspaceError, normalize_workspace_path
from structured_output import _codex_structured_output_schema
from structured_schemas import generation_output_schema as _generation_output_schema

_REVISION_PATCH_KEYS = {"benchmark_case_updates", "metadata_updates", "environment_ops", "revision_rationale"}
_BENCHMARK_CASE_UPDATE_KEYS = {"prompt", "setup", "inputs", "environment"}
_METADATA_UPDATE_KEYS = {
    "score_x",
    "private_root_cause",
    "expected_fix_properties",
    "hidden_failure_modes",
    "shallow_solution_traps",
    "candidate_visibility_boundaries",
    "proxy_claim",
    "diagnostic_pressure",
    "scoring_contract",
    "leakage_risks",
    "known_limits",
    "coverage_tags",
    "negative_controls",
}
_ENVIRONMENT_OP_KEYS = {"op", "path", "old_text", "new_text", "create_if_missing", "replace_all"}


def _agent_artifact_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    agent_artifact = payload.get("agent_artifact")
    if not isinstance(agent_artifact, dict):
        raise ProviderError("generator output must include agent_artifact object")
    normalized = {
        "benchmark_case": dict(agent_artifact.get("benchmark_case", {})),
    }
    if agent_artifact.get("runtime_requirements") is not None:
        normalized["runtime_requirements"] = agent_artifact.get("runtime_requirements")
    if agent_artifact.get("environment_artifact") is not None:
        normalized["environment_artifact"] = agent_artifact.get("environment_artifact")
    return normalized


def _candidate_from_generation_payload(
    *,
    run_id: str,
    envelope: GenerationEnvelope,
    design: DesignBrief,
    attempt: int,
    role_name: str,
    payload: dict[str, Any],
) -> CandidateSample:
    agent_artifact = _agent_artifact_from_payload(payload)
    judge_artifact = _judge_artifact_from_payload(payload)
    content = {
        "generation_envelope_id": envelope.id,
        "design_id": design.id,
        "cell": design.cell.model_dump(),
        "output": {
            "agent_artifact": agent_artifact,
            "judge_artifact": judge_artifact,
            "ability_z": payload.get("ability_z", {}),
            "environment_y": payload.get("environment_y", {}),
        },
        "agent_artifact": agent_artifact,
        "judge_artifact": judge_artifact,
        "ability_z": payload.get("ability_z", {}),
        "environment_y": payload.get("environment_y", {}),
    }
    return CandidateSample(
        id=f"{run_id}-candidate-{design.id}-{attempt}",
        design_id=design.id,
        content_hash=stable_hash(content),
        cell=design.cell,
        output=dict(content["output"]),
        agent_artifact=agent_artifact,
        judge_artifact=judge_artifact,
        ability_z=dict(payload.get("ability_z", {})),
        environment_y=dict(payload.get("environment_y", {})),
        difficulty=design.cell.difficulty,
        case_type=design.cell.case_type,
        provenance={
            "design_id": design.id,
            "generation_envelope_id": envelope.id,
            "generator": role_name,
        },
    )


def _test_command_from_design(design: DesignBrief) -> str:
    commands = design.runtime_requirements.get("commands") if isinstance(design.runtime_requirements, dict) else None
    if isinstance(commands, dict) and isinstance(commands.get("test"), str) and commands["test"].strip():
        return commands["test"].strip()
    spec_commands = design.environment_artifact_spec.get("commands") if isinstance(design.environment_artifact_spec, dict) else None
    if isinstance(spec_commands, dict) and isinstance(spec_commands.get("test"), str) and spec_commands["test"].strip():
        return spec_commands["test"].strip()
    return "python -m pytest -q"


def _workspace_tool_final_shape(domain: DomainConfig) -> dict[str, Any]:
    return {
        "benchmark_case": {
            "prompt": "complete agent-facing task prompt",
            "setup": "complete setup instructions",
            "inputs": {},
            "environment": {},
        },
        "runtime_requirements": "runtime requirements object matching the design brief",
        "workspace_commands": {"test": "test command for the Executioner workspace"},
        "judge_artifact": _example_output_for_domain(domain)["judge_artifact"],
        "ability_z": {"name": "target ability", "sub_abilities": ["specific sub-ability"]},
        "environment_y": {"name": "target environment", "assumptions": ["assumption"]},
    }


def _finalize_candidate_tool_api_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "benchmark_case",
            "runtime_requirements",
            "workspace_commands",
            "judge_artifact",
            "ability_z",
            "environment_y",
        ],
        "properties": {
            "benchmark_case": {
                "type": "object",
                "description": "Benchmark metadata matching the prompt-provided finalize_candidate_json_schema.",
            },
            "runtime_requirements": {
                "type": "object",
                "description": "Runtime requirements matching the prompt-provided finalize_candidate_json_schema.",
            },
            "workspace_commands": {
                "type": "object",
                "description": "Workspace commands matching the prompt-provided finalize_candidate_json_schema.",
            },
            "judge_artifact": {
                "type": "object",
                "description": "Private judge artifact matching the prompt-provided finalize_candidate_json_schema.",
            },
            "ability_z": {
                "type": "object",
                "description": "Target ability metadata matching the prompt-provided finalize_candidate_json_schema.",
            },
            "environment_y": {
                "type": "object",
                "description": "Target environment metadata matching the prompt-provided finalize_candidate_json_schema.",
            },
        },
    }


def _workspace_tool_schemas(domain: DomainConfig) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "write_file",
            "description": "Create or replace one evaluated-agent-visible file in the Executioner benchmark workspace. Content must look like a normal project file; do not write benchmark-author notes, BUG labels, root-cause explanations, intended fixes, fault-location hints, grader notes, or test/assertion text that teaches the diagnosis.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"path": {"type": "string"}, "content": {"type": "string", "maxLength": 262144}},
                "required": ["path", "content"],
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "read_file",
            "description": "Read one file previously written to the Executioner benchmark workspace.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "list_files",
            "description": "List paths currently present in the Executioner benchmark workspace.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "finalize_candidate",
            "description": "Finish benchmark generation after all workspace files have been authored. Provide structured benchmark metadata only; do not include file contents.",
            "parameters": _finalize_candidate_tool_api_parameters(),
            "strict": False,
        },
    ]


def _finalize_candidate_tool_parameters(domain: DomainConfig) -> dict[str, Any]:
    output_schema = _generation_output_schema(domain) if isinstance(domain.output_schema, dict) else {}
    output_properties = output_schema.get("properties") if isinstance(output_schema.get("properties"), dict) else {}
    defs = deepcopy(output_schema.get("$defs") if isinstance(output_schema.get("$defs"), dict) else {})

    benchmark_case_schema = deepcopy(domain.benchmark_case_schema or defs.get("benchmark_case") or {})
    benchmark_case_schema.pop("$schema", None)
    if not benchmark_case_schema:
        benchmark_case_schema = {"type": "object", "additionalProperties": True}
    benchmark_case_schema.setdefault("type", "object")
    benchmark_case_schema.setdefault("additionalProperties", True)
    benchmark_properties = benchmark_case_schema.setdefault("properties", {})
    benchmark_properties.setdefault("prompt", {"type": "string", "minLength": 20})
    benchmark_properties.setdefault("setup", {"type": "string"})
    benchmark_properties["inputs"] = {"type": "object", "additionalProperties": True}
    benchmark_properties["environment"] = {"type": "object", "additionalProperties": True}
    benchmark_case_schema["required"] = sorted(set(benchmark_case_schema.get("required") or []) | {"prompt", "setup", "inputs", "environment"})

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "benchmark_case",
            "runtime_requirements",
            "workspace_commands",
            "judge_artifact",
            "ability_z",
            "environment_y",
        ],
        "properties": {
            "benchmark_case": benchmark_case_schema,
            "runtime_requirements": deepcopy(output_properties.get("agent_artifact", {}))
            .get("properties", {})
            .get("runtime_requirements", {"type": "object"}),
            "workspace_commands": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "required": ["test"],
                "properties": {"test": {"type": "string", "minLength": 1}},
            },
            "judge_artifact": deepcopy(output_properties.get("judge_artifact", {"type": "object"})),
            "ability_z": deepcopy(output_properties.get("ability_z", {"type": "object"})),
            "environment_y": deepcopy(output_properties.get("environment_y", {"type": "object"})),
        },
        "$defs": defs,
    }


def _finalize_candidate_validation_schema(domain: DomainConfig) -> dict[str, Any]:
    schema = _codex_structured_output_schema(_finalize_candidate_tool_parameters(domain))
    benchmark_properties = schema.get("properties", {}).get("benchmark_case", {}).get("properties", {})
    for key in ("inputs", "environment"):
        field_schema = benchmark_properties.get(key)
        if isinstance(field_schema, dict):
            field_schema["additionalProperties"] = True
    return schema


_WORKSPACE_MAX_FILES = 20
_WORKSPACE_MAX_FILE_BYTES = 262_144
_WORKSPACE_MAX_TOTAL_BYTES = 524_288


def _workspace_total_bytes(workspace: ExecutionWorkspace) -> int:
    return sum(len(workspace.read_file(p).encode("utf-8")) for p in workspace.list_files())


def _normalize_tool_call_for_input(item: dict[str, Any]) -> dict[str, Any]:
    call_id = _nonempty_string(item.get("call_id")) or _nonempty_string(item.get("id")) or ""
    args = item.get("arguments")
    if isinstance(args, dict):
        args = json.dumps(args, sort_keys=True)
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": str(item.get("name") or ""),
        "arguments": args or "{}",
    }


def _execute_workspace_tool(name: str, args: dict[str, Any], workspace: ExecutionWorkspace, domain: DomainConfig) -> dict[str, Any]:
    if name == "write_file":
        path = normalize_workspace_path(args.get("path"), "write_file.path")
        content = args.get("content")
        if not isinstance(content, str):
            raise ProviderError("write_file.content must be a string")
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > _WORKSPACE_MAX_FILE_BYTES:
            raise ProviderError(f"write_file content exceeds {_WORKSPACE_MAX_FILE_BYTES // 1024} KB limit ({content_bytes} bytes)")
        is_new_file = path not in workspace.list_files()
        if is_new_file and len(workspace.list_files()) >= _WORKSPACE_MAX_FILES:
            raise ProviderError(f"workspace file count limit reached ({_WORKSPACE_MAX_FILES} files)")
        total_after = _workspace_total_bytes(workspace) + (content_bytes if is_new_file else content_bytes - len(workspace.read_file(path).encode("utf-8")))
        if total_after > _WORKSPACE_MAX_TOTAL_BYTES:
            raise ProviderError(f"workspace total size would exceed {_WORKSPACE_MAX_TOTAL_BYTES // 1024} KB limit")
        workspace.write_file(path, content)
        return {"ok": True, "path": path, "bytes": content_bytes}
    if name == "read_file":
        path = normalize_workspace_path(args.get("path"), "read_file.path")
        return {"ok": True, "path": path, "content": workspace.read_file(path)}
    if name == "list_files":
        return {"ok": True, "files": workspace.list_files()}
    if name == "finalize_candidate":
        if len(workspace.list_files()) < 3:
            raise ProviderError("finalize_candidate requires at least three workspace files")
        _finalize_payload_from_tool_args(args, domain)
        return {"ok": True, "files": workspace.list_files()}
    raise ProviderError(f"unknown workspace tool: {name}")


def _finalize_payload_from_tool_args(args: dict[str, Any], domain: DomainConfig) -> dict[str, Any]:
    if "payload_json" in args:
        raise ProviderError("finalize_candidate.payload_json is no longer supported; pass structured fields directly")
    parsed = args
    if not isinstance(parsed, dict):
        raise ProviderError("finalize_candidate arguments must be a JSON object")
    required = {"benchmark_case", "runtime_requirements", "workspace_commands", "judge_artifact", "ability_z", "environment_y"}
    missing = sorted(required - set(parsed))
    if missing:
        raise ProviderError(f"finalize_candidate missing required keys: {missing}")
    errors = sorted(
        Draft202012Validator(_finalize_candidate_validation_schema(domain)).iter_errors(parsed),
        key=lambda error: list(error.path),
    )
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.path) or "<root>"
        raise ProviderError(f"finalize_candidate schema violation at {path}: {first.message}")
    return parsed


def _generation_payload_from_workspace_final(finalized: dict[str, Any], workspace: ExecutionWorkspace, design: DesignBrief) -> dict[str, Any]:
    commands = finalized.get("workspace_commands")
    if isinstance(commands, dict):
        workspace.commands = {str(key): str(value) for key, value in commands.items()}
    if not workspace.commands.get("test"):
        workspace.commands["test"] = _test_command_from_design(design)
    payload = workspace.artifact_payload()
    _enforce_required_workspace_files(payload, design)
    agent_artifact = {
        "benchmark_case": dict(finalized.get("benchmark_case") or {}),
        "runtime_requirements": dict(finalized.get("runtime_requirements") or design.runtime_requirements),
        "environment_artifact": {"kind": "executioner_workspace", "payload": payload},
    }
    return {
        "agent_artifact": agent_artifact,
        "judge_artifact": dict(finalized.get("judge_artifact") or {}),
        "ability_z": dict(finalized.get("ability_z") or {}),
        "environment_y": dict(finalized.get("environment_y") or {}),
    }


def _enforce_required_workspace_files(payload: dict[str, Any], design: DesignBrief) -> None:
    spec = design.environment_artifact_spec if isinstance(design.environment_artifact_spec, dict) else {}
    required_files = spec.get("required_files")
    if not isinstance(required_files, list) or not required_files:
        return
    files = payload.get("files")
    present: set[str] = set()
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                present.add(item["path"])
    missing = sorted(str(path) for path in required_files if isinstance(path, str) and path not in present)
    if missing:
        raise ProviderError(f"workspace is missing required_files from design.environment_artifact_spec: {missing}")


def _judge_artifact_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    judge_artifact = payload.get("judge_artifact")
    if not isinstance(judge_artifact, dict):
        raise ProviderError("generator output must include judge_artifact object")
    return {
        "score_x": dict(judge_artifact.get("score_x", {})),
        "private_root_cause": str(judge_artifact.get("private_root_cause", "")),
        "expected_fix_properties": list(judge_artifact.get("expected_fix_properties", [])),
        "hidden_failure_modes": list(judge_artifact.get("hidden_failure_modes", [])),
        "shallow_solution_traps": list(judge_artifact.get("shallow_solution_traps", [])),
        "candidate_visibility_boundaries": list(judge_artifact.get("candidate_visibility_boundaries", [])),
        "proxy_claim": str(judge_artifact.get("proxy_claim", "")),
        "diagnostic_pressure": list(judge_artifact.get("diagnostic_pressure", [])),
        "scoring_contract": dict(judge_artifact.get("scoring_contract", {})),
        "leakage_risks": list(judge_artifact.get("leakage_risks", [])),
        "known_limits": list(judge_artifact.get("known_limits", [])),
        "coverage_tags": list(judge_artifact.get("coverage_tags", [])),
        "negative_controls": list(judge_artifact.get("negative_controls", [])),
    }


def _revision_patch_shape(domain: DomainConfig) -> dict[str, Any]:
    shape: dict[str, Any] = {
        "benchmark_case_updates": {
            "prompt": "optional complete replacement prompt",
            "setup": "optional complete replacement setup",
            "inputs": "optional complete replacement inputs object",
            "environment": "optional complete replacement environment object",
        },
        "metadata_updates": {
            "score_x": "optional complete replacement scoring object",
            "private_root_cause": "optional complete replacement private root-cause diagnosis",
            "expected_fix_properties": "optional complete replacement list of judge-only expected repair properties",
            "hidden_failure_modes": "optional complete replacement list of judge-only hidden failure modes",
            "shallow_solution_traps": "optional complete replacement list of judge-only shallow solution traps",
            "candidate_visibility_boundaries": "optional complete replacement list of what must remain hidden from the evaluated agent",
            "proxy_claim": "optional complete replacement proxy claim",
            "diagnostic_pressure": "optional complete replacement list",
            "scoring_contract": "optional complete replacement scoring contract",
            "leakage_risks": "optional complete replacement list",
            "known_limits": "optional complete replacement list",
            "coverage_tags": "optional complete replacement list",
            "negative_controls": "optional complete replacement list",
        },
        "environment_ops": [],
        "revision_rationale": "short private rationale for the patch",
    }
    if domain.domain_id == "benchmark_code_debug":
        shape["environment_ops"] = [
            {
                "op": "edit_file",
                "path": "relative/path.py",
                "old_text": "exact text to replace; omit only when creating a new file",
                "new_text": "replacement text or complete new file contents",
                "create_if_missing": False,
                "replace_all": False,
            }
        ]
    return shape


def _apply_revision_patch(
    candidate: CandidateSample,
    patch: dict[str, Any],
    *,
    execution_workspace: ExecutionWorkspace | None = None,
    workspace_subdir: str | None = None,
) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ProviderError("revision patch must be a JSON object")
    current_runtime_requirements = candidate.agent_artifact.runtime_requirements
    if "runtime_requirements" in patch:
        returned_runtime_requirements = patch.pop("runtime_requirements")
        if returned_runtime_requirements != current_runtime_requirements:
            raise ProviderError("revision patch cannot change runtime_requirements; revise files/metadata while preserving the seed runtime contract")
    unknown_keys = set(patch) - _REVISION_PATCH_KEYS
    if unknown_keys:
        raise ProviderError(f"revision patch contains unsupported top-level keys: {sorted(unknown_keys)}")

    benchmark_case = deepcopy(candidate.agent_artifact.benchmark_case)
    metadata: dict[str, Any] = {
        "score_x": deepcopy(candidate.judge_artifact.score_x),
        "ability_z": deepcopy(candidate.ability_z),
        "environment_y": deepcopy(candidate.environment_y),
        "private_root_cause": candidate.judge_artifact.private_root_cause,
        "expected_fix_properties": list(candidate.judge_artifact.expected_fix_properties),
        "hidden_failure_modes": list(candidate.judge_artifact.hidden_failure_modes),
        "shallow_solution_traps": list(candidate.judge_artifact.shallow_solution_traps),
        "candidate_visibility_boundaries": list(candidate.judge_artifact.candidate_visibility_boundaries),
        "proxy_claim": candidate.judge_artifact.proxy_claim,
        "diagnostic_pressure": list(candidate.judge_artifact.diagnostic_pressure),
        "scoring_contract": deepcopy(candidate.judge_artifact.scoring_contract),
        "leakage_risks": list(candidate.judge_artifact.leakage_risks),
        "known_limits": list(candidate.judge_artifact.known_limits),
        "coverage_tags": list(candidate.judge_artifact.coverage_tags),
        "negative_controls": deepcopy(candidate.judge_artifact.negative_controls),
    }
    environment_artifact = (
        candidate.agent_artifact.environment_artifact.model_dump(mode="json")
        if candidate.agent_artifact.environment_artifact is not None
        else None
    )
    runtime_requirements = deepcopy(current_runtime_requirements)

    changed = False
    benchmark_updates = patch.get("benchmark_case_updates", {})
    if benchmark_updates is None:
        benchmark_updates = {}
    if not isinstance(benchmark_updates, dict):
        raise ProviderError("revision patch benchmark_case_updates must be an object")
    unknown_benchmark_keys = set(benchmark_updates) - _BENCHMARK_CASE_UPDATE_KEYS
    if unknown_benchmark_keys:
        raise ProviderError(f"revision patch contains unsupported benchmark_case_updates keys: {sorted(unknown_benchmark_keys)}")
    for key, value in benchmark_updates.items():
        benchmark_case[key] = value
        changed = True

    metadata_updates = patch.get("metadata_updates", {})
    if metadata_updates is None:
        metadata_updates = {}
    if not isinstance(metadata_updates, dict):
        raise ProviderError("revision patch metadata_updates must be an object")
    unknown_metadata_keys = set(metadata_updates) - _METADATA_UPDATE_KEYS
    if unknown_metadata_keys:
        raise ProviderError(f"revision patch contains unsupported metadata_updates keys: {sorted(unknown_metadata_keys)}")
    for key, value in metadata_updates.items():
        metadata[key] = value
        changed = True

    environment_ops = patch.get("environment_ops", [])
    if environment_ops is None:
        environment_ops = []
    if not isinstance(environment_ops, list):
        raise ProviderError("revision patch environment_ops must be a list")
    if environment_ops:
        if not environment_artifact or environment_artifact.get("kind") != "executioner_workspace":
            raise ProviderError("revision patch environment_ops require an executioner_workspace environment_artifact")
        workspace: ExecutionWorkspace | None = None
        close_workspace = False
        try:
            if execution_workspace is not None:
                workspace = execution_workspace
                if workspace_subdir is not None:
                    source_workspace = ExecutionWorkspace.from_artifact(environment_artifact.get("payload", {}))
                    try:
                        workspace.reset(workspace_subdir)
                        for path in source_workspace.list_files():
                            workspace.write_file(path, source_workspace.read_file(path))
                    finally:
                        source_workspace.close()
            else:
                workspace = ExecutionWorkspace.from_artifact(environment_artifact.get("payload", {}))
                close_workspace = True
            for index, op in enumerate(environment_ops):
                if not isinstance(op, dict):
                    raise ProviderError(f"revision patch environment_ops.{index} must be an object")
                _apply_environment_edit(workspace, op, index)
            environment_artifact["payload"] = workspace.artifact_payload()
        except ExecutionWorkspaceError as exc:
            raise ProviderError(f"revision patch execution workspace error: {exc.subcode} at {exc.path}: {exc.message}") from exc
        finally:
            if close_workspace and workspace is not None:
                workspace.close()
        changed = True

    if not changed:
        raise ProviderError("revision patch made no candidate changes")

    return {
        "benchmark_case": benchmark_case,
        "runtime_requirements": runtime_requirements,
        "environment_artifact": environment_artifact,
        **metadata,
    }


def _apply_environment_edit(workspace: ExecutionWorkspace, op: dict[str, Any], index: int) -> None:
    if op.get("op") != "edit_file":
        raise ProviderError(f"revision patch environment_ops.{index}.op must be 'edit_file'")
    unknown_keys = set(op) - _ENVIRONMENT_OP_KEYS
    if unknown_keys:
        raise ProviderError(f"revision patch environment_ops.{index} contains unsupported keys: {sorted(unknown_keys)}")
    path = normalize_workspace_path(op.get("path"), f"revision patch environment_ops.{index}.path")
    old_text = op.get("old_text")
    new_text = op.get("new_text")
    create_if_missing = bool(op.get("create_if_missing", False))
    replace_all = bool(op.get("replace_all", False))
    if not isinstance(new_text, str):
        raise ProviderError(f"revision patch environment_ops.{index}.new_text must be a string")

    if create_if_missing:
        if old_text not in (None, ""):
            raise ProviderError(f"revision patch environment_ops.{index}.old_text must be omitted when create_if_missing=true")
        workspace.edit_file(path=path, old_text=None, new_text=new_text, create_if_missing=True)
        return

    if not isinstance(old_text, str) or old_text == "":
        raise ProviderError(f"revision patch environment_ops.{index}.old_text must be a non-empty exact string")
    current = workspace.read_file(path)
    occurrences = current.count(old_text)
    if occurrences == 0:
        raise ProviderError(f"revision patch environment_ops.{index}.old_text did not match file")
    if occurrences > 1 and not replace_all:
        raise ProviderError(
            f"revision patch environment_ops.{index}.old_text matched {occurrences} times; set replace_all=true only when every occurrence should change"
        )
    try:
        workspace.edit_file(path=path, old_text=old_text, new_text=new_text, replace_all=replace_all)
    except ProviderError as exc:
        raise ProviderError(f"revision patch environment_ops.{index} failed: {exc}") from exc


def _example_output_for_domain(domain: DomainConfig) -> dict[str, Any]:
    agent_artifact: dict[str, Any] = {
        "benchmark_case": {
            "prompt": "agent-facing task prompt string",
            "setup": "optional agent-facing setup",
            "inputs": {},
            "environment": {},
        }
    }
    if domain.domain_id == "benchmark_code_debug":
        agent_artifact["runtime_requirements"] = {
            "kind": "filesystem_task",
            "execution": {"mode": "task_image", "base_image": "python:3.11-slim", "os": "linux", "arch": "amd64"},
            "language": {"name": "python", "version": "3.11+"},
            "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
            "commands": {"test": "python -m pytest -q"},
            "network": "disabled_during_eval",
        }
        agent_artifact["environment_artifact"] = {
            "kind": "executioner_workspace",
            "payload": {
                "session_id": "executioner session id",
                "logical_root": "/workspace",
                "workspace_root": "host path to durable workspace",
                "files": [{"path": "relative/path.py"}, {"path": "tests/test_behavior.py"}, {"path": "README.md"}],
                "commands": {"test": "python -m pytest -q"},
            },
        }
    return {
        "agent_artifact": agent_artifact,
        "judge_artifact": {
            "score_x": {
                "score_type": "one allowed scoring method",
                "range": [0, 1],
                "dimensions": [
                    {
                        "name": "dimension name",
                        "weight": 0.5,
                        "high_score_criterion": "judge-facing behavior that earns full credit",
                        "low_score_criterion": "judge-facing behavior that earns zero credit",
                    }
                ],
            },
            "private_root_cause": "judge-only hidden diagnosis and causal mechanism; never visible to the evaluated agent",
            "expected_fix_properties": ["judge-only properties a strong repair should have"],
            "hidden_failure_modes": ["judge-only regressions or bad fixes hidden checks should catch"],
            "shallow_solution_traps": ["judge-only tempting repair that should fail or score poorly"],
            "candidate_visibility_boundaries": [
                "Do not reveal the private root cause, faulty expression, expected fix, or shallow traps in agent_artifact."
            ],
            "proxy_claim": "judge-facing claim for why score_x should indicate ability_z in environment_y",
            "diagnostic_pressure": ["judge-facing pressure exerted by this case"],
            "scoring_contract": {
                "credit": ["observable behavior that earns credit"],
                "penalties": ["shallow or bad behavior that loses credit"],
                "uncertainty_policy": "when judges should mark uncertainty",
            },
            "leakage_risks": ["how the case or scorer can be gamed"],
            "known_limits": ["what this benchmark case does not prove"],
            "coverage_tags": ["short coverage tags"],
            "negative_controls": [{"output": "known-bad agent output", "should_fail_because": "why score_x should penalize it"}],
        },
        "ability_z": {"name": "target ability", "sub_abilities": ["specific sub-ability"]},
        "environment_y": {"name": "target environment", "assumptions": ["assumption"]},
    }

def revision_patch_metrics(patch: Any) -> dict[str, Any]:
    if not isinstance(patch, dict):
        return {}
    environment_ops = patch.get("environment_ops")
    ops = environment_ops if isinstance(environment_ops, list) else []
    files_touched = sorted(
        {
            str(op.get("path"))
            for op in ops
            if isinstance(op, dict) and isinstance(op.get("path"), str) and op.get("path").strip()
        }
    )
    bytes_added = 0
    bytes_removed = 0
    full_rewrites = 0
    create_count = 0
    replace_all_count = 0
    for op in ops:
        if not isinstance(op, dict):
            continue
        old_text = op.get("old_text")
        new_text = op.get("new_text")
        if isinstance(new_text, str):
            bytes_added += len(new_text.encode("utf-8"))
        if isinstance(old_text, str):
            bytes_removed += len(old_text.encode("utf-8"))
        if bool(op.get("create_if_missing", False)):
            create_count += 1
            full_rewrites += 1
        if bool(op.get("replace_all", False)):
            replace_all_count += 1
    op_count = len(ops)
    return {
        "revision_op_count": op_count,
        "revision_edit_file_count": sum(1 for op in ops if isinstance(op, dict) and op.get("op") == "edit_file"),
        "revision_files_touched": len(files_touched),
        "revision_files_touched_list": files_touched,
        "revision_bytes_added": bytes_added,
        "revision_bytes_removed": bytes_removed,
        "revision_bytes_changed": bytes_added + bytes_removed,
        "revision_full_rewrite_count": full_rewrites,
        "revision_create_file_count": create_count,
        "revision_replace_all_count": replace_all_count,
        "revision_full_rewrite_ratio": (full_rewrites / op_count) if op_count else 0.0,
    }
