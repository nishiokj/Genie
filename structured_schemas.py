from __future__ import annotations

from copy import deepcopy
from typing import Any

from config import DomainConfig


EVIDENCE_REF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source": {"type": "string"},
        "path": {"type": "string"},
        "value": {"type": "string"},
    },
    "required": ["source", "path", "value"],
}

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string"},
        "route_code": {"type": "string"},
        "subcodes": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": EVIDENCE_REF_SCHEMA},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "route_code", "subcodes", "evidence", "rationale"],
}

DESIGN_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "case_type": {"type": "string"},
        "difficulty": {"type": "integer"},
        "scenario": {"type": "string"},
        "target_ability": {"type": "string"},
        "target_environment": {"type": "string"},
        "design_intent": {"type": "string"},
        "environment_premise": {"type": "object", "additionalProperties": True},
        "runtime_requirements": {"type": "object", "additionalProperties": True},
        "environment_artifact_spec": {"type": "object", "additionalProperties": True},
        "failure_mode_family": {"type": "string"},
        "diagnostic_pressure": {"type": "array", "items": {"type": "string"}},
        "why_weak_agents_fail": {"type": "array", "items": {"type": "string"}},
        "tempting_shallow_solutions": {"type": "array", "items": {"type": "string"}},
        "success_evidence_required": {"type": "array", "items": {"type": "string"}},
        "minimum_depth_requirements": {"type": "array", "items": {"type": "string"}},
        "forbidden_shortcuts": {"type": "array", "items": {"type": "string"}},
        "non_goals": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "case_type",
        "difficulty",
        "scenario",
        "target_ability",
        "target_environment",
        "design_intent",
        "environment_premise",
        "runtime_requirements",
        "environment_artifact_spec",
        "failure_mode_family",
        "diagnostic_pressure",
        "why_weak_agents_fail",
        "tempting_shallow_solutions",
        "success_evidence_required",
        "minimum_depth_requirements",
        "forbidden_shortcuts",
        "non_goals",
    ],
}

DESIGN_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"designs": {"type": "array", "items": DESIGN_ITEM_SCHEMA}},
    "required": ["designs"],
}

REVISION_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "benchmark_case_updates": {"type": "object", "additionalProperties": True},
        "metadata_updates": {"type": "object", "additionalProperties": True},
        "environment_ops": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {"type": "string", "enum": ["edit_file"]},
                    "path": {"type": "string", "minLength": 1, "maxLength": 240},
                    "old_text": {"type": "string", "maxLength": 1600},
                    "new_text": {"type": "string", "minLength": 1, "maxLength": 2400},
                    "replace_all": {"type": "boolean"},
                    "create_if_missing": {"type": "boolean"},
                },
                "required": ["op", "path", "new_text"],
            },
        },
        "revision_rationale": {"type": "string", "maxLength": 800},
    },
    "required": ["benchmark_case_updates", "metadata_updates", "environment_ops", "revision_rationale"],
}

ADVERSARY_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "revision_disposition": {"type": "string"},
        "disposition_rationale": {"type": "string"},
        "attack_summary": {"type": "string"},
        "attacks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "attack_target": {"type": "string"},
                    "attack_type": {"type": "string"},
                    "exploit_path": {"type": "string"},
                    "evidence": {"type": "string"},
                    "severity": {"type": "string"},
                    "why_it_invalidates_proxy": {"type": "string"},
                },
                "required": [
                    "attack_target",
                    "attack_type",
                    "exploit_path",
                    "evidence",
                    "severity",
                    "why_it_invalidates_proxy",
                ],
            },
        },
        "cheap_pass_strategy": {"type": "string"},
        "proxy_damage": {"type": "string"},
        "survival_requirements": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "revision_disposition",
        "disposition_rationale",
        "attack_summary",
        "attacks",
        "cheap_pass_strategy",
        "proxy_damage",
        "survival_requirements",
    ],
}


def generation_output_schema(domain: DomainConfig) -> dict[str, Any]:
    schema = deepcopy(domain.output_schema)
    if domain.deterministic_rules.get("require_runtime_requirements"):
        agent_artifact = schema.get("properties", {}).get("agent_artifact")
        if isinstance(agent_artifact, dict):
            required = list(agent_artifact.get("required", []))
            for field in ("benchmark_case", "runtime_requirements", "environment_artifact"):
                if field not in required:
                    required.append(field)
            agent_artifact["required"] = required
        defs = schema.setdefault("$defs", {})
        benchmark_case_schema = deepcopy(domain.benchmark_case_schema or defs.get("benchmark_case") or {})
        if isinstance(benchmark_case_schema, dict):
            benchmark_case_schema.pop("$schema", None)
            benchmark_case_schema.setdefault("type", "object")
            benchmark_case_schema.setdefault("properties", {})
            benchmark_case_schema["properties"] = {
                **benchmark_case_schema.get("properties", {}),
                "prompt": {"type": "string", "minLength": 20},
                "setup": {"type": "string"},
                "inputs": {"type": "object"},
                "environment": {"type": "object"},
            }
            required_case_fields = list(benchmark_case_schema.get("required", []))
            for field in ("prompt", "setup", "inputs", "environment"):
                if field not in required_case_fields:
                    required_case_fields.append(field)
            benchmark_case_schema["required"] = required_case_fields
            defs["benchmark_case"] = benchmark_case_schema
        runtime_schema = defs.get("runtime_requirements")
        if isinstance(runtime_schema, dict):
            runtime_schema["required"] = ["kind", "execution", "language", "dependencies", "commands", "network"]
            runtime_schema["properties"] = {
                **runtime_schema.get("properties", {}),
                "kind": {"type": "string", "enum": ["filesystem_task"]},
                "execution": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["mode", "base_image"],
                    "properties": {
                        "mode": {"type": "string", "enum": ["task_image", "container"]},
                        "base_image": {"type": "string", "minLength": 1},
                        "os": {"type": "string"},
                        "arch": {"type": "string"},
                    },
                },
                "language": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["name"],
                    "properties": {"name": {"type": "string", "minLength": 1}, "version": {"type": "string"}},
                },
                "dependencies": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["policy"],
                    "properties": {
                        "policy": {"type": "string", "minLength": 1},
                        "packages": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "commands": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["test"],
                    "properties": {"test": {"type": "string", "minLength": 1}},
                },
                "network": {"type": "string", "minLength": 1},
            }
        environment_schema = defs.get("environment_artifact")
        if isinstance(environment_schema, dict):
            environment_schema["required"] = ["kind", "payload"]
            environment_schema["properties"] = {
                **environment_schema.get("properties", {}),
                "kind": {"type": "string", "enum": ["executioner_workspace"]},
                "payload": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["workspace_root", "commands", "files"],
                    "properties": {
                        "session_id": {"type": "string"},
                        "logical_root": {"type": "string"},
                        "workspace_root": {"type": "string", "minLength": 1},
                        "files": {
                            "type": "array",
                            "minItems": 3,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["path"],
                                "properties": {
                                    "path": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                        "commands": {
                            "type": "object",
                            "additionalProperties": True,
                            "required": ["test"],
                            "properties": {"test": {"type": "string", "minLength": 1}},
                        },
                    },
                },
            }
        ability_schema = schema.get("properties", {}).get("ability_z")
        if isinstance(ability_schema, dict) and domain.abilities:
            ability_props = ability_schema.setdefault("properties", {})
            ability_props["name"] = {"type": "string", "enum": list(domain.abilities)}
        environment_y_schema = schema.get("properties", {}).get("environment_y")
        if isinstance(environment_y_schema, dict) and domain.environments:
            environment_y_props = environment_y_schema.setdefault("properties", {})
            environment_y_props["name"] = {"type": "string", "enum": list(domain.environments)}
    return schema
