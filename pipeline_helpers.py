from __future__ import annotations

import re
from typing import Any

from config import RuntimeConfig
from models import (
    CandidateSample,
    ContextPolicy,
    DesignBrief,
    DesignVerdict,
    GenerationPipelineResult,
    RouteCode,
    SampleVerdict,
    StageRecord,
    Verdict,
    stable_hash,
)
from provider_errors import ProviderError, ProviderStructuredOutputError


def _producer_context_policy(retry_route_code: RouteCode | None) -> ContextPolicy:
    if retry_route_code is None:
        return ContextPolicy.FRESH
    if retry_route_code in {RouteCode.RETRY_INFRA, RouteCode.RETRY_PARSE, RouteCode.RETRY_PROVIDER_EMPTY}:
        return ContextPolicy.SAME_INPUT_RETRY
    return ContextPolicy.CRITERIA_PLUS_ROUTE_CODE


def _write_generation_result(result: GenerationPipelineResult) -> None:
    if result.result_path is None:
        return
    result.result_path.parent.mkdir(parents=True, exist_ok=True)
    result.result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _gate_caveat_subcodes(*verdicts: SampleVerdict) -> list[str]:
    subcodes: list[str] = []
    for verdict in verdicts:
        if verdict.verdict != Verdict.REJECT:
            continue
        label = f"{verdict.check_kind}_gate_rejected"
        if label not in subcodes:
            subcodes.append(label)
        for subcode in verdict.subcodes:
            if subcode not in subcodes:
                subcodes.append(subcode)
    return subcodes


def _gate_provider_error_route(*verdicts: SampleVerdict) -> RouteCode | None:
    for verdict in verdicts:
        if verdict.route_code in {RouteCode.RETRY_PARSE, RouteCode.RETRY_INFRA, RouteCode.RETRY_PROVIDER_EMPTY}:
            return verdict.route_code
    return None


def _provider_error_route_code(exc: ProviderError) -> RouteCode:
    message = str(exc)
    if (
        isinstance(exc, ProviderStructuredOutputError)
        or "invalid JSON" in message
        or "structured output" in message
        or "revision patch" in message
        or "environment_ops" in message
    ):
        return RouteCode.RETRY_PARSE
    if "empty" in message.lower():
        return RouteCode.RETRY_PROVIDER_EMPTY
    return RouteCode.RETRY_INFRA


def _provider_error_verdict(candidate: CandidateSample, check_kind: str, route_code: RouteCode) -> SampleVerdict:
    return SampleVerdict(
        candidate_id=candidate.id,
        check_kind=check_kind,  # type: ignore[arg-type]
        verdict=Verdict.REJECT,
        route_code=route_code,
        subcodes=["provider_error"],
        rationale="Provider failed before a valid gate verdict could be parsed.",
    )


def _provider_error_output(exc: ProviderError) -> dict[str, Any]:
    output: dict[str, Any] = {"error": str(exc)}
    if isinstance(exc, ProviderStructuredOutputError):
        output["raw_provider_output"] = exc.raw_content
        if isinstance(exc.parsing_error, BaseException):
            output["parse_error"] = {
                "message": str(exc.parsing_error),
                "line": getattr(exc.parsing_error, "lineno", None),
                "column": getattr(exc.parsing_error, "colno", None),
                "position": getattr(exc.parsing_error, "pos", None),
            }
    return output


def _graph_recursion_limit(config: RuntimeConfig) -> int:
    design_rounds = config.domain.max_design_retries + 1
    designs_per_round = max(1, config.target_n * 2)
    generation_attempts = config.domain.max_generation_retries + 1
    per_design_steps = 2 + (generation_attempts * 4) + 1
    design_steps = 2
    return 10 + design_rounds * (design_steps + designs_per_round * per_design_steps)


def _stage_label(record: StageRecord) -> str:
    if record.role == "validate_design_batch_deterministically":
        return "design_det"
    if record.role == "audit_design":
        return "design_audit"
    if record.role == "generate_candidate_sample":
        return "generation"
    if record.role == "validate_candidate_deterministically":
        return "validation_det"
    if record.role == "quality_gate_candidate":
        return "quality_gate"
    if record.role == "rubric_gate_candidate":
        return "rubric_gate"
    if record.role == "curate_committed_sample":
        return "curation"
    return record.stage_kind.value


def _format_progress_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(_format_progress_value(item) for item in value if item is not None) or "-"
    if hasattr(value, "value"):
        return str(value.value)
    text = str(value)
    if not text:
        return ""
    return text.replace(" ", "_")


def _event_fields(fields: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        if key in {"prompt", "proxy", "candidate", "design", "stage_input", "stage_output"}:
            continue
        if hasattr(value, "value"):
            safe[key] = value.value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [item.value if hasattr(item, "value") else item for item in value if isinstance(item, (str, int, float, bool)) or hasattr(item, "value")]
        else:
            safe[key] = str(value)
    return safe


def _short_id(value: str) -> str:
    if len(value) <= 48:
        return value
    return f"{value[:22]}...{value[-22:]}"


def _candidate_progress(candidate: CandidateSample) -> dict[str, Any]:
    ability = candidate.ability_z.get("name") if isinstance(candidate.ability_z, dict) else None
    prompt = candidate.agent_artifact.benchmark_case.get("prompt")
    return {
        "id": candidate.id,
        "case_type": candidate.case_type,
        "ability": ability,
        "prompt": prompt,
        "proxy": candidate.judge_artifact.proxy_claim,
    }


def _local_design_verdict(design: DesignBrief, route_code: RouteCode, subcodes: list[str]) -> DesignVerdict:
    return DesignVerdict(
        design_id=design.id,
        verdict=Verdict.REJECT,
        route_code=route_code,
        subcodes=subcodes,
    )


def _require(value: Any | None, name: str) -> Any:
    if value is None:
        raise RuntimeError(f"pipeline state missing {name}")
    return value


def _local_meta(error: str | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "provider": "local",
        "model": "deterministic",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
        "cost_usd": 0.0,
        "prompt_hash": stable_hash({"local": True}),
    }
    if error:
        meta["error"] = error
    return meta


def _provider_error_meta(exc: ProviderError, model_config: Any) -> dict[str, Any]:
    latency_ms = _provider_error_latency_ms(str(exc))
    return {
        "provider": str(getattr(model_config, "provider", "unknown")),
        "model": str(getattr(model_config, "model", "unknown")),
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": latency_ms,
        "cost_usd": 0.0,
        "reasoning_effort": getattr(model_config, "reasoning_effort", None),
        "prompt_hash": stable_hash(
            {
                "provider_error": True,
                "provider": str(getattr(model_config, "provider", "unknown")),
                "model": str(getattr(model_config, "model", "unknown")),
            }
        ),
        "error": str(exc),
    }


def _provider_error_latency_ms(message: str) -> int:
    match = re.search(r"elapsed_ms=(\d+)", message)
    if match:
        return int(match.group(1))
    return 0
