from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, SecretStr


GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_PROVIDER_ALIASES = {"gemini", "google", "google_ai", "google_genai"}
XAI_PROVIDER_ALIASES = {"xai", "grok"}


class ModelConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-5.5"
    embedding_provider: str = "local"
    embedding_model: str = "local-hash-embedding"
    base_url: Optional[str] = None
    api_key: Optional[SecretStr] = Field(default=None, repr=False)
    auth_file: Optional[Path] = None
    reasoning_effort: Optional[str] = None
    max_tokens: Optional[int] = None
    request_timeout_seconds: float = 180.0


class DomainConfig(BaseModel):
    domain_id: str
    case_types: list[str]
    difficulties: list[int]
    scenarios: list[str]
    abilities: list[str] = Field(default_factory=list)
    environments: list[str] = Field(default_factory=list)
    diagnostic_pressure_types: list[str] = Field(default_factory=list)
    scoring_methods: list[str] = Field(default_factory=list)
    route_codes: list[str]
    subcodes: list[str]
    novelty_threshold: float = 0.08
    max_design_retries: int = 2
    max_generation_retries: int = 2
    deterministic_rules: dict[str, Any] = Field(default_factory=dict)
    semantic_rules: list[str] = Field(default_factory=list)
    general_probe_principles: dict[str, Any] = Field(default_factory=dict)
    anti_overfit_policy: list[str] = Field(default_factory=list)
    quality_gate_rules: list[str] = Field(default_factory=list)
    rubric_gate_rules: list[str] = Field(default_factory=list)
    generator_guidance: dict[str, Any] = Field(default_factory=dict)
    output_schema_path: Optional[str] = None
    output_schema: dict[str, Any] = Field(default_factory=dict)
    benchmark_case_schema: dict[str, Any]


class RuntimeConfig(BaseModel):
    domain: DomainConfig
    domain_path: Path
    target_stage: str = "benchmark"
    target_n: int = 5
    seed: int = 42
    run_id: str
    data_dir: Path = Path("data")
    logs_dir: Path = Path("logs")
    models: ModelConfig = Field(default_factory=ModelConfig)
    generator_model: Optional[ModelConfig] = None
    adversary_model: Optional[ModelConfig] = None
    revisor_model: Optional[ModelConfig] = None
    quality_gate_model: Optional[ModelConfig] = None
    rubric_gate_model: Optional[ModelConfig] = None
    console_progress: bool = True
    instruction: Optional[str] = None


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = _strip_env_value(value.strip())
            if key and key not in os.environ:
                os.environ[key] = value


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_domain(path: str | Path) -> DomainConfig:
    domain_path = Path(path)
    with domain_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    schema_path = raw.get("output_schema_path")
    if schema_path:
        resolved = Path(schema_path)
        if not resolved.is_absolute():
            resolved = domain_path.parent / resolved
        with resolved.open("r", encoding="utf-8") as schema_handle:
            raw["output_schema"] = json.load(schema_handle)
    _bind_benchmark_case_schema(raw)
    return DomainConfig.model_validate(raw)


def _bind_benchmark_case_schema(raw: dict[str, Any]) -> None:
    """Keep the top-level output schema and domain benchmark-case schema in sync."""
    output_schema = raw.get("output_schema")
    benchmark_case_schema = raw.get("benchmark_case_schema")
    if not isinstance(output_schema, dict) or not isinstance(benchmark_case_schema, dict):
        return
    defs = output_schema.setdefault("$defs", {})
    if not isinstance(defs, dict):
        return
    bound = deepcopy(benchmark_case_schema)
    bound.pop("$schema", None)
    defs["benchmark_case"] = bound


def build_runtime_config(
    *,
    domain_path: str | Path,
    target_stage: str,
    target_n: int,
    seed: int,
    run_id: str,
    models: Optional[ModelConfig] = None,
    generator_model: Optional[ModelConfig] = None,
    adversary_model: Optional[ModelConfig] = None,
    revisor_model: Optional[ModelConfig] = None,
    quality_gate_model: Optional[ModelConfig] = None,
    rubric_gate_model: Optional[ModelConfig] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    auth_file: Optional[str | Path] = None,
    embedding_model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_tokens: Optional[int] = None,
    request_timeout_seconds: Optional[float] = None,
    console_progress: bool = True,
    instruction: Optional[str] = None,
) -> RuntimeConfig:
    load_env_file()
    domain = load_domain(domain_path)
    runtime_models = models or build_model_config_from_env(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        auth_file=auth_file,
        embedding_model=embedding_model,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        request_timeout_seconds=request_timeout_seconds,
    )
    return RuntimeConfig(
        domain=domain,
        domain_path=Path(domain_path),
        target_stage=target_stage,
        target_n=target_n,
        seed=seed,
        run_id=run_id,
        models=runtime_models,
        generator_model=generator_model,
        adversary_model=adversary_model,
        revisor_model=revisor_model,
        quality_gate_model=quality_gate_model,
        rubric_gate_model=rubric_gate_model,
        console_progress=console_progress,
        instruction=instruction,
    )


def build_model_config_from_env(
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    auth_file: Optional[str | Path] = None,
    embedding_model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_tokens: Optional[int] = None,
    request_timeout_seconds: Optional[float] = None,
    env_policy_defaults: bool = True,
) -> ModelConfig:
    model_provider = provider or os.getenv("MODEL_PROVIDER", "openai")
    resolved_reasoning_effort = reasoning_effort
    if resolved_reasoning_effort is None and env_policy_defaults:
        resolved_reasoning_effort = os.getenv("MODEL_REASONING_EFFORT")
    resolved_max_tokens = max_tokens
    if resolved_max_tokens is None and env_policy_defaults:
        resolved_max_tokens = _optional_positive_int(os.getenv("MODEL_MAX_TOKENS"))
    resolved_timeout = request_timeout_seconds
    if resolved_timeout is None and env_policy_defaults:
        resolved_timeout = float(os.getenv("MODEL_TIMEOUT_SECONDS", "180"))
    return ModelConfig(
        provider=model_provider,
        model=model or os.getenv("MODEL_NAME", "gpt-5.5"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
        embedding_model=embedding_model or os.getenv("EMBEDDING_MODEL", "local-hash-embedding"),
        base_url=_default_model_base_url(model_provider, base_url or os.getenv("MODEL_BASE_URL")),
        api_key=_optional_secret(_default_model_api_key(model_provider, api_key or os.getenv("MODEL_API_KEY"))),
        auth_file=_resolve_optional_path(auth_file or os.getenv("MODEL_AUTH_FILE")),
        reasoning_effort=resolved_reasoning_effort or None,
        max_tokens=resolved_max_tokens,
        request_timeout_seconds=resolved_timeout if resolved_timeout is not None else 180.0,
    )


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _optional_secret(value: str | None) -> SecretStr | None:
    if value is None or not value.strip():
        return None
    return SecretStr(value.strip())


def _optional_positive_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    parsed = int(value.strip())
    if parsed <= 0:
        raise ValueError("token limits must be positive integers")
    return parsed


def _normalized_provider(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_")


def _is_gemini_provider(value: str | None) -> bool:
    return _normalized_provider(value) in GEMINI_PROVIDER_ALIASES


def _is_xai_provider(value: str | None) -> bool:
    return _normalized_provider(value) in XAI_PROVIDER_ALIASES


def _default_model_base_url(provider: str | None, explicit_base_url: str | None) -> str | None:
    if explicit_base_url and explicit_base_url.strip():
        return explicit_base_url.strip()
    if _is_gemini_provider(provider):
        return GEMINI_OPENAI_BASE_URL
    return None


def _default_model_api_key(provider: str | None, explicit_api_key: str | None) -> str | None:
    if _is_gemini_provider(provider) and os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY")
    if _is_xai_provider(provider):
        xai_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if xai_key:
            return xai_key
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key
    return None
