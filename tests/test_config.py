from __future__ import annotations

import os

from config import ModelConfig, build_runtime_config, load_domain, load_env_file
from main import _design_from_instruction


def test_load_env_file_sets_missing_values(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("MODEL_API_KEY=sk-test\n", encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["MODEL_API_KEY"] == "sk-test"


def test_load_env_file_strips_quotes_and_export(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MODEL_NAME", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text('export MODEL_NAME="gpt-test"\n', encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["MODEL_NAME"] == "gpt-test"


def test_load_env_file_does_not_override_existing_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "from-shell")
    env_path = tmp_path / ".env"
    env_path.write_text("MODEL_API_KEY=from-file\n", encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["MODEL_API_KEY"] == "from-shell"


def test_reasoning_effort_is_not_invented_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_REASONING_EFFORT", raising=False)

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.reasoning_effort is None


def test_model_defaults_to_gpt_5_5(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_NAME", raising=False)

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.model == "gpt-5.5"


def test_embedding_defaults_to_local_hash(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.embedding_provider == "local"
    assert config.models.embedding_model == "local-hash-embedding"


def test_openai_model_env_is_ignored(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "ignored-model")

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.model == "gpt-5.5"


def test_reasoning_effort_can_be_disabled_with_empty_env(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_REASONING_EFFORT", "")

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.reasoning_effort is None


def test_model_auth_file_can_be_set_with_env(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "auth.json"
    monkeypatch.setenv("MODEL_AUTH_FILE", str(auth_file))

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.auth_file == auth_file


def test_gemini_provider_uses_gemini_key_and_openai_compatible_base_url(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("MODEL_NAME", "gemini-3.1-flash-lite")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.provider == "gemini"
    assert config.models.model == "gemini-3.1-flash-lite"
    assert config.models.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert config.models.api_key is not None
    assert config.models.api_key.get_secret_value() == "gemini-test-key"
    assert config.models.max_tokens is None


def test_xai_provider_uses_xai_key_and_native_langchain_base_url(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "xai")
    monkeypatch.setenv("MODEL_NAME", "grok-4.3")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.provider == "xai"
    assert config.models.model == "grok-4.3"
    assert config.models.base_url is None
    assert config.models.api_key is not None
    assert config.models.api_key.get_secret_value() == "xai-test-key"


def test_xai_provider_accepts_grok_key_alias(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "grok")
    monkeypatch.setenv("MODEL_NAME", "grok-4.3")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", "grok-test-key")

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.provider == "grok"
    assert config.models.base_url is None
    assert config.models.api_key is not None
    assert config.models.api_key.get_secret_value() == "grok-test-key"


def test_model_max_tokens_can_be_set_with_env(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_MAX_TOKENS", "8192")

    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=42,
        run_id="test",
        console_progress=False,
    )

    assert config.models.max_tokens == 8192


def test_runtime_config_accepts_explicit_models_without_env_transport(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "env-model")

    primary = ModelConfig(provider="gemini", model="explicit-primary", reasoning_effort="medium")

    config = build_runtime_config(
        domain_path="domains/benchmark_code_debug.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=1,
        run_id="explicit-model-test",
        models=primary,
    )

    assert config.models.model == "explicit-primary"


def test_runtime_config_keeps_instruction() -> None:
    config = build_runtime_config(
        domain_path="domains/benchmark_haiku.yaml",
        target_stage="benchmark",
        target_n=1,
        seed=1,
        run_id="instruction-test",
        instruction="Make a haiku benchmark about restrained grief.",
    )

    assert config.instruction == "Make a haiku benchmark about restrained grief."


def test_domain_loads_output_schema_from_json_file() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    assert domain.output_schema["type"] == "object"
    assert "agent_artifact" in domain.output_schema["required"]
    assert "judge_artifact" in domain.output_schema["required"]


def test_domain_output_schema_uses_domain_benchmark_case_schema() -> None:
    domain = load_domain("domains/flaky_concurrency_bug_triage_python.yaml")

    assert domain.output_schema["$defs"]["benchmark_case"]["required"] == [
        "case_id",
        "repo_files",
        "failing_test",
        "pytest_trace",
        "task_instructions",
    ]


def test_instruction_design_uses_domain_taxonomy() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")

    design = _design_from_instruction("Create a late-autumn layoffs haiku benchmark.", domain, run_id="one-shot")

    assert design.id == "one-shot-instruction-design"
    assert design.design_intent == "Create a late-autumn layoffs haiku benchmark."
    assert design.cell.case_type in domain.case_types
    assert design.cell.difficulty in domain.difficulties
    assert design.cell.scenario in domain.scenarios
    assert design.target_ability in domain.abilities
    assert design.target_environment in domain.environments


def test_benchmark_output_schema_requires_private_judge_outlets() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    judge_required = domain.output_schema["properties"]["judge_artifact"]["required"]

    assert "private_root_cause" in judge_required
    assert "expected_fix_properties" in judge_required
    assert "hidden_failure_modes" in judge_required
    assert "shallow_solution_traps" in judge_required
    assert "candidate_visibility_boundaries" in judge_required
