from __future__ import annotations

import json
import socket
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from http.client import IncompleteRead
from typing import Any

import pytest

from agents import (
    Adversary,
    Designer,
    QualityGate,
    RubricGate,
    SampleGenerator,
    _coerce_gate_verdict,
    candidate_prompt_view,
)
from config import ModelConfig, load_domain
from codex_client import CodexClient, CodexResponse, _codex_stream_idle_timeout_seconds, _read_codex_sse
from generation_artifacts import (
    _candidate_from_generation_payload,
    _example_output_for_domain,
    _judge_artifact_from_payload,
    validate_generation_contract,
)
from model_client import ModelClient
from models import AdversaryReport, CandidateSample, DesignBrief, GenerationEnvelope, RouteCode, TaxonomyCell, Verdict
from provider_errors import ProviderError, ProviderStructuredOutputError
from services.execution_workspace import ExecutionWorkspace
from structured_output import _codex_structured_output_schema, _structured_output_schema
from structured_schemas import generation_output_schema as _generation_output_schema

_TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def _executioner_artifact(files: list[dict[str, str]], commands: dict[str, str] | None = None) -> dict[str, Any]:
    temp_dir = tempfile.TemporaryDirectory(prefix="test-executioner-workspace-")
    _TEMP_DIRS.append(temp_dir)
    workspace = ExecutionWorkspace(root=Path(temp_dir.name), commands=commands or {"test": "python -m pytest -q"})
    for item in files:
        workspace.write_file(item["path"], item["content"])
    payload = workspace.artifact_payload()
    workspace.close()
    return {"kind": "executioner_workspace", "payload": payload}


def test_execution_workspace_candidate_subdirs_keep_artifacts_stable() -> None:
    temp_dir = tempfile.TemporaryDirectory(prefix="test-executioner-run-workspace-")
    _TEMP_DIRS.append(temp_dir)
    workspace = ExecutionWorkspace(root=Path(temp_dir.name), commands={"test": "python -m pytest -q"})
    workspace.reset("candidates/candidate-a")
    workspace.write_file("README.md", "candidate a instructions")
    payload_a = workspace.artifact_payload()

    workspace.reset("candidates/candidate-b")
    workspace.write_file("README.md", "candidate b instructions")
    payload_b = workspace.artifact_payload()

    assert payload_a["workspace_root"] != payload_b["workspace_root"]
    assert workspace.read_file("README.md") == "candidate b instructions"

    prior_workspace = ExecutionWorkspace.from_artifact(payload_a)
    try:
        assert prior_workspace.read_file("README.md") == "candidate a instructions"
    finally:
        prior_workspace.close()
        workspace.close()


class _FakeResponse:
    def __init__(
        self,
        content: Any,
        *,
        usage_metadata: dict[str, int] | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.usage_metadata = usage_metadata or {"input_tokens": 1, "output_tokens": 1}
        self.response_metadata = response_metadata or {"model_name": "fake-model"}


class _FakeToolResponse(_FakeResponse):
    def __init__(self, tool_calls: list[dict[str, Any]], *, content: str = "") -> None:
        super().__init__(content, usage_metadata={"input_tokens": 3, "output_tokens": 2})
        self.tool_calls = tool_calls


class _FakeLangChainModel:
    def __init__(
        self,
        response: _FakeResponse | Exception,
        capture: dict[str, Any] | None = None,
        *,
        structured: bool = False,
        bound_tools: bool = False,
    ) -> None:
        self.response = response
        self.capture = capture
        self.structured = structured
        self.bound_tools = bound_tools

    def bind(self, **kwargs: Any) -> "_FakeLangChainModel":
        if self.capture is not None:
            self.capture.setdefault("bind_kwargs", {}).update(kwargs)
        return self

    def with_structured_output(self, schema: dict[str, Any], **kwargs: Any) -> "_FakeLangChainModel":
        if self.capture is not None:
            self.capture["structured_schema"] = schema
            self.capture["structured_kwargs"] = kwargs
        return _FakeLangChainModel(self.response, self.capture, structured=True)

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> "_FakeLangChainModel":
        if self.capture is not None:
            self.capture["tools"] = tools
            self.capture["tool_kwargs"] = kwargs
        return _FakeLangChainModel(self.response, self.capture, bound_tools=True)

    def invoke(self, messages):
        if self.capture is not None:
            self.capture["messages"] = messages
        if isinstance(self.response, Exception):
            raise self.response
        if not self.structured:
            return self.response
        content = self.response.content
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
            parsing_error = None
        except json.JSONDecodeError as exc:
            parsed = None
            parsing_error = exc
        return {"raw": self.response, "parsed": parsed, "parsing_error": parsing_error}


_TEST_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "properties": {},
}


def test_reasoning_effort_is_sent_to_langchain_for_openai_reasoning_models(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _FakeLangChainModel(_FakeResponse('{"ok": true}'))

    monkeypatch.setattr("model_client._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(ModelConfig(provider="openai", model="gpt-5-mini", reasoning_effort="medium"))
    client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert captured["model"] == "gpt-5-mini"
    assert captured["kwargs"]["model_provider"] == "openai"
    assert captured["kwargs"]["reasoning_effort"] == "medium"
    assert "temperature" not in captured["kwargs"]


def test_reasoning_effort_is_not_sent_for_non_openai_models(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _FakeLangChainModel(_FakeResponse('{"ok": true}'))

    monkeypatch.setattr("model_client._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(ModelConfig(provider="anthropic", model="claude-sonnet-test", reasoning_effort="medium"))
    client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert captured["kwargs"]["model_provider"] == "anthropic"
    assert captured["kwargs"]["temperature"] == 0.4
    assert "reasoning_effort" not in captured["kwargs"]


def test_reasoning_effort_none_is_not_sent_to_openai_compatible_non_reasoning_model(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _FakeLangChainModel(_FakeResponse('{"ok": true}'))

    monkeypatch.setattr("model_client._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(
        ModelConfig(
            provider="openai",
            model="moonshotai/Kimi-K2.6",
            base_url="https://api.deepinfra.com/v1/openai",
            api_key="deepinfra-test-key",
            reasoning_effort="none",
        )
    )
    client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert captured["model"] == "moonshotai/Kimi-K2.6"
    assert captured["kwargs"]["model_provider"] == "openai"
    assert captured["kwargs"]["base_url"] == "https://api.deepinfra.com/v1/openai"
    assert "reasoning_effort" not in captured["kwargs"]


def test_reasoning_effort_none_is_not_sent_to_gemini_3(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _FakeLangChainModel(_FakeResponse('{"ok": true}'))

    monkeypatch.setattr("model_client._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(
        ModelConfig(
            provider="gemini",
            model="gemini-3-flash-preview",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key="gemini-test-key",
            reasoning_effort="none",
        )
    )
    client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert captured["kwargs"]["temperature"] == 1.0
    assert "reasoning_effort" not in captured["kwargs"]


def test_gemini_provider_uses_openai_compatible_langchain_provider(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _FakeLangChainModel(_FakeResponse('{"ok": true}'), captured)

    monkeypatch.setattr("model_client._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(
        ModelConfig(
            provider="gemini",
            model="gemini-3.1-flash-lite",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key="gemini-test-key",
            reasoning_effort="medium",
            max_tokens=32000,
        )
    )
    payload, meta = client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert payload == {"ok": True}
    assert captured["model"] == "gemini-3.1-flash-lite"
    assert captured["kwargs"]["model_provider"] == "openai"
    assert captured["kwargs"]["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert captured["kwargs"]["api_key"] == "gemini-test-key"
    assert captured["kwargs"]["temperature"] == 1.0
    assert captured["kwargs"]["reasoning_effort"] == "medium"
    assert captured["kwargs"]["max_tokens"] == 32000
    assert captured["structured_schema"] == _structured_output_schema(_TEST_SCHEMA)
    assert captured["structured_kwargs"] == {"method": "json_schema", "include_raw": True}
    assert meta["provider"] == "gemini"


def test_xai_provider_uses_langchain_xai_and_extra_body_reasoning(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _FakeLangChainModel(_FakeResponse('{"ok": true}'), captured)

    monkeypatch.setattr("model_client._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(
        ModelConfig(
            provider="xai",
            model="grok-4.3",
            api_key="xai-test-key",
            reasoning_effort="high",
            max_tokens=4096,
        )
    )
    payload, meta = client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert payload == {"ok": True}
    assert captured["model"] == "grok-4.3"
    assert captured["kwargs"]["model_provider"] == "xai"
    assert captured["kwargs"]["api_key"] == "xai-test-key"
    assert captured["kwargs"]["extra_body"] == {"reasoning_effort": "high"}
    assert captured["kwargs"]["max_tokens"] == 4096
    assert "reasoning_effort" not in captured["kwargs"]
    assert meta["provider"] == "xai"


def test_complete_json_repairs_nul_hex_text_sequences(monkeypatch) -> None:
    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        return _FakeLangChainModel(
            _FakeResponse('{"word": "clich\\u0000e9", "range": "3\\u0000E2\\u000080\\u00009110 words"}')
        )

    monkeypatch.setattr("model_client._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(ModelConfig(model="gpt-5-mini", reasoning_effort="medium"))
    payload, meta = client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert payload == {"word": "cliché", "range": "3‑10 words"}
    assert meta["text_normalization_replacements"] == 2


def test_complete_json_error_preserves_raw_provider_output(monkeypatch) -> None:
    raw_output = '{"rationale": "bad\nnewline"}'

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        return _FakeLangChainModel(_FakeResponse(raw_output))

    monkeypatch.setattr("model_client._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(ModelConfig(model="gpt-5-mini", reasoning_effort="medium"))
    with pytest.raises(ProviderStructuredOutputError) as exc_info:
        client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert exc_info.value.raw_content == raw_output
    assert exc_info.value.parsing_error.lineno == 1


def test_complete_text_returns_plain_model_output(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        captured["kwargs"] = kwargs
        return _FakeLangChainModel(
            _FakeResponse("A small test output.", usage_metadata={"input_tokens": 3, "output_tokens": 4}),
            captured,
        )

    monkeypatch.setattr("model_client._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(ModelConfig(model="gpt-5-mini", reasoning_effort="medium"))
    output, meta = client.complete_text(system="Follow instructions.", user="Write.")

    assert output == "A small test output."
    assert captured["messages"][1] == ("user", "Write.")
    assert meta["output_tokens"] == 4


def test_langchain_tool_loop_uses_bind_tools_and_replays_outputs(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    response = _FakeToolResponse(
        [
            {
                "name": "write_file",
                "args": {"path": "README.md", "content": "hello"},
                "id": "call_2",
            }
        ]
    )

    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _FakeLangChainModel(response, captured)

    monkeypatch.setattr("model_client._load_init_chat_model", lambda: fake_init_chat_model)
    client = ModelClient(ModelConfig(provider="gemini", model="gemini-3-flash-preview", reasoning_effort="medium"))

    result = client.complete_with_tools(
        system="Use tools.",
        tools=[
            {
                "type": "function",
                "name": "write_file",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            }
        ],
        input_items=[
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "write"}]},
            {
                "type": "function_call",
                "name": "list_files",
                "call_id": "call_1",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": '{"ok": true, "files": []}'},
        ],
    )

    assert captured["model"] == "gemini-3-flash-preview"
    assert captured["tools"][0]["function"]["name"] == "write_file"
    assert captured["tool_kwargs"]["tool_choice"] == "auto"
    assert captured["tool_kwargs"]["parallel_tool_calls"] is False
    assert [type(message).__name__ for message in captured["messages"]] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
    ]
    assert result.output_items == [
        {
            "type": "function_call",
            "id": "call_2",
            "call_id": "call_2",
            "name": "write_file",
            "arguments": json.dumps({"path": "README.md", "content": "hello"}),
        }
    ]


def test_local_embedding_does_not_initialize_provider_model(monkeypatch) -> None:
    def fail_init_embeddings():
        raise AssertionError("remote embedding provider should not be initialized")

    monkeypatch.setattr("model_client._load_init_embeddings", fail_init_embeddings)

    client = ModelClient(ModelConfig(embedding_provider="local", embedding_model="local-hash-embedding"))
    vector, meta = client.embed("same text same text")

    assert len(vector) == 128
    assert meta["provider"] == "local"
    assert meta["model"] == "local-hash-embedding"


def test_accept_with_reject_signal_code_is_coerced_to_reject() -> None:
    verdict, route_code, subcodes = _coerce_gate_verdict(
        verdict=Verdict.ACCEPT,
        route_code=RouteCode.ACCEPT,
        subcodes=["proxy_strong", "weak_diagnostic_pressure"],
    )

    assert verdict == Verdict.REJECT
    assert route_code == RouteCode.REJECT_SEMANTIC_MISMATCH
    assert "weak_diagnostic_pressure" in subcodes


def test_model_invocation_errors_are_wrapped(monkeypatch) -> None:
    def fake_init_chat_model(model: str, **kwargs: Any) -> _FakeLangChainModel:
        return _FakeLangChainModel(TimeoutError("read timed out"))

    monkeypatch.setattr("model_client._load_init_chat_model", lambda: fake_init_chat_model)

    client = ModelClient(ModelConfig(request_timeout_seconds=12))

    try:
        client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)
    except Exception as exc:
        assert type(exc).__name__ == "ProviderError"
        assert "structured model invocation failed" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_complete_json_omits_codex_native_json_schema(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": time.time() + 3600,
                    "account_id": "acct_test",
                }
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, Any]] = []

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return self.body

    def fake_urlopen(request, timeout):
        requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": request.data})
        return FakeResponse(
            (
                "data: "
                + json.dumps({"type": "response.output_text.delta", "delta": json.dumps({"ok": True})})
                + "\n\n"
                + "data: "
                + json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "model": "gpt-5.5",
                            "usage": {"input_tokens": 3, "output_tokens": 2},
                        },
                    }
                )
                + "\n\n"
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(
        ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file, reasoning_effort="medium", max_tokens=32000)
    )
    payload, meta = client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert payload == {"ok": True}
    assert meta["provider"] == "codex"
    assert meta["model"] == "gpt-5.5"
    body = json.loads(requests[0]["body"].decode("utf-8"))
    assert body["store"] is False
    assert body["stream"] is True
    assert "max_output_tokens" not in body
    assert "max_tokens" not in body
    assert body["reasoning"] == {"effort": "medium"}
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": '{"task": "test"}'}],
        }
    ]
    assert "tools" not in body
    assert "tool_choice" not in body
    assert "parallel_tool_calls" not in body
    assert "include" not in body
    assert "prompt_cache_key" not in body
    assert "text" not in body
    assert "CODEX JSON OUTPUT CONTRACT" in body["instructions"]
    assert json.dumps(_structured_output_schema(_TEST_SCHEMA), sort_keys=True, separators=(",", ":")) in body["instructions"]


def test_codex_client_parses_content_part_done_structured_json(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps({"tokens": {"access_token": "access-token", "refresh_token": "refresh-token", "expires_at": time.time() + 3600}}),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return (
                "data: "
                + json.dumps(
                    {
                        "type": "response.content_part.done",
                        "item_id": "msg_1",
                        "part": {"type": "output_text", "text": json.dumps({"ok": True})},
                    }
                )
                + "\n\n"
                + 'data: {"type":"response.completed","response":{"model":"gpt-5.5","usage":{}}}\n\n'
            ).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file))
    payload, _ = client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert payload == {"ok": True}


def test_codex_client_parses_output_item_done_structured_json(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps({"tokens": {"access_token": "access-token", "refresh_token": "refresh-token", "expires_at": time.time() + 3600}}),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return (
                "data: "
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "id": "msg_1",
                            "type": "message",
                            "content": [{"type": "output_text", "text": json.dumps({"ok": True})}],
                        },
                    }
                )
                + "\n\n"
                + 'data: {"type":"response.completed","response":{"model":"gpt-5.5","usage":{}}}\n\n'
            ).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file))
    payload, _ = client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert payload == {"ok": True}


def test_codex_client_parses_completed_output_json(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps({"tokens": {"access_token": "access-token", "refresh_token": "refresh-token", "expires_at": time.time() + 3600}}),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return (
                "data: "
                + json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "model": "gpt-5.5",
                            "output": [{"type": "output_json", "json": {"ok": True}}],
                            "usage": {"input_tokens": 3, "output_tokens": 2},
                        },
                    }
                )
                + "\n\n"
            ).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file))
    payload, meta = client.complete_json(system="Return JSON only.", user='{"task": "test"}', schema=_TEST_SCHEMA)

    assert payload == {"ok": True}
    assert meta["input_tokens"] == 3
    assert meta["output_tokens"] == 2


def test_codex_structured_schema_normalizes_anyof_without_mutating_source() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ]
            }
        },
    }
    original = json.loads(json.dumps(schema))

    compiled = _codex_structured_output_schema(schema)

    assert schema == original
    assert compiled["type"] == "object"
    assert compiled["additionalProperties"] is False
    assert compiled["required"] == ["value"]
    assert compiled["properties"]["value"]["type"] == ["string", "null"]
    assert "anyOf" not in compiled["properties"]["value"]


def test_codex_client_refreshes_expired_tokens(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "old-token",
                    "refresh_token": "refresh-token",
                    "expires_at": time.time() - 1,
                }
            }
        ),
        encoding="utf-8",
    )
    requests: list[dict[str, Any]] = []

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return self.body

    def fake_urlopen(request, timeout):
        requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": request.data})
        if request.full_url == CodexClient.token_endpoint:
            return FakeResponse(json.dumps({"access_token": "new-token", "expires_in": 3600}).encode("utf-8"))
        return FakeResponse(
            (
                'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
                'data: {"type":"response.completed","response":{"model":"gpt-5.5","usage":{}}}\n\n'
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file))
    output, _ = client.complete_text(system="Say hi.", user="Hi.")

    assert output == "hello"
    assert requests[0]["url"] == CodexClient.token_endpoint
    assert b"grant_type=refresh_token" in requests[0]["body"]
    assert requests[1]["headers"]["Authorization"] == "Bearer new-token"
    updated_auth = json.loads(auth_file.read_text(encoding="utf-8"))
    assert updated_auth["tokens"]["access_token"] == "new-token"
    assert updated_auth["tokens"]["refresh_token"] == "refresh-token"


def test_codex_client_uses_env_auth_cache_for_refresh(tmp_path, monkeypatch) -> None:
    source_auth = tmp_path / "source-auth.json"
    cache_auth = tmp_path / "cache" / "codex-auth.json"
    source_auth.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "source-token",
                    "refresh_token": "source-refresh",
                    "expires_at": time.time() - 1,
                }
            }
        ),
        encoding="utf-8",
    )
    cache_auth.parent.mkdir()
    cache_auth.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "cache-token",
                    "refresh_token": "cache-refresh",
                    "expires_at": time.time() - 1,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_AUTH_CACHE_FILE", str(cache_auth))
    requests: list[dict[str, Any]] = []

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return self.body

    def fake_urlopen(request, timeout):
        requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": request.data})
        if request.full_url == CodexClient.token_endpoint:
            return FakeResponse(json.dumps({"access_token": "new-cache-token", "expires_in": 3600}).encode("utf-8"))
        return FakeResponse(
            (
                'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
                'data: {"type":"response.completed","response":{"model":"gpt-5.5","usage":{}}}\n\n'
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=source_auth))
    output, _ = client.complete_text(system="Say hi.", user="Hi.")

    assert output == "hello"
    assert b"refresh_token=cache-refresh" in requests[0]["body"]
    assert requests[1]["headers"]["Authorization"] == "Bearer new-cache-token"
    source_payload = json.loads(source_auth.read_text(encoding="utf-8"))
    cache_payload = json.loads(cache_auth.read_text(encoding="utf-8"))
    assert source_payload["tokens"]["access_token"] == "source-token"
    assert cache_payload["tokens"]["access_token"] == "new-cache-token"


def test_codex_client_rereads_shared_cache_after_stale_refresh_token(tmp_path, monkeypatch) -> None:
    cache_auth = tmp_path / "cache" / "codex-auth.json"
    cache_auth.parent.mkdir()
    cache_auth.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "expired-token",
                    "refresh_token": "stale-refresh",
                    "expires_at": time.time() - 1,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_AUTH_CACHE_FILE", str(cache_auth))
    refresh_calls = 0

    def fake_refresh(self):
        nonlocal refresh_calls
        refresh_calls += 1
        cache_auth.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": "fresh-from-peer",
                        "refresh_token": "rotated-refresh",
                        "expires_at": time.time() + 3600,
                    }
                }
            ),
            encoding="utf-8",
        )
        raise ProviderError('Codex token refresh failed 400: {"error":"invalid_grant"}')

    monkeypatch.setattr(CodexClient, "_refresh_access_token", fake_refresh)

    client = CodexClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=tmp_path / "source-auth.json"))

    assert client._access_token() == "fresh-from-peer"
    assert refresh_calls == 1


def test_codex_client_tolerates_incomplete_read_after_completed_event(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": time.time() + 3600,
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            raise IncompleteRead(
                (
                    'data: {"type":"response.output_text.delta","delta":"done"}\n\n'
                    'data: {"type":"response.completed","response":{"model":"gpt-5.5","usage":{}}}\n\n'
                ).encode("utf-8"),
                128,
            )

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file))
    output, _ = client.complete_text(system="Say done.", user="Go.")

    assert output == "done"


def test_codex_client_rejects_incomplete_read_before_completed_event(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": time.time() + 3600,
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            raise IncompleteRead(b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n', 128)

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file))
    with pytest.raises(ProviderError, match="incomplete HTTP read before response.completed"):
        client.complete_text(system="Say partial.", user="Go.")


def test_codex_client_times_out_idle_stream_without_completed_event(tmp_path, monkeypatch) -> None:
    auth_file = tmp_path / "codex-auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": time.time() + 3600,
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def readline(self):
            raise socket.timeout("timed out")

    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 0.25
        return clock["now"]

    monkeypatch.setattr("agents.time.monotonic", fake_monotonic)
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=auth_file, request_timeout_seconds=1))
    with pytest.raises(ProviderError, match="Codex stream idle timeout"):
        client.complete_text(system="Say partial.", user="Go.")


def test_codex_sse_reader_allows_stream_past_request_timeout_when_bytes_arrive(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self) -> None:
            self.calls = 0

        def readline(self):
            self.calls += 1
            if self.calls <= 8:
                return b'data: {"type":"response.output_text.delta","delta":"."}\n\n'
            return b'data: {"type":"response.completed","response":{"model":"gpt-5.5","usage":{}}}\n\n'

    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 0.25
        return clock["now"]

    monkeypatch.setattr("agents.time.monotonic", fake_monotonic)

    raw = _read_codex_sse(FakeResponse(), timeout_seconds=1, idle_timeout_seconds=1)

    assert "response.completed" in raw


def test_codex_sse_reader_emits_wait_and_payload_events_after_read_timeout() -> None:
    class FakeResponse:
        def __init__(self) -> None:
            self.calls = 0

        def readline(self):
            self.calls += 1
            if self.calls == 1:
                raise socket.timeout("timed out")
            return (
                'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.5"}}\n\n'
                'data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.5","usage":{"input_tokens":3,"output_tokens":2}}}\n\n'
            ).encode("utf-8")

    events: list[dict[str, Any]] = []

    raw = _read_codex_sse(FakeResponse(), timeout_seconds=1, stream_event_callback=events.append)

    assert "response.completed" in raw
    assert [event["stream_event"] for event in events] == [
        "stream_open",
        "read_wait",
        "response.created",
        "response.completed",
    ]
    assert events[-1]["response_id"] == "resp_1"
    assert events[-1]["input_tokens"] == 3
    assert events[-1]["output_tokens"] == 2


def test_codex_stream_idle_timeout_defaults_to_request_timeout(monkeypatch) -> None:
    monkeypatch.delenv("CODEX_STREAM_IDLE_TIMEOUT_SECONDS", raising=False)

    assert _codex_stream_idle_timeout_seconds(600) == 600


def test_codex_client_missing_auth_file_has_clear_error(tmp_path) -> None:
    client = ModelClient(ModelConfig(provider="codex", model="gpt-5.5", auth_file=tmp_path / "missing.json"))

    with pytest.raises(ProviderError, match="Codex auth file not found"):
        client.complete_text(system="Say hi.", user="Hi.")


def _code_design() -> DesignBrief:
    cell = TaxonomyCell(case_type="proxy_strong", difficulty=4, scenario="edge")
    return DesignBrief.create(
        design_id="design-1",
        cell=cell,
        target_ability="fault_localization",
        target_environment="single_turn_debug_with_test",
        design_intent="Create a compact benchmark around a realistic stateful debugging failure.",
        environment_premise={
            "product_context": "billing reconciliation worker",
            "codebase_shape": "parser, reconciliation engine, and tests",
            "state_model": "invoice rows flow through parse, normalize, summarize",
            "core_invariant": "refunds reduce revenue in the original period",
            "failure_surface": "partial refunds inflate the summary",
            "tempting_wrong_fix": "round or clamp the displayed total",
            "actual_causal_region": "refund normalization before aggregation",
            "required_depth": "requires tracing value semantics across modules",
        },
        runtime_requirements={
            "kind": "filesystem_task",
            "execution": {"mode": "task_image", "base_image": "python:3.11-slim", "os": "linux", "arch": "amd64"},
            "language": {"name": "python", "version": "3.11+"},
            "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
            "commands": {"test": "python -m pytest -q"},
            "network": "disabled_during_eval",
        },
        environment_artifact_spec={"kind": "executioner_workspace"},
        failure_mode_family="misleading aggregate caused by upstream normalization",
        diagnostic_pressure=["misleading downstream symptom"],
        why_weak_agents_fail=["they patch only the formatter"],
        tempting_shallow_solutions=["clamp the output total"],
        success_evidence_required=["preserves refund invariant"],
        minimum_depth_requirements=["multi-file causal trace"],
        forbidden_shortcuts=["one-line display mask"],
        non_goals=["missing import"],
    )


def _code_candidate(design: DesignBrief) -> CandidateSample:
    return CandidateSample(
        id="candidate-1",
        design_id=design.id,
        content_hash="hash",
        cell=design.cell,
        agent_artifact={
            "benchmark_case": {
                "prompt": "Debug the reconciliation worker.",
                "setup": "Run pytest.",
                "inputs": {},
                "environment": {"runtime": "python"},
            },
            "environment_artifact": _executioner_artifact(
                [
                    {"path": "billing/parser.py", "content": "def parse_row(row):\n    return dict(row)\n"},
                    {
                        "path": "billing/reconcile.py",
                        "content": "from billing.parser import parse_row\n\n\ndef summarize(rows):\n    totals = {}\n    for row in rows:\n        parsed = parse_row(row)\n        totals[parsed['period']] = totals.get(parsed['period'], 0) + abs(parsed['amount'])\n    return totals\n",
                    },
                    {"path": "README.md", "content": "Patch the service without editing tests.\n"},
                    {
                        "path": "tests/test_reconcile.py",
                        "content": "from billing.reconcile import summarize\n\n\ndef test_refund_reduces_total():\n    assert summarize([{'period': '2026-03', 'amount': 100}, {'period': '2026-03', 'amount': -25}]) == {'2026-03': 75}\n",
                    },
                ]
            ),
            "runtime_requirements": {
                "kind": "filesystem_task",
                "execution": {"mode": "task_image", "base_image": "python:3.11-slim", "os": "linux", "arch": "amd64"},
                "language": {"name": "python", "version": "3.11+"},
                "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
                "commands": {"test": "python -m pytest -q"},
                "network": "disabled_during_eval",
            },
        },
        judge_artifact={
            "score_x": {"score_type": "hard_checks_plus_rubric", "dimensions": [{"name": "causal_fix", "weight": 1.0}]},
            "proxy_claim": "The benchmark proxies debugging ability by requiring an invariant-preserving fix.",
            "diagnostic_pressure": ["misleading downstream symptom"],
            "scoring_contract": {"credit": ["fixes refund invariant"], "penalties": ["display-only mask"]},
            "leakage_risks": ["visible assertion may invite a clamp"],
            "known_limits": ["single case"],
            "coverage_tags": ["stateful_debugging"],
            "negative_controls": [{"output": "Clamp the total.", "should_fail_because": "masks the bug"}],
        },
        ability_z={"name": "fault_localization"},
        environment_y={"name": "single_turn_debug_with_test"},
        difficulty=design.cell.difficulty,
        case_type=design.cell.case_type,
    )


def _attack_report(candidate_id: str) -> AdversaryReport:
    return AdversaryReport(
        candidate_id=candidate_id,
        revision_disposition="revise",
        attack_summary="The task is too local and can be solved by clamping the aggregate.",
        cheap_pass_strategy="Change abs(parsed['amount']) to parsed['amount'] without understanding period invariants.",
        proxy_damage="The score would mostly reflect matching the visible assertion.",
        survival_requirements=["add cross-period refund pressure", "remove README boilerplate"],
    )


class _PatchClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.user_payload: dict[str, Any] | None = None

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], temperature: float = 0.4):
        assert temperature == 0.2
        assert "RevisionGenerator" in system
        assert "You revise benchmark artifacts, not solver submissions" in system
        assert "Return a revision patch JSON object only" in system
        assert "unmodified starter workspace must still fail" in system
        assert "Do not solve the agent-facing bug" in system
        assert "Benchmark Case Generator" not in system
        assert "required_revision_patch_shape" in user
        assert "benchmark_invariants" in user
        assert "do not solve the bug under test" in user
        self.user_payload = json.loads(user)
        assert self.user_payload["required_revision_patch_shape"]["environment_ops"][0]["op"] == "edit_file"
        return self.payload, {
            "provider": "test",
            "model": "fake",
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 1,
            "cost_usd": 0.0,
        }


class _GenerateClient:
    def __init__(self, *, require_contract: bool = True) -> None:
        self.require_contract = require_contract
        self.system = ""
        self.user_payload: dict[str, Any] | None = None

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], temperature: float = 0.4):
        if self.require_contract:
            assert "DESIGN IMPLEMENTATION CONTRACT" in system
        self.system = system
        self.user_payload = json.loads(user)
        return {
            "agent_artifact": {
                "benchmark_case": {
                    "prompt": "Debug the provided workspace.",
                    "setup": "Run pytest.",
                    "inputs": {},
                    "environment": {"runtime": "python"},
                },
                "runtime_requirements": {
                    "kind": "filesystem_task",
                    "execution": {"mode": "task_image", "base_image": "python:3.11-slim", "os": "linux", "arch": "amd64"},
                    "language": {"name": "python", "version": "3.11+"},
                    "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
                    "commands": {"test": "python -m pytest -q"},
                    "network": "disabled_during_eval",
                },
            },
            "judge_artifact": {
                "score_x": {
                    "score_type": "hard_checks_plus_rubric",
                    "dimensions": [
                        {
                            "name": "causal_fix",
                            "weight": 1.0,
                            "high_score_criterion": "Causal fix.",
                            "low_score_criterion": "Shallow fix.",
                        }
                    ],
                },
                "proxy_claim": "This case proxies debugging ability with a visible workspace and judge-facing scoring criteria.",
                "diagnostic_pressure": ["misleading symptom", "cross-module invariant"],
                "scoring_contract": {"credit": ["causal fix"], "penalties": ["test edits"]},
                "leakage_risks": ["Visible tests may invite shallow fixes."],
                "known_limits": ["Small synthetic workspace."],
                "coverage_tags": ["debugging"],
                "negative_controls": [{"output": "edit tests", "should_fail_because": "weakens benchmark"}],
            },
            "ability_z": {"name": "fault_localization"},
            "environment_y": {"name": "single_turn_debug_with_test"},
        }, {
            "provider": "test",
            "model": "fake",
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 1,
            "cost_usd": 0.0,
        }


class _ToolGenerateClient:
    config = ModelConfig(provider="codex", model="gpt-5.5", reasoning_effort="none")

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.user_payload: dict[str, Any] | None = None

    def supports_function_tools(self) -> bool:
        return True

    @contextmanager
    def stream_context(self, context: dict[str, Any]):
        yield

    def complete_with_tools(self, *, system: str, input_items: list[dict[str, Any]], tools: list[dict[str, Any]]) -> CodexResponse:
        self.calls.append({"system": system, "input_items": list(input_items), "tools": tools})
        if self.user_payload is None and input_items:
            content = input_items[0].get("content")
            if isinstance(content, str):
                self.user_payload = json.loads(content)
            elif isinstance(content, list):
                self.user_payload = json.loads("".join(part.get("text", "") for part in content if isinstance(part, dict)))
        if len(self.calls) == 1:
            return CodexResponse(
                content="",
                usage_metadata={"input_tokens": 10, "output_tokens": 5},
                response_metadata={"model_name": "gpt-5.5"},
                output_items=[
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "write_file",
                        "arguments": json.dumps({"path": "README.md", "content": "# Debug task\nRun pytest.\n"}),
                    },
                    {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_2",
                        "name": "write_file",
                        "arguments": json.dumps({"path": "app.py", "content": "def total(items):\n    return sum(items) + 1\n"}),
                    },
                    {
                        "type": "function_call",
                        "id": "fc_3",
                        "call_id": "call_3",
                        "name": "write_file",
                        "arguments": json.dumps(
                            {"path": "tests/test_app.py", "content": "from app import total\n\n\ndef test_total():\n    assert total([1, 2]) == 3\n"}
                        ),
                    },
                ],
            )
        return CodexResponse(
            content="",
            usage_metadata={"input_tokens": 20, "output_tokens": 8},
            response_metadata={"model_name": "gpt-5.5"},
            output_items=[
                {
                    "type": "function_call",
                    "id": "fc_4",
                    "call_id": "call_4",
                    "name": "finalize_candidate",
                    "arguments": json.dumps(
                        {
                            "benchmark_case": {
                                "prompt": "Fix the failing Python workspace.",
                                "setup": "Run pytest.",
                                "inputs": {"symptom": "failing total test"},
                                "environment": {"language": "python"},
                            },
                            "runtime_requirements": {
                                "kind": "filesystem_task",
                                "execution": {
                                    "mode": "task_image",
                                    "base_image": "python:3.11-slim",
                                    "os": "linux",
                                    "arch": "amd64",
                                },
                                "language": {"name": "python", "version": "3.11+"},
                                "dependencies": {"policy": "stdlib_plus_runner", "packages": ["pytest"]},
                                "commands": {"test": "python -m pytest -q"},
                                "network": "disabled_during_eval",
                            },
                            "workspace_commands": {"test": "python -m pytest -q"},
                            "judge_artifact": {
                                "score_x": {
                                    "score_type": "hard_checks_plus_rubric",
                                    "range": [0, 1],
                                    "dimensions": [
                                        {
                                            "name": "fix_correctness",
                                            "weight": 1.0,
                                            "high_score_criterion": "The patch makes the failing behavior correct without editing tests.",
                                            "low_score_criterion": "The patch does not address the failing behavior or edits tests.",
                                        }
                                    ],
                                },
                                "private_root_cause": "The implementation adds one extra unit so the total is incorrect.",
                                "expected_fix_properties": ["The implementation should sum the provided items without adding an extra unit."],
                                "hidden_failure_modes": ["Changing the visible test instead of the implementation should fail."],
                                "shallow_solution_traps": ["Hard-coding the visible input would not generalize."],
                                "candidate_visibility_boundaries": ["Do not reveal the extra-unit issue in candidate-facing materials."],
                                "proxy_claim": "This small failing workspace checks whether an agent can inspect a local Python project and make a bounded causal debugging fix.",
                                "diagnostic_pressure": ["The failing assertion requires inspecting implementation behavior.", "The workspace is small enough to run locally but still requires a code change."],
                                "scoring_contract": {
                                    "credit": ["fix implementation"],
                                    "penalties": ["edit tests"],
                                    "uncertainty_policy": None,
                                },
                                "leakage_risks": ["The visible test exposes expected behavior for one input."],
                                "known_limits": ["This fixture is intentionally small and does not test large-project navigation."],
                                "coverage_tags": ["python"],
                                "negative_controls": [
                                    {"output": "Change the test expected value to four.", "should_fail_because": "The benchmark requires fixing implementation behavior."}
                                ],
                            },
                            "ability_z": {"name": "fault_localization", "sub_abilities": []},
                            "environment_y": {"name": "single_turn_debug_with_test", "assumptions": []},
                        }
                    ),
                }
            ],
        )


class _DesignClient:
    def __init__(self) -> None:
        self.system = ""
        self.user_payload: dict[str, Any] | None = None

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], temperature: float = 0.4):
        self.system = system
        self.user_payload = json.loads(user)
        return {"designs": []}, {
            "provider": "test",
            "model": "fake",
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 1,
            "cost_usd": 0.0,
        }


class _CaptureJsonClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.user_payload: dict[str, Any] | None = None
        self.schema: dict[str, Any] | None = None
        self.system: str | None = None

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], temperature: float = 0.4):
        self.user_payload = json.loads(user)
        self.schema = schema
        self.system = system
        return self.payload, {
            "provider": "test",
            "model": "fake",
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 1,
            "cost_usd": 0.0,
        }


def test_design_retry_payload_explains_runtime_image_mode() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    client = _DesignClient()
    designer = Designer(client, domain)

    designer.design(
        run_id="run",
        target_n=1,
        coverage_snapshot={},
        retry_route_code=RouteCode.REJECT_CRITERIA_MISMATCH,
        retry_subcodes=["unsupported_runtime_requirements"],
    )

    assert "never a combined value" in client.system
    rejection = client.user_payload["prior_design_rejection"]
    assert rejection["subcodes"] == ["unsupported_runtime_requirements"]
    assert any("task_image/container" in item for item in rejection["retry_guidance"])


def test_design_payload_includes_user_instruction() -> None:
    domain = load_domain("domains/benchmark_haiku.yaml")
    client = _DesignClient()
    designer = Designer(client, domain)

    designer.design(
        run_id="run",
        target_n=1,
        coverage_snapshot={},
        instruction="Make the case about late-autumn layoffs without direct job-loss language.",
    )

    assert client.user_payload["user_instruction"] == "Make the case about late-autumn layoffs without direct job-loss language."
    assert "instruction_policy" in client.user_payload


def test_generation_retry_payload_includes_actionable_code_debug_guidance() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    client = _GenerateClient()
    generator = SampleGenerator(client, domain)

    generator.generate(
        run_id="run",
        design=design,
        attempt=2,
        retry_route_code=RouteCode.REJECT_SCHEMA,
        retry_subcodes=["workspace_tests_do_not_reproduce_failure", "answer_leak_fix_instruction"],
    )

    rejection = client.user_payload["prior_generation_rejection"]
    assert rejection["subcodes"] == ["workspace_tests_do_not_reproduce_failure", "answer_leak_fix_instruction"]
    assert any("starter code has at least one deterministic failing pytest assertion" in item for item in rejection["retry_guidance"])
    assert any("tells the evaluated agent what fix to make" in item for item in rejection["retry_guidance"])
    assert client.user_payload["design_brief"]["runtime_requirements"]["kind"] == "filesystem_task"
    assert client.user_payload["example_output"]["agent_artifact"]["runtime_requirements"]["commands"]["test"] == "python -m pytest -q"


def test_code_debug_structured_schema_requires_workspace_and_runtime() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    schema = _generation_output_schema(domain)
    strict_schema = _structured_output_schema(schema)

    required = schema["properties"]["agent_artifact"]["required"]
    runtime = schema["$defs"]["runtime_requirements"]
    environment = schema["$defs"]["environment_artifact"]

    assert "benchmark_case" in required
    assert "runtime_requirements" in required
    assert "environment_artifact" in required
    assert runtime["properties"]["kind"]["enum"] == ["filesystem_task"]
    assert runtime["properties"]["execution"]["properties"]["mode"]["enum"] == ["task_image", "container"]
    assert "test" in runtime["properties"]["commands"]["required"]
    assert environment["properties"]["kind"]["enum"] == ["executioner_workspace"]
    assert environment["properties"]["payload"]["properties"]["files"]["minItems"] == 3
    benchmark_case = schema["$defs"]["benchmark_case"]
    assert "setup" in benchmark_case["required"]
    assert "inputs" in benchmark_case["required"]
    assert "environment" in benchmark_case["required"]
    assert benchmark_case["properties"]["setup"]["type"] == "string"
    assert benchmark_case["properties"]["inputs"]["type"] == "object"
    assert benchmark_case["properties"]["environment"]["type"] == "object"
    assert schema["properties"]["ability_z"]["properties"]["name"]["enum"] == domain.abilities
    assert schema["properties"]["environment_y"]["properties"]["name"]["enum"] == domain.environments
    assert "$schema" not in strict_schema
    assert _object_schemas_with_open_additional_properties(strict_schema) == []
    assert _object_schemas_with_optional_properties(strict_schema) == []


def _object_schemas_with_open_additional_properties(schema: Any, path: str = "$") -> list[str]:
    if not isinstance(schema, dict):
        return []
    paths: list[str] = []
    if _test_schema_has_object_type(schema) and schema.get("additionalProperties") is not False:
        paths.append(path)
    for key in ("properties", "$defs", "definitions"):
        children = schema.get(key)
        if isinstance(children, dict):
            for name, child in children.items():
                paths.extend(_object_schemas_with_open_additional_properties(child, f"{path}.{key}.{name}"))
    items = schema.get("items")
    if isinstance(items, dict):
        paths.extend(_object_schemas_with_open_additional_properties(items, f"{path}.items"))
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for index, variant in enumerate(variants):
                paths.extend(_object_schemas_with_open_additional_properties(variant, f"{path}.{key}[{index}]"))
    return paths


def _object_schemas_with_optional_properties(schema: Any, path: str = "$") -> list[str]:
    if not isinstance(schema, dict):
        return []
    paths: list[str] = []
    properties = schema.get("properties")
    if _test_schema_has_object_type(schema) and isinstance(properties, dict):
        missing = set(properties) - set(schema.get("required", []))
        if missing:
            paths.append(f"{path}: {sorted(missing)}")
    for key in ("properties", "$defs", "definitions"):
        children = schema.get(key)
        if isinstance(children, dict):
            for name, child in children.items():
                paths.extend(_object_schemas_with_optional_properties(child, f"{path}.{key}.{name}"))
    items = schema.get("items")
    if isinstance(items, dict):
        paths.extend(_object_schemas_with_optional_properties(items, f"{path}.items"))
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for index, variant in enumerate(variants):
                paths.extend(_object_schemas_with_optional_properties(variant, f"{path}.{key}[{index}]"))
    return paths


def _test_schema_has_object_type(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    return schema_type == "object" or (isinstance(schema_type, list) and "object" in schema_type)


def test_generation_accepts_structured_envelope() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    client = _GenerateClient()
    generator = SampleGenerator(client, domain)
    envelope = GenerationEnvelope.from_design(
        design,
        envelope_id="billing-ledger-v1",
        domain_ref="domains/benchmark_code_debug.yaml",
        seed_context={"seed_family": "billing-ledger"},
    )

    candidate, _ = generator.generate_from_envelope(run_id="run", envelope=envelope, attempt=1)

    assert set(client.user_payload["generation_envelope"]) == {"design"}
    assert "generator_policy" not in client.user_payload["generation_envelope"]
    assert "tags" not in client.user_payload["generation_envelope"]
    assert "seed_context" not in client.user_payload["generation_envelope"]
    assert "id" not in client.user_payload["generation_envelope"]["design"]
    assert "content_hash" not in client.user_payload["generation_envelope"]["design"]
    assert "parent_design_batch_id" not in client.user_payload["generation_envelope"]["design"]
    assert "content_hash" not in client.user_payload["design_brief"]
    assert "runner_context" not in client.user_payload
    assert candidate.provenance["generation_envelope_id"] == "billing-ledger-v1"
    assert "generator_variant" not in candidate.provenance


def test_code_generation_can_author_workspace_with_tools() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    client = _ToolGenerateClient()

    candidate, meta = SampleGenerator(client, domain).generate(run_id="run", design=design, attempt=1)

    assert meta["workspace_tool_loop"] is True
    assert meta["workspace_tool_calls"] == 4
    assert meta["workspace_file_count"] == 3
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {"write_file", "read_file", "list_files", "finalize_candidate"}
    finalize_tool = next(tool for tool in client.calls[0]["tools"] if tool["name"] == "finalize_candidate")
    assert finalize_tool["strict"] is False
    assert "payload_json" not in finalize_tool["parameters"]["properties"]
    assert set(finalize_tool["parameters"]["required"]) == {
        "benchmark_case",
        "runtime_requirements",
        "workspace_commands",
        "judge_artifact",
        "ability_z",
        "environment_y",
    }
    assert finalize_tool["parameters"]["properties"]["benchmark_case"]["type"] == "object"
    assert client.user_payload["finalize_candidate_json_schema"]["properties"]["benchmark_case"]["properties"]["environment"]["type"] == "object"
    assert "environment" in client.user_payload["finalize_candidate_json_schema"]["properties"]["benchmark_case"]["required"]
    second_input = client.calls[1]["input_items"]
    assert any(item.get("type") == "function_call" and item.get("name") == "write_file" for item in second_input)
    assert any(item.get("type") == "function_call_output" and item.get("call_id") == "call_1" for item in second_input)
    assert not any(item.get("type") == "reasoning" for item in second_input)
    workspace = candidate.agent_artifact.environment_artifact
    assert workspace is not None
    assert workspace.kind == "executioner_workspace"
    files = {item["path"] for item in workspace.payload["files"]}
    assert "app.py" in files
    assert "tests/test_app.py" in files
    assert workspace.payload["commands"]["test"] == "python -m pytest -q"


def test_code_generation_does_not_use_workspace_tools_by_default() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    client = _GenerateClient()

    candidate, meta = SampleGenerator(client, domain).generate(run_id="run", design=design, attempt=1)

    assert client.user_payload is not None
    assert candidate.agent_artifact.benchmark_case["prompt"] == "Debug the provided workspace."


def test_workspace_tool_generation_enforces_required_files() -> None:
    design = _code_design()
    design.environment_artifact_spec = {"required_files": ["README.md", "app.py", "tests/test_app.py", "missing.py"]}
    domain = load_domain("domains/benchmark_code_debug.yaml")
    client = _ToolGenerateClient()

    with pytest.raises(ProviderError, match="missing.py"):
        SampleGenerator(client, domain).generate(run_id="run", design=design, attempt=1)


def test_gate_prompts_keep_quality_and_rubric_boundaries() -> None:
    assert "Judge benchmark quality independently of the rubric" in QualityGate.system_prompt
    assert "RubricGate handles rubric validity" in QualityGate.system_prompt
    assert "Validate the rubric for this exact benchmark artifact" in RubricGate.system_prompt
    assert "QualityGate handles benchmark quality outside the rubric" in RubricGate.system_prompt
    assert "toy-shaped" not in RubricGate.system_prompt


def test_gate_prompt_candidate_view_omits_pipeline_bookkeeping() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    candidate = _code_candidate(design)
    candidate.id = "candidate-1-rev"
    candidate.provenance["revision_of"] = "candidate-1"
    client = _CaptureJsonClient(
        {
            "verdict": "accept",
            "route_code": "accept",
            "subcodes": [],
            "evidence": [],
            "rationale": "The benchmark is acceptable.",
        }
    )

    QualityGate(client, domain).validate(candidate)

    assert client.user_payload is not None
    prompt_candidate = client.user_payload["candidate"]
    assert "output" not in prompt_candidate
    assert "id" not in prompt_candidate
    assert "design_id" not in prompt_candidate
    assert "content_hash" not in prompt_candidate
    assert "provenance" not in prompt_candidate
    assert "revision_of" not in json.dumps(prompt_candidate)
    assert prompt_candidate["agent_visible_artifact"]["environment_artifact"]["payload"]["files"]
    assert "evaluator_private_context" not in prompt_candidate
    assert prompt_candidate["benchmark_context"]["ability_z"] == candidate.ability_z
    assert prompt_candidate["benchmark_context"]["environment_y"] == candidate.environment_y
    assert prompt_candidate == candidate_prompt_view(candidate, include_evaluator_private=False)


def test_adversary_prompt_candidate_view_omits_pipeline_bookkeeping() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    candidate = _code_candidate(design)
    candidate.id = "candidate-1-rev"
    candidate.provenance["revision_of"] = "candidate-1"
    client = _CaptureJsonClient(
        {
            "revision_disposition": "pass",
            "disposition_rationale": "No blocking attack found.",
            "attack_summary": "",
            "attacks": [],
            "cheap_pass_strategy": "",
            "proxy_damage": "",
            "survival_requirements": [],
        }
    )

    Adversary(client, domain).attack(candidate, design)

    assert client.user_payload is not None
    prompt_candidate = client.user_payload["candidate"]
    assert "output" not in prompt_candidate
    assert "id" not in prompt_candidate
    assert "design_id" not in prompt_candidate
    assert "content_hash" not in prompt_candidate
    assert "provenance" not in prompt_candidate
    assert "revision_of" not in json.dumps(prompt_candidate)
    assert prompt_candidate["agent_visible_artifact"]["environment_artifact"]["payload"]["files"]
    assert prompt_candidate["evaluator_private_context"]["judge_artifact"]["negative_controls"]
    assert "expected_fix_properties" not in prompt_candidate["evaluator_private_context"]["judge_artifact"]
    assert client.user_payload["design_brief"]["target_ability"] == design.target_ability
    assert "environment_premise" not in client.user_payload["design_brief"]
    assert "id" not in client.user_payload["design_brief"]
    assert "content_hash" not in client.user_payload["design_brief"]
    assert "parent_design_batch_id" not in client.user_payload["design_brief"]
    assert "required_json_shape" not in client.user_payload
    assert "general_probe_principles" not in client.user_payload["attack_surface"]
    assert "attack_types" in client.user_payload["attack_surface"]


def test_adversary_binary_disposition_mode_removes_nuke_state() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    candidate = _code_candidate(design)
    client = _CaptureJsonClient(
        {
            "revision_disposition": "nuke",
            "disposition_rationale": "This is badly broken.",
            "attack_summary": "The task can be cheaply passed.",
            "attacks": [
                {
                    "attack_target": "implementation",
                    "attack_type": "cheap_pass",
                    "exploit_path": "Patch the obvious local branch.",
                    "evidence": "agent_visible_artifact.environment_artifact.payload.files[src/app.py].content",
                    "severity": "critical",
                    "why_it_invalidates_proxy": "It does not require the target ability.",
                }
            ],
            "cheap_pass_strategy": "Patch the local branch.",
            "proxy_damage": "Critical.",
            "survival_requirements": [],
        }
    )

    report, _ = Adversary(client, domain, disposition_mode="binary").attack(candidate, design)

    assert report.revision_disposition == "revise"
    assert client.schema is not None
    assert client.schema["properties"]["revision_disposition"]["enum"] == ["pass", "revise"]
    assert client.user_payload is not None
    assert client.user_payload["attack_surface"]["output_contract"]["revision_disposition"] == "pass or revise"
    assert client.user_payload["decision_policy"]["revision_disposition"] == "pass or revise"
    assert "nuke" not in json.dumps(client.user_payload)
    assert client.system is not None
    assert "be nuked" not in client.system
    assert "nuke" not in client.system


def test_adversary_schema_caps_report_size() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    adversary = Adversary(_CaptureJsonClient({}), domain)

    attacks_schema = adversary._schema["properties"]["attacks"]
    assert attacks_schema["maxItems"] == 3
    assert attacks_schema["items"]["properties"]["exploit_path"]["maxLength"] == 450
    assert adversary._schema["properties"]["survival_requirements"]["maxItems"] == 3


def test_quality_gate_does_not_count_private_judge_artifact_as_answer_leak() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    candidate = _code_candidate(design)
    client = _CaptureJsonClient(
        {
            "verdict": "reject",
            "route_code": "reject_leakage",
            "subcodes": ["answer_leak_root_cause_disclosure"],
            "evidence": [
                {
                    "source": "candidate",
                    "path": "evaluator_private_context.judge_artifact.proxy_claim",
                    "value": "The hidden proxy claim names the causal bug.",
                }
            ],
            "rationale": "Private judge context describes the answer.",
        }
    )

    verdict, _ = QualityGate(client, domain).validate(candidate)

    assert verdict.route_code == RouteCode.REJECT_SEMANTIC_MISMATCH
    assert "answer_leak_root_cause_disclosure" not in verdict.subcodes


def test_quality_gate_schema_excludes_inactive_answer_leak_code() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    candidate = _code_candidate(design)
    client = _CaptureJsonClient(
        {
            "verdict": "accept",
            "route_code": "accept",
            "subcodes": ["proxy_strong"],
            "evidence": [],
            "rationale": "The candidate is acceptable.",
        }
    )

    QualityGate(client, domain).validate(candidate)

    assert client.schema is not None
    subcode_enum = client.schema["properties"]["subcodes"]["items"]["enum"]
    assert "answer_leak_in_candidate_materials" not in subcode_enum
    assert "answer_leak_fix_instruction" in subcode_enum


def test_quality_gate_sanitizes_specific_answer_leak_without_visible_evidence() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    candidate = _code_candidate(design)
    client = _CaptureJsonClient(
        {
            "verdict": "reject",
            "route_code": "reject_leakage",
            "subcodes": ["answer_leak_root_cause_disclosure"],
            "evidence": [
                {
                    "source": "candidate",
                    "path": "evaluator_private_context.judge_artifact.private_root_cause",
                    "value": "The hidden diagnosis names the causal bug.",
                }
            ],
            "rationale": "Private judge context describes the root cause.",
        }
    )

    verdict, _ = QualityGate(client, domain).validate(candidate)

    assert verdict.route_code == RouteCode.REJECT_SEMANTIC_MISMATCH
    assert "answer_leak_root_cause_disclosure" not in verdict.subcodes


def test_quality_gate_keeps_specific_answer_leak_for_agent_visible_evidence() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    candidate = _code_candidate(design)
    client = _CaptureJsonClient(
        {
            "verdict": "reject",
            "route_code": "reject_leakage",
            "subcodes": ["answer_leak_fix_instruction"],
            "evidence": [
                {
                    "source": "candidate",
                    "path": "agent_visible_artifact.benchmark_case.prompt",
                    "value": "The prompt tells the solver what fix to make.",
                }
            ],
            "rationale": "Visible candidate material tells the agent the fix.",
        }
    )

    verdict, _ = QualityGate(client, domain).validate(candidate)

    assert verdict.route_code == RouteCode.REJECT_LEAKAGE
    assert verdict.subcodes == ["answer_leak_fix_instruction"]


def test_adversary_does_not_count_private_context_as_answer_leakage() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    candidate = _code_candidate(design)
    client = _CaptureJsonClient(
        {
            "revision_disposition": "revise",
            "disposition_rationale": "Private scoring context exposes the intended fix.",
            "attack_summary": "The hidden rubric is too explicit.",
            "attacks": [
                {
                    "attack_target": "leakage",
                    "attack_type": "answer_leakage",
                    "exploit_path": "Read evaluator_private_context.judge_artifact.proxy_claim.",
                    "evidence": "evaluator_private_context.judge_artifact.proxy_claim",
                    "severity": "high",
                    "why_it_invalidates_proxy": "It names the intended diagnosis.",
                }
            ],
            "cheap_pass_strategy": "Read hidden metadata.",
            "proxy_damage": "High.",
            "survival_requirements": [],
        }
    )

    report, _ = Adversary(client, domain).attack(candidate, design)

    assert report.attacks[0]["attack_type"] == "proxy_overclaim"


def test_adversary_keeps_answer_leakage_for_agent_visible_evidence() -> None:
    design = _code_design()
    domain = load_domain("domains/benchmark_code_debug.yaml")
    candidate = _code_candidate(design)
    client = _CaptureJsonClient(
        {
            "revision_disposition": "revise",
            "disposition_rationale": "Visible files expose the intended fix.",
            "attack_summary": "README leaks the answer.",
            "attacks": [
                {
                    "attack_target": "leakage",
                    "attack_type": "answer_leakage",
                    "exploit_path": "Follow the README instead of diagnosing the bug.",
                    "evidence": "agent_visible_artifact.environment_artifact.payload.files[README.md].content",
                    "severity": "high",
                    "why_it_invalidates_proxy": "It tells the solver what to patch.",
                }
            ],
            "cheap_pass_strategy": "Follow README.",
            "proxy_damage": "High.",
            "survival_requirements": [],
        }
    )

    report, _ = Adversary(client, domain).attack(candidate, design)

    assert report.attacks[0]["attack_type"] == "answer_leakage"


def test_revision_prompt_omits_transport_bookkeeping() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    candidate.id = "candidate-1-rev"
    candidate.provenance["revision_of"] = "candidate-0"
    domain = load_domain("domains/benchmark_code_debug.yaml")
    client = _PatchClient(
        {
            "benchmark_case_updates": {
                "prompt": "Debug the reconciliation worker with multi-period refund coverage.",
            },
            "environment_ops": [
                {
                    "op": "edit_file",
                    "path": "tests/test_reconcile.py",
                    "old_text": "def test_refund_reduces_total():\n    assert summarize([{'period': '2026-03', 'amount': 100}, {'period': '2026-03', 'amount': -25}]) == {'2026-03': 75}\n",
                    "new_text": "def test_refund_reduces_total():\n    assert summarize([{'period': '2026-03', 'amount': 100}, {'period': '2026-03', 'amount': -25}]) == {'2026-03': 75}\n\n\ndef test_multiple_periods_stay_separate():\n    rows = [{'period': '2026-03', 'amount': 100}, {'period': '2026-04', 'amount': -25}]\n    assert summarize(rows) == {'2026-03': 100, '2026-04': -25}\n",
                },
            ],
            "revision_rationale": "Add regression pressure without solving the starter bug.",
        }
    )

    SampleGenerator(client, domain).revise_from_attack(
        run_id="run",
        design=design,
        candidate=candidate,
        report=_attack_report(candidate.id),
        attempt=2,
    )

    assert client.user_payload is not None
    assert "id" not in client.user_payload["design_brief"]
    assert "content_hash" not in client.user_payload["design_brief"]
    assert "parent_design_batch_id" not in client.user_payload["design_brief"]
    assert "id" not in client.user_payload["prior_candidate"]
    assert "design_id" not in client.user_payload["prior_candidate"]
    assert "content_hash" not in client.user_payload["prior_candidate"]
    assert "output" not in client.user_payload["prior_candidate"]
    assert "provenance" not in client.user_payload["prior_candidate"]
    assert "revision_of" not in json.dumps(client.user_payload)
    assert "candidate_id" not in client.user_payload["adversary_attack_report"]


def test_generator_example_output_has_private_judge_outlets() -> None:
    domain = load_domain("domains/benchmark_code_debug.yaml")
    example = _example_output_for_domain(domain)
    judge = example["judge_artifact"]

    assert "private_root_cause" in judge
    assert "expected_fix_properties" in judge
    assert "hidden_failure_modes" in judge
    assert "shallow_solution_traps" in judge
    assert "candidate_visibility_boundaries" in judge


def test_generation_contract_preflight_accepts_current_domains() -> None:
    for path in Path("domains").glob("*.yaml"):
        domain = load_domain(path)
        assert validate_generation_contract(domain) == []


def test_generator_example_matches_legacy_benchmark_case_schema() -> None:
    domain = load_domain("domains/flaky_concurrency_bug_triage_python.yaml")
    example = _example_output_for_domain(domain)

    assert set(example["agent_artifact"]["benchmark_case"]) >= {
        "case_id",
        "repo_files",
        "failing_test",
        "pytest_trace",
        "task_instructions",
    }


def test_generator_payload_defaults_private_judge_outlets() -> None:
    payload = {
        "judge_artifact": {
            "score_x": {"score_type": "rubric", "dimensions": [{"name": "quality", "weight": 1.0}]},
            "proxy_claim": "This proxy claim is long enough to describe why the task pressures the claimed ability in context.",
            "diagnostic_pressure": ["state tracing", "regression awareness"],
            "scoring_contract": {"credit": ["causal fix"], "penalties": ["shallow patch"]},
            "leakage_risks": ["visible symptom may overconstrain the diagnosis"],
            "known_limits": ["single small workspace"],
            "coverage_tags": ["debug"],
            "negative_controls": [{"output": "patch the visible assertion", "should_fail_because": "does not fix the cause"}],
        }
    }

    judge = _judge_artifact_from_payload(payload)

    assert judge["private_root_cause"] == ""
    assert judge["expected_fix_properties"] == []
    assert judge["hidden_failure_modes"] == []
    assert judge["shallow_solution_traps"] == []
    assert judge["candidate_visibility_boundaries"] == []


def test_generator_payload_drops_optional_null_benchmark_case_fields() -> None:
    design = _code_design()
    payload, _ = _GenerateClient().complete_json(system="DESIGN IMPLEMENTATION CONTRACT", user="{}", schema={})
    payload["agent_artifact"]["benchmark_case"]["setup"] = None

    candidate = _candidate_from_generation_payload(
        run_id="run",
        envelope=GenerationEnvelope.from_design(design),
        design=design,
        attempt=1,
        role_name="test",
        payload=payload,
    )

    assert "setup" not in candidate.agent_artifact.benchmark_case


def test_revision_applies_patch_to_prior_candidate_and_execution_workspace() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    domain = load_domain("domains/benchmark_code_debug.yaml")
    generator = SampleGenerator(
        _PatchClient(
            {
                "benchmark_case_updates": {
                    "prompt": "Debug the reconciliation worker across parser, normalizer, and summary behavior."
                },
                "metadata_updates": {
                    "proxy_claim": "The revised case proxies debugging ability by requiring the solver to preserve refund semantics across multiple periods, not merely satisfy one visible assertion."
                },
                "environment_ops": [
                    {
                        "op": "edit_file",
                        "path": "billing/reconcile.py",
                        "old_text": "abs(parsed['amount'])",
                        "new_text": "parsed['amount']",
                    },
                    {
                        "op": "edit_file",
                        "path": "tests/test_reconcile.py",
                        "old_text": "def test_refund_reduces_total():\n    assert summarize([{'period': '2026-03', 'amount': 100}, {'period': '2026-03', 'amount': -25}]) == {'2026-03': 75}\n",
                        "new_text": "def test_refund_reduces_total():\n    assert summarize([{'period': '2026-03', 'amount': 100}, {'period': '2026-03', 'amount': -25}]) == {'2026-03': 75}\n\n\ndef test_multiple_periods_stay_separate():\n    rows = [{'period': '2026-03', 'amount': 100}, {'period': '2026-04', 'amount': -25}]\n    assert summarize(rows) == {'2026-03': 100, '2026-04': -25}\n",
                    },
                ],
                "revision_rationale": "Replace the local visible-assertion task with multi-period state pressure.",
            }
        ),
        domain,
    )

    revised, meta = generator.revise_from_attack(
        run_id="run",
        design=design,
        candidate=candidate,
        report=_attack_report(candidate.id),
        attempt=2,
    )

    assert revised.id == "run-candidate-design-1-2-rev"
    assert revised.provenance["revision_of"] == candidate.id
    assert revised.agent_artifact.benchmark_case["prompt"].startswith("Debug the reconciliation worker across")
    assert "multiple periods" in revised.judge_artifact.proxy_claim
    assert revised.agent_artifact.environment_artifact is not None
    paths = [item["path"] for item in revised.agent_artifact.environment_artifact.payload["files"]]
    assert "README.md" in paths
    assert "billing/reconcile.py" in paths
    workspace = ExecutionWorkspace.from_artifact(revised.agent_artifact.environment_artifact.payload)
    assert "abs(parsed['amount'])" not in workspace.read_file("billing/reconcile.py")
    assert "test_multiple_periods_stay_separate" in workspace.read_file("tests/test_reconcile.py")
    workspace.close()
    assert "revision_rationale" not in revised.output
    assert "environment_ops" not in revised.output
    assert revised.output["agent_artifact"]["runtime_requirements"]["commands"]["test"] == "python -m pytest -q"
    assert revised.output["agent_artifact"]["environment_artifact"]["payload"]["commands"]["test"] == "python -m pytest -q"
    assert meta["revision_op_count"] == 2
    assert meta["revision_edit_file_count"] == 2
    assert meta["revision_files_touched"] == 2
    assert meta["revision_full_rewrite_ratio"] == 0.0


def test_revision_rejects_unsupported_environment_operation() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    domain = load_domain("domains/benchmark_code_debug.yaml")
    generator = SampleGenerator(
        _PatchClient(
            {
                "environment_ops": [
                    {"op": "write_file", "path": "billing/reconcile.py", "content": "replacement"},
                ],
                "revision_rationale": "Old tool schema should fail.",
            }
        ),
        domain,
    )

    with pytest.raises(ProviderError, match="op must be 'edit_file'"):
        generator.revise_from_attack(
            run_id="run",
            design=design,
            candidate=candidate,
            report=_attack_report(candidate.id),
            attempt=2,
        )


def test_revision_rejects_environment_edit_without_exact_match() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    domain = load_domain("domains/benchmark_code_debug.yaml")
    generator = SampleGenerator(
        _PatchClient(
            {
                "environment_ops": [
                    {
                        "op": "edit_file",
                        "path": "billing/reconcile.py",
                        "old_text": "not present",
                        "new_text": "replacement",
                    },
                ],
                "revision_rationale": "Exact match failure should route as provider/schema error.",
            }
        ),
        domain,
    )

    with pytest.raises(ProviderError, match="old_text did not match file"):
        generator.revise_from_attack(
            run_id="run",
            design=design,
            candidate=candidate,
            report=_attack_report(candidate.id),
            attempt=2,
        )


def test_revision_rejects_create_if_missing_for_existing_file() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    domain = load_domain("domains/benchmark_code_debug.yaml")
    generator = SampleGenerator(
        _PatchClient(
            {
                "environment_ops": [
                    {
                        "op": "edit_file",
                        "path": "billing/reconcile.py",
                        "new_text": "def summarize(rows):\n    return {}\n",
                        "create_if_missing": True,
                    },
                ],
            }
        ),
        domain,
    )

    with pytest.raises(ProviderError, match="path already exists"):
        generator.revise_from_attack(
            run_id="run",
            design=design,
            candidate=candidate,
            report=_attack_report(candidate.id),
            attempt=2,
        )


def test_revision_rejects_old_full_candidate_output() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    domain = load_domain("domains/benchmark_code_debug.yaml")
    generator = SampleGenerator(
        _PatchClient(
            {
                "benchmark_case": {"prompt": "old full-object style"},
                "score_x": {"score_type": "hard_checks_plus_rubric"},
                "environment_artifact": {"kind": "executioner_workspace", "payload": {}},
            }
        ),
        domain,
    )

    with pytest.raises(ProviderError, match="unsupported top-level keys"):
        generator.revise_from_attack(
            run_id="run",
            design=design,
            candidate=candidate,
            report=_attack_report(candidate.id),
            attempt=2,
        )


def test_revision_tolerates_unchanged_runtime_requirements_field() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    domain = load_domain("domains/benchmark_code_debug.yaml")
    generator = SampleGenerator(
        _PatchClient(
            {
                "runtime_requirements": candidate.agent_artifact.runtime_requirements,
                "metadata_updates": {
                    "proxy_claim": "The revised metadata still preserves the same runtime contract while tightening the judge-facing proxy claim."
                },
            }
        ),
        domain,
    )

    revised, _ = generator.revise_from_attack(
        run_id="run",
        design=design,
        candidate=candidate,
        report=_attack_report(candidate.id),
        attempt=2,
    )

    assert revised.agent_artifact.runtime_requirements == candidate.agent_artifact.runtime_requirements
    assert revised.output["agent_artifact"]["runtime_requirements"] == candidate.agent_artifact.runtime_requirements


def test_revision_rejects_runtime_contract_changes() -> None:
    design = _code_design()
    candidate = _code_candidate(design)
    domain = load_domain("domains/benchmark_code_debug.yaml")
    changed_runtime = dict(candidate.agent_artifact.runtime_requirements)
    changed_runtime["commands"] = {"test": "pytest -q"}
    generator = SampleGenerator(
        _PatchClient(
            {
                "runtime_requirements": changed_runtime,
                "metadata_updates": {"proxy_claim": "Try to change runtime."},
            }
        ),
        domain,
    )

    with pytest.raises(ProviderError, match="cannot change runtime_requirements"):
        generator.revise_from_attack(
            run_id="run",
            design=design,
            candidate=candidate,
            report=_attack_report(candidate.id),
            attempt=2,
        )
