from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from codex_client import CodexClient, CodexResponse
from config import GEMINI_PROVIDER_ALIASES, XAI_PROVIDER_ALIASES, ModelConfig
from model_helpers import _nonempty_string
from provider_errors import ProviderError, ProviderStructuredOutputError
from structured_output import (
    _codex_structured_response,
    _codex_system_with_json_schema_instruction,
    _message_text,
    _response_model_name,
    _structured_output_schema,
    _structured_response_parts,
    _usage_metadata,
)
from text_hygiene import normalize_text_tree


class ModelClient:
    def __init__(self, config: ModelConfig, stream_event_callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.config = config
        self._codex_client: CodexClient | None = None
        self._stream_event_callback = stream_event_callback
        self._stream_context_stack: list[dict[str, Any] | None] = []

    @contextmanager
    def stream_context(self, context: dict[str, Any]) -> Iterator[None]:
        self._stream_context_stack.append(context)
        try:
            yield
        finally:
            self._stream_context_stack.pop()

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.4,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        response = self._invoke_structured_chat(system=system, user=user, temperature=temperature, schema=schema)
        latency_ms = int((time.perf_counter() - started) * 1000)
        payload, raw_response, parsing_error = _structured_response_parts(response)
        raw_content = _message_text(raw_response) if raw_response is not None else ""
        if parsing_error is not None:
            raise ProviderStructuredOutputError(
                f"provider returned invalid structured output: {parsing_error}",
                raw_content=raw_content,
                parsing_error=parsing_error,
            )
        if payload is None:
            raise ProviderStructuredOutputError("provider returned empty structured output", raw_content=raw_content)
        if not isinstance(payload, dict):
            raise ProviderStructuredOutputError(
                f"provider returned non-object structured output: {type(payload).__name__}",
                raw_content=raw_content,
            )
        usage = _usage_metadata(raw_response if raw_response is not None else response)
        meta = {
            "provider": self.config.provider,
            "model": _response_model_name(raw_response if raw_response is not None else response) or self.config.model,
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
            "reasoning_effort": self.config.reasoning_effort,
            "structured_output": True,
        }
        payload, replacements = normalize_text_tree(payload)
        meta["text_normalization_replacements"] = replacements
        return payload, meta

    def complete_text(self, *, system: str, user: str, temperature: float = 0.7) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        response = self._invoke_chat(system=system, user=user, temperature=temperature)
        latency_ms = int((time.perf_counter() - started) * 1000)
        content = _message_text(response)
        if not content:
            raise ProviderError("provider returned empty text content")
        usage = _usage_metadata(response)
        content, replacements = normalize_text_tree(content)
        return content, {
            "provider": self.config.provider,
            "model": _response_model_name(response) or self.config.model,
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
            "reasoning_effort": self.config.reasoning_effort,
            "text_normalization_replacements": replacements,
        }

    def complete_with_tools(
        self,
        *,
        system: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> CodexResponse:
        if _model_provider(self.config) != "codex":
            return self._complete_with_langchain_tools(system=system, input_items=input_items, tools=tools)
        if self._codex_client is None:
            self._codex_client = CodexClient(self.config)
        return self._codex_client.invoke_with_tools(
            system=system,
            input_items=input_items,
            tools=tools,
            stream_event_callback=self._emit_stream_event,
        )

    def supports_function_tools(self) -> bool:
        return True

    def _complete_with_langchain_tools(
        self,
        *,
        system: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> CodexResponse:
        model = self._init_chat_model(temperature=0.2)
        if not hasattr(model, "bind_tools"):
            raise ProviderError("function tool loop requires a LangChain model with bind_tools")
        bind_tools = _langchain_bind_tool_schemas(tools)
        try:
            tool_model = model.bind_tools(bind_tools, tool_choice="auto", parallel_tool_calls=False)
        except TypeError:
            tool_model = model.bind_tools(bind_tools)
        messages = _langchain_tool_messages(system, input_items)
        try:
            response = tool_model.invoke(messages)
        except Exception as exc:
            raise ProviderError(f"tool model invocation failed: {exc}") from exc
        output_items = _langchain_tool_output_items(response)
        usage = _usage_metadata(response)
        return CodexResponse(
            content=_message_text(response),
            usage_metadata=usage,
            response_metadata={"model_name": _response_model_name(response) or self.config.model},
            output_items=output_items,
        )

    def embed(self, text: str) -> tuple[list[float], dict[str, Any]]:
        started = time.perf_counter()
        if self.config.embedding_provider in {"local", "hash", "deterministic"}:
            vector = _local_embedding(text)
            latency_ms = int((time.perf_counter() - started) * 1000)
            return vector, {
                "provider": self.config.embedding_provider,
                "model": self.config.embedding_model,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": latency_ms,
                "cost_usd": 0.0,
            }
        model = self._init_embedding_model()
        try:
            embedding = model.embed_query(text)
        except Exception as exc:
            raise ProviderError(f"embedding invocation failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        return list(embedding), {
            "provider": self.config.embedding_provider,
            "model": self.config.embedding_model,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
        }

    def _invoke_chat(self, *, system: str, user: str, temperature: float) -> Any:
        if _model_provider(self.config) == "codex":
            if self._codex_client is None:
                self._codex_client = CodexClient(self.config)
            return self._codex_client.invoke(
                system=system,
                user=user,
                stream_event_callback=self._emit_stream_event,
            )
        model = self._init_chat_model(temperature=temperature)
        try:
            return model.invoke([("system", system), ("user", user)])
        except Exception as exc:
            raise ProviderError(f"model invocation failed: {exc}") from exc

    def _invoke_structured_chat(self, *, system: str, user: str, temperature: float, schema: dict[str, Any]) -> Any:
        schema = _structured_output_schema(schema)
        if _model_provider(self.config) == "codex":
            if self._codex_client is None:
                self._codex_client = CodexClient(self.config)
            response = self._codex_client.invoke(
                system=_codex_system_with_json_schema_instruction(system, schema),
                user=user,
                structured_output=True,
                stream_event_callback=self._emit_stream_event,
            )
            return _codex_structured_response(response)
        model = self._init_chat_model(temperature=temperature)
        if not hasattr(model, "with_structured_output"):
            raise ProviderError("structured JSON output is required, but the LangChain model has no with_structured_output")
        try:
            structured_model = model.with_structured_output(schema, method="json_schema", include_raw=True)
        except TypeError:
            structured_model = model.with_structured_output(schema, include_raw=True)
        try:
            return structured_model.invoke([("system", system), ("user", user)])
        except Exception as exc:
            raise ProviderError(f"structured model invocation failed: {exc}") from exc

    def _emit_stream_event(self, event: dict[str, Any]) -> None:
        if self._stream_event_callback is None:
            return
        context = self._stream_context_stack[-1] if self._stream_context_stack else None
        self._stream_event_callback({**(context or {}), **event})

    def emit_stream_event(self, event: dict[str, Any]) -> None:
        self._emit_stream_event(event)

    def _init_chat_model(self, *, temperature: float) -> Any:
        init_chat_model = _load_init_chat_model()
        kwargs: dict[str, Any] = {"timeout": self.config.request_timeout_seconds}
        if _supports_temperature(self.config):
            kwargs["temperature"] = _effective_temperature(self.config, temperature)
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if self.config.api_key is not None:
            kwargs["api_key"] = self.config.api_key.get_secret_value()
        reasoning_effort = _reasoning_effort_param(self.config)
        if reasoning_effort is not None:
            if _model_provider(self.config) in XAI_PROVIDER_ALIASES:
                kwargs["extra_body"] = {**dict(kwargs.get("extra_body") or {}), "reasoning_effort": reasoning_effort}
            else:
                kwargs["reasoning_effort"] = reasoning_effort
        if _supports_max_tokens(self.config):
            if self.config.max_tokens is not None:
                kwargs["max_tokens"] = self.config.max_tokens
        if _model_has_provider_prefix(self.config.model):
            return init_chat_model(self.config.model, **kwargs)
        return init_chat_model(
            self.config.model,
            model_provider=_langchain_chat_provider(self.config),
            **kwargs,
        )

    def _init_embedding_model(self) -> Any:
        init_embeddings = _load_init_embeddings()
        kwargs: dict[str, Any] = {}
        if self.config.base_url and self.config.embedding_provider == self.config.provider:
            kwargs["base_url"] = self.config.base_url
        if _model_has_provider_prefix(self.config.embedding_model):
            return init_embeddings(self.config.embedding_model, **kwargs)
        return init_embeddings(self.config.embedding_model, provider=self.config.embedding_provider, **kwargs)



def _langchain_bind_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            converted.append(tool)
            continue
        if isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}, "additionalProperties": False}),
                },
            }
        )
    return converted


def _tool_call_arguments(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _langchain_tool_messages(system: str, input_items: list[dict[str, Any]]) -> list[Any]:
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    except ImportError as exc:
        raise ProviderError("LangChain tool loop requires langchain-core message classes") from exc

    messages: list[Any] = [SystemMessage(content=system)]
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_tool_calls() -> None:
        if not pending_tool_calls:
            return
        messages.append(AIMessage(content="", tool_calls=list(pending_tool_calls)))
        pending_tool_calls.clear()

    for item in input_items:
        item_type = item.get("type")
        if item_type == "message":
            flush_tool_calls()
            role = item.get("role")
            text = _input_item_text(item)
            if role == "assistant":
                messages.append(AIMessage(content=text))
            else:
                messages.append(HumanMessage(content=text))
            continue
        if item_type == "function_call":
            pending_tool_calls.append(
                {
                    "name": str(item.get("name") or ""),
                    "args": _tool_call_arguments(item),
                    "id": _tool_call_id(item),
                }
            )
            continue
        if item_type == "function_call_output":
            flush_tool_calls()
            messages.append(ToolMessage(content=str(item.get("output") or ""), tool_call_id=_tool_call_id(item)))
            continue
    flush_tool_calls()
    return messages


def _input_item_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return ""


def _tool_call_id(item: dict[str, Any]) -> str:
    return _nonempty_string(item.get("call_id")) or _nonempty_string(item.get("id")) or "call_missing"


def _langchain_tool_output_items(response: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    raw_tool_calls = getattr(response, "tool_calls", None)
    if isinstance(raw_tool_calls, list):
        for index, call in enumerate(raw_tool_calls):
            if not isinstance(call, dict):
                continue
            name = _nonempty_string(call.get("name"))
            if not name:
                continue
            args = call.get("args")
            call_id = _nonempty_string(call.get("id")) or f"call_{index + 1}"
            items.append(
                {
                    "type": "function_call",
                    "id": call_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps(args if isinstance(args, dict) else {}),
                }
            )
    if items:
        return items
    additional = getattr(response, "additional_kwargs", None)
    raw_openai_calls = additional.get("tool_calls") if isinstance(additional, dict) else None
    if not isinstance(raw_openai_calls, list):
        return []
    for index, call in enumerate(raw_openai_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = _nonempty_string(function.get("name"))
        if not name:
            continue
        items.append(
            {
                "type": "function_call",
                "id": _nonempty_string(call.get("id")) or f"call_{index + 1}",
                "call_id": _nonempty_string(call.get("id")) or f"call_{index + 1}",
                "name": name,
                "arguments": function.get("arguments") if isinstance(function.get("arguments"), str) else "{}",
            }
        )
    return items



def _load_init_chat_model() -> Any:
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:
        raise ProviderError(
            "LangChain chat model support is required. Install langchain and the provider integration package."
        ) from exc
    return init_chat_model


def _load_init_embeddings() -> Any:
    try:
        from langchain.embeddings import init_embeddings
    except ImportError as exc:
        raise ProviderError(
            "LangChain embedding support is required. Install langchain and the embedding provider integration package."
        ) from exc
    return init_embeddings


def _supports_reasoning_effort(config: ModelConfig) -> bool:
    provider = _model_provider(config)
    normalized = config.model.lower()
    if provider in XAI_PROVIDER_ALIASES:
        return True
    if provider in GEMINI_PROVIDER_ALIASES:
        return True
    return provider == "openai" and normalized.startswith(
        ("gpt-5", "o1", "o3", "o4", "openai:gpt-5", "openai:o1", "openai:o3", "openai:o4")
    )


def _reasoning_effort_param(config: ModelConfig) -> str | None:
    if not config.reasoning_effort or not _supports_reasoning_effort(config):
        return None
    normalized = config.reasoning_effort.strip().lower()
    if normalized == "none" and (_model_provider(config) not in GEMINI_PROVIDER_ALIASES or _is_gemini_3_or_newer(config)):
        return None
    return config.reasoning_effort


def _supports_temperature(config: ModelConfig) -> bool:
    provider = _model_provider(config)
    normalized = config.model.lower()
    return provider != "openai" or not normalized.startswith(("gpt-5", "openai:gpt-5"))


def _supports_max_tokens(config: ModelConfig) -> bool:
    return _model_provider(config) in {"openai", *GEMINI_PROVIDER_ALIASES, *XAI_PROVIDER_ALIASES}


def _effective_temperature(config: ModelConfig, requested: float) -> float:
    if _is_gemini_3_or_newer(config):
        return 1.0
    return requested


def _is_gemini_3_or_newer(config: ModelConfig) -> bool:
    if _model_provider(config) not in GEMINI_PROVIDER_ALIASES:
        return False
    normalized = config.model.lower()
    return normalized.startswith(("gemini-3", "google_genai:gemini-3", "google-genai:gemini-3"))


def _model_provider(config: ModelConfig) -> str:
    if _model_has_provider_prefix(config.model):
        return config.model.split(":", 1)[0].lower().replace("-", "_")
    return config.provider.lower().replace("-", "_")


def _uses_openai_chat_compat(config: ModelConfig) -> bool:
    return _model_provider(config) in {"openai", *GEMINI_PROVIDER_ALIASES}


def _langchain_chat_provider(config: ModelConfig) -> str:
    if _model_provider(config) in GEMINI_PROVIDER_ALIASES:
        return "openai"
    if _model_provider(config) in XAI_PROVIDER_ALIASES:
        return "xai"
    return config.provider


def _model_has_provider_prefix(model: str) -> bool:
    return ":" in model and not model.startswith(("http://", "https://"))



def _local_embedding(text: str, *, dimensions: int = 128) -> list[float]:
    buckets = [0.0] * dimensions
    tokens = text.lower().split()
    if not tokens:
        return buckets
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        buckets[index] += sign
    norm = sum(value * value for value in buckets) ** 0.5
    if norm == 0:
        return buckets
    return [value / norm for value in buckets]
