from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import IncompleteRead
from pathlib import Path
from typing import Any, Callable, Iterator
import urllib.error
import urllib.parse
import urllib.request

try:
    import fcntl
except ImportError:  # pragma: no cover - Codex runtime is Unix, keep local imports portable.
    fcntl = None

from config import ModelConfig
from model_helpers import _nonempty_string, _positive_number
from provider_errors import ProviderError

@dataclass
class CodexResponse:
    content: str
    usage_metadata: dict[str, int]
    response_metadata: dict[str, Any]
    output_items: list[dict[str, Any]] | None = None


class CodexClient:
    base_url = "https://chatgpt.com/backend-api/codex"
    token_endpoint = "https://auth.openai.com/oauth/token"
    client_id = "app_EMoamEEZ73f0CkXaXp7hrann"
    refresh_buffer_seconds = 300

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.auth_source_file = config.auth_file or Path.home() / ".codex" / "auth.json"
        self.auth_cache_file = _codex_auth_cache_file(self.auth_source_file)
        self.auth_file = _codex_select_auth_file(self.auth_source_file, self.auth_cache_file)
        self.auth_root: dict[str, Any] = {}
        self.auth_uses_nested_tokens = False
        try:
            self.tokens = self._load_tokens()
        except ProviderError:
            if self.auth_file != self.auth_source_file and self.auth_source_file.exists():
                self.auth_file = self.auth_source_file
                self.tokens = self._load_tokens()
            else:
                raise

    def invoke(
        self,
        *,
        system: str,
        user: str,
        structured_output: bool = False,
        stream_event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> CodexResponse:
        token = self._access_token()
        body = {
            "model": self.config.model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user}],
                }
            ],
            "stream": True,
            "store": False,
            "instructions": system,
        }
        if self.config.reasoning_effort:
            body["reasoning"] = {"effort": self.config.reasoning_effort}
        unsupported_fields = ["max_tokens"] if self.config.max_tokens is not None else None
        _emit_codex_stream_event(
            stream_event_callback,
            "request_config",
            model=self.config.model,
            reasoning_effort=self.config.reasoning_effort,
            timeout_seconds=self.config.request_timeout_seconds,
            stream_idle_timeout_seconds=_codex_stream_idle_timeout_seconds(self.config.request_timeout_seconds),
            max_tokens_declared=self.config.max_tokens,
            max_tokens_applied=False if unsupported_fields else None,
            unsupported_fields=unsupported_fields,
            structured_output=structured_output,
        )
        request_body_json = json.dumps(body, sort_keys=True, separators=(",", ":"))
        _emit_codex_stream_event(
            stream_event_callback,
            "request_body",
            url=f"{self.config.base_url or self.base_url}/responses",
            body_json=request_body_json,
            body_bytes=len(request_body_json.encode("utf-8")),
        )
        request = urllib.request.Request(
            f"{self.config.base_url or self.base_url}/responses",
            data=request_body_json.encode("utf-8"),
            headers=self._headers(token),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds) as response:
                raw = _read_codex_sse(
                    response,
                    timeout_seconds=self.config.request_timeout_seconds,
                    stream_event_callback=stream_event_callback,
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Codex API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Codex connection error: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderError(f"Codex stream timed out: {exc}") from exc
        except OSError as exc:
            raise ProviderError(f"Codex stream I/O error: {exc}") from exc
        return self._parse_sse(raw)

    def invoke_with_tools(
        self,
        *,
        system: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream_event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> CodexResponse:
        token = self._access_token()
        body = {
            "model": self.config.model,
            "input": input_items,
            "stream": True,
            "store": False,
            "instructions": system,
            "tools": tools,
            "tool_choice": "required",
            "parallel_tool_calls": False,
        }
        if self.config.reasoning_effort:
            body["reasoning"] = {"effort": self.config.reasoning_effort}
        _emit_codex_stream_event(
            stream_event_callback,
            "request_config",
            model=self.config.model,
            reasoning_effort=self.config.reasoning_effort,
            timeout_seconds=self.config.request_timeout_seconds,
            stream_idle_timeout_seconds=_codex_stream_idle_timeout_seconds(self.config.request_timeout_seconds),
            max_tokens_declared=self.config.max_tokens,
            max_tokens_applied=False if self.config.max_tokens is not None else None,
            unsupported_fields=["max_tokens"] if self.config.max_tokens is not None else None,
            structured_output=False,
            tool_choice="required",
            parallel_tool_calls=False,
            tools=[tool.get("name") for tool in tools],
        )
        request_body_json = json.dumps(body, sort_keys=True, separators=(",", ":"))
        _emit_codex_stream_event(
            stream_event_callback,
            "request_body",
            url=f"{self.config.base_url or self.base_url}/responses",
            body_json=request_body_json,
            body_bytes=len(request_body_json.encode("utf-8")),
        )
        request = urllib.request.Request(
            f"{self.config.base_url or self.base_url}/responses",
            data=request_body_json.encode("utf-8"),
            headers=self._headers(token),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds) as response:
                raw = _read_codex_sse(
                    response,
                    timeout_seconds=self.config.request_timeout_seconds,
                    stream_event_callback=stream_event_callback,
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Codex API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Codex connection error: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderError(f"Codex stream timed out: {exc}") from exc
        except OSError as exc:
            raise ProviderError(f"Codex stream I/O error: {exc}") from exc
        return self._parse_sse(raw)

    def _headers(self, token: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        account_id = self.tokens.get("chatgpt_account_id") or self.tokens.get("account_id")
        if account_id:
            headers["Chatgpt-Account-Id"] = str(account_id)
        return headers

    def _load_tokens(self) -> dict[str, Any]:
        if not self.auth_file.exists():
            raise ProviderError(f"Codex auth file not found: {self.auth_file}")
        try:
            raw = json.loads(self.auth_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ProviderError(f"Codex auth file could not be read: {self.auth_file}") from exc
        if not isinstance(raw, dict):
            raise ProviderError(f"Codex auth file root must be an object: {self.auth_file}")
        self.auth_root = dict(raw)
        self.auth_uses_nested_tokens = isinstance(raw.get("tokens"), dict)
        tokens = raw.get("tokens") if isinstance(raw.get("tokens"), dict) else raw
        if not isinstance(tokens, dict):
            raise ProviderError(f"Codex auth file has no token object: {self.auth_file}")
        access_token = _nonempty_string(tokens.get("access_token"))
        refresh_token = _nonempty_string(tokens.get("refresh_token"))
        if not access_token or not refresh_token:
            raise ProviderError(f"Codex auth file must contain access_token and refresh_token: {self.auth_file}")
        normalized = dict(tokens)
        account_id = (
            _nonempty_string(raw.get("chatgpt_account_id"))
            or _nonempty_string(tokens.get("chatgpt_account_id"))
            or _nonempty_string(tokens.get("account_id"))
            or _extract_codex_account_id(_nonempty_string(tokens.get("id_token")) or "")
        )
        if account_id:
            normalized["chatgpt_account_id"] = account_id
        normalized["expires_at"] = _positive_number(tokens.get("expires_at")) or _jwt_expiry(access_token) or time.time() + 3600
        return normalized

    def _access_token(self) -> str:
        expires_at = float(self.tokens.get("expires_at") or 0)
        if time.time() < expires_at - self.refresh_buffer_seconds:
            return str(self.tokens["access_token"])
        return self._refresh_access_token_with_lock()

    def _refresh_access_token_with_lock(self) -> str:
        with _codex_auth_file_lock(_codex_auth_lock_file(self.auth_cache_file)):
            try:
                self.tokens = self._load_tokens()
            except ProviderError:
                if self.auth_file != self.auth_source_file and self.auth_source_file.exists():
                    self.auth_file = self.auth_source_file
                    self.tokens = self._load_tokens()
                else:
                    raise
            expires_at = float(self.tokens.get("expires_at") or 0)
            if time.time() < expires_at - self.refresh_buffer_seconds:
                return str(self.tokens["access_token"])
            try:
                return self._refresh_access_token()
            except ProviderError as exc:
                if not _is_likely_stale_refresh_token_error(exc):
                    raise
                try:
                    self.tokens = self._load_tokens()
                except ProviderError:
                    raise exc
                expires_at = float(self.tokens.get("expires_at") or 0)
                if time.time() < expires_at - self.refresh_buffer_seconds:
                    return str(self.tokens["access_token"])
                raise

    def _refresh_access_token(self) -> str:
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": str(self.tokens["refresh_token"]),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.token_endpoint,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Codex token refresh failed {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Codex token refresh connection error: {exc}") from exc
        access_token = _nonempty_string(data.get("access_token"))
        if not access_token:
            raise ProviderError("Codex token refresh response did not include access_token")
        self.tokens["access_token"] = access_token
        self.tokens["refresh_token"] = _nonempty_string(data.get("refresh_token")) or self.tokens["refresh_token"]
        self.tokens["expires_at"] = time.time() + float(data.get("expires_in") or 3600)
        if data.get("id_token"):
            self.tokens["id_token"] = data["id_token"]
            account_id = _extract_codex_account_id(str(data["id_token"]))
            if account_id:
                self.tokens["chatgpt_account_id"] = account_id
        self._store_tokens()
        return access_token

    def _store_tokens(self) -> None:
        if self.auth_uses_nested_tokens:
            self.auth_root["tokens"] = dict(self.tokens)
            self.auth_root["last_refresh"] = datetime.now(timezone.utc).isoformat()
            payload = self.auth_root
        else:
            payload = dict(self.tokens)
        try:
            self._write_auth_payload(self.auth_file, payload)
        except OSError as exc:
            if self.auth_file == self.auth_cache_file:
                raise ProviderError(f"Codex auth cache could not be written: {self.auth_cache_file}") from exc
            try:
                self._write_auth_payload(self.auth_cache_file, payload)
            except OSError as cache_exc:
                raise ProviderError(
                    f"Codex auth refresh succeeded but tokens could not be written to {self.auth_file} "
                    f"or cache {self.auth_cache_file}"
                ) from cache_exc
            self.auth_file = self.auth_cache_file

    def _write_auth_payload(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        os.replace(tmp, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def _parse_sse(self, raw: str) -> CodexResponse:
        chunks: list[str] = []
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        model = self.config.model
        response_id = None
        response_error: str | None = None
        output_items: list[dict[str, Any]] = []
        output_item_ids: set[str] = set()
        text_delta_item_ids: set[str] = set()
        text_part_event_ids: set[str] = set()
        completed_text_item_ids: set[str] = set()
        # Fallback for streams that finish arguments before output_item.done.
        pending_calls: dict[str, dict[str, Any]] = {}

        def _pending_call(item_id: str | None, call_id: str | None) -> dict[str, Any] | None:
            if item_id and item_id in pending_calls:
                return pending_calls[item_id]
            if call_id and call_id in pending_calls:
                return pending_calls[call_id]
            return None

        def _track_pending_call(call: dict[str, Any]) -> None:
            if call.get("item_id"):
                pending_calls[call["item_id"]] = call
            if call.get("call_id"):
                pending_calls[call["call_id"]] = call

        def _drop_pending_call(call: dict[str, Any] | None) -> None:
            if not call:
                return
            pending_calls.pop(call.get("item_id") or "", None)
            pending_calls.pop(call.get("call_id") or "", None)

        def append_text(text: str, item_id: str | None = None) -> None:
            if not text:
                return
            if item_id and item_id in text_delta_item_ids:
                return
            if item_id and item_id in completed_text_item_ids:
                return
            chunks.append(text)
            if item_id:
                completed_text_item_ids.add(item_id)

        def _commit_output_item(item: dict[str, Any]) -> None:
            item_id = _nonempty_string(item.get("id"))
            if item_id and item_id in output_item_ids:
                return
            output_items.append(item)
            if item_id:
                output_item_ids.add(item_id)
            append_text(_codex_output_item_text(item), item_id)

        for payload in _sse_payloads(raw):
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "response.created":
                response = event.get("response") if isinstance(event.get("response"), dict) else {}
                response_id = response.get("id") or event.get("id") or response_id
            elif event_type == "response.output_text.delta":
                item_id = _nonempty_string(event.get("item_id"))
                if item_id:
                    text_delta_item_ids.add(item_id)
                chunks.append(str(event.get("delta") or ""))
            elif event_type == "response.output_text.done":
                append_text(event.get("text") if isinstance(event.get("text"), str) else "", _nonempty_string(event.get("item_id")))
            elif event_type == "response.output_item.added":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") in {"function_call", "tool_call", "function_tool_call"}:
                    item_id = _nonempty_string(item.get("id"))
                    call_id = _nonempty_string(item.get("call_id")) or item_id
                    call = _pending_call(item_id, call_id) or {"arguments": ""}
                    call["item_id"] = item_id
                    call["call_id"] = call_id
                    call["name"] = _nonempty_string(item.get("name")) or call.get("name")
                    call["type"] = item.get("type")
                    if isinstance(item.get("arguments"), str) and item["arguments"]:
                        call["arguments"] = item["arguments"]
                    _track_pending_call(call)
            elif event_type == "response.function_call_arguments.delta":
                item_id = _nonempty_string(event.get("item_id"))
                call_id = _nonempty_string(event.get("call_id"))
                call = _pending_call(item_id, call_id)
                if call is None:
                    call = {"item_id": item_id, "call_id": call_id, "arguments": "", "name": _nonempty_string(event.get("name"))}
                    _track_pending_call(call)
                call["arguments"] = call.get("arguments", "") + str(event.get("delta") or "")
                _track_pending_call(call)
            elif event_type == "response.function_call_arguments.done":
                item_id = _nonempty_string(event.get("item_id"))
                call_id = _nonempty_string(event.get("call_id"))
                call = _pending_call(item_id, call_id)
                final_args = event.get("arguments") if isinstance(event.get("arguments"), str) else (call or {}).get("arguments") or ""
                name = _nonempty_string(event.get("name")) or (call or {}).get("name")
                effective_call_id = call_id or (call or {}).get("call_id") or item_id or ""
                effective_item_id = item_id or (call or {}).get("item_id") or call_id or ""
                effective_type = (call or {}).get("type") or "function_tool_call"
                # Commit now; output_item.done may never arrive.
                synthetic = {
                    "type": effective_type,
                    "id": effective_item_id,
                    "call_id": effective_call_id,
                    "name": name or "",
                    "arguments": final_args,
                }
                _track_pending_call({**(call or {}), "synthetic": synthetic})
                _commit_output_item(synthetic)
            elif event_type in {"response.content_part.added", "response.content_part.done"}:
                part_key = ":".join(
                    [
                        event_type,
                        str(event.get("item_id") or ""),
                        str(event.get("output_index") or ""),
                        str(event.get("content_index") or ""),
                    ]
                )
                if part_key not in text_part_event_ids:
                    text_part_event_ids.add(part_key)
                    append_text(_codex_content_part_text(event.get("part")), _nonempty_string(event.get("item_id")))
            elif event_type == "response.refusal.delta":
                chunks.append(str(event.get("delta") or ""))
            elif event_type == "response.refusal.done":
                append_text(
                    event.get("refusal") if isinstance(event.get("refusal"), str) else "",
                    _nonempty_string(event.get("item_id")),
                )
            elif event_type == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict):
                    _commit_output_item(item)
                    call = _pending_call(_nonempty_string(item.get("id")), _nonempty_string(item.get("call_id")))
                    _drop_pending_call(call)
            elif event_type == "response.completed":
                response = event.get("response") if isinstance(event.get("response"), dict) else {}
                model = str(response.get("model") or model)
                response_id = response.get("id") or response_id
                response_output = _codex_normalize_output_items(response.get("output"))
                if response_output and not output_items:
                    output_items = response_output
                if not chunks:
                    append_text(_codex_response_text(response))
                raw_usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
                usage = {
                    "input_tokens": int(raw_usage.get("input_tokens") or 0),
                    "output_tokens": int(raw_usage.get("output_tokens") or 0),
                }
            elif event_type in {"response.failed", "response.incomplete", "response.cancelled", "error"}:
                response = event.get("response") if isinstance(event.get("response"), dict) else {}
                response_error = _codex_error_message(event, response) or event_type
        content = "".join(chunks)
        if response_error:
            detail = f": {response_error}" if response_error else ""
            partial = f" partial_content_chars={len(content)}" if content else ""
            raise ProviderError(f"Codex stream ended unsuccessfully{detail}{partial}")
        return CodexResponse(
            content=content,
            usage_metadata=usage,
            response_metadata={"model_name": model, "id": response_id},
            output_items=output_items,
        )
def _jwt_payload(token: str) -> dict[str, Any]:
    import base64

    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode((payload + padding).encode("ascii")).decode("utf-8")
        parsed = json.loads(decoded)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _jwt_expiry(token: str) -> float | None:
    return _positive_number(_jwt_payload(token).get("exp"))


def _extract_codex_account_id(id_token: str) -> str | None:
    auth_claim = _jwt_payload(id_token).get("https://api.openai.com/auth")
    if not isinstance(auth_claim, dict):
        return None
    return _nonempty_string(auth_claim.get("chatgpt_account_id"))


def _sse_payloads(raw: str) -> list[str]:
    payloads: list[str] = []
    for frame in raw.replace("\r\n", "\n").split("\n\n"):
        lines = []
        for line in frame.split("\n"):
            if line.startswith("data:"):
                lines.append(line[5:].strip())
        payload = "\n".join(lines).strip()
        if payload and payload != "[DONE]":
            payloads.append(payload)
    return payloads


def _read_codex_sse(
    response: Any,
    *,
    timeout_seconds: float = 180.0,
    idle_timeout_seconds: float | None = None,
    stream_event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    chunks: list[bytes] = []
    started_at = time.monotonic()
    idle_timeout = _codex_stream_idle_timeout_seconds(timeout_seconds, idle_timeout_seconds)
    idle_deadline = started_at + idle_timeout
    processed_payloads = 0
    last_read_wait_emit_ms = -5000
    _emit_codex_stream_event(
        stream_event_callback,
        "stream_open",
        elapsed_ms=0,
        bytes_read=0,
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout,
    )
    while True:
        now = time.monotonic()
        if now >= idle_deadline:
            elapsed_ms = int((now - started_at) * 1000)
            raise ProviderError(
                f"Codex stream idle timeout after {idle_timeout:g}s without bytes "
                f"(elapsed_ms={elapsed_ms}, bytes_read={sum(len(item) for item in chunks)})"
            )
        read_timeout = max(0.1, min(5.0, idle_deadline - now))
        _set_response_read_timeout(response, read_timeout)
        try:
            chunk = _read_sse_chunk(response)
        except (TimeoutError, socket.timeout, OSError) as exc:
            if not _is_timeout_error(exc):
                raise
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            idle_remaining_ms = max(0, int((idle_deadline - time.monotonic()) * 1000))
            if elapsed_ms - last_read_wait_emit_ms >= 5000 or idle_remaining_ms == 0:
                _emit_codex_stream_event(
                    stream_event_callback,
                    "read_wait",
                    elapsed_ms=elapsed_ms,
                    idle_remaining_ms=idle_remaining_ms,
                    bytes_read=sum(len(item) for item in chunks),
                )
                last_read_wait_emit_ms = elapsed_ms
            time.sleep(min(0.25, max(0.0, idle_remaining_ms / 1000)))
            continue
        except IncompleteRead as exc:
            if exc.partial:
                chunks.append(exc.partial)
            raw = b"".join(chunks).decode("utf-8", errors="replace")
            processed_payloads = _emit_new_codex_payload_events(
                raw,
                processed_payloads,
                stream_event_callback,
                started_at=started_at,
            )
            if _codex_stream_terminal(raw):
                return raw
            raise ProviderError("Codex stream ended with an incomplete HTTP read before response.completed") from exc
        if not chunk:
            raw = b"".join(chunks).decode("utf-8", errors="replace")
            if raw:
                return raw
            raise ProviderError("Codex stream returned an empty response")
        chunks.append(chunk)
        idle_deadline = time.monotonic() + idle_timeout
        raw = b"".join(chunks).decode("utf-8", errors="replace")
        processed_payloads = _emit_new_codex_payload_events(
            raw,
            processed_payloads,
            stream_event_callback,
            started_at=started_at,
        )
        if _codex_stream_terminal(raw):
            return raw


def _read_sse_chunk(response: Any) -> bytes:
    readline = getattr(response, "readline", None)
    if callable(readline):
        return readline()
    return response.read(65536)


def _set_response_read_timeout(response: Any, timeout_seconds: float) -> None:
    socket_obj = _find_socket_like(response)
    if socket_obj is None:
        return
    try:
        socket_obj.settimeout(timeout_seconds)
    except Exception:
        return


def _find_socket_like(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any | None:
    if seen is None:
        seen = set()
    if value is None or depth > 6 or id(value) in seen:
        return None
    seen.add(id(value))
    if callable(getattr(value, "settimeout", None)):
        return value
    for attr in ("fp", "raw", "_sock", "sock", "_fp", "_connection"):
        try:
            child = getattr(value, attr)
        except Exception:
            continue
        found = _find_socket_like(child, depth=depth + 1, seen=seen)
        if found is not None:
            return found
    return None


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    return "timed out" in str(exc).lower()


def _emit_new_codex_payload_events(
    raw: str,
    processed_payloads: int,
    stream_event_callback: Callable[[dict[str, Any]], None] | None,
    *,
    started_at: float,
) -> int:
    payloads = _sse_payloads(raw)
    for payload in payloads[processed_payloads:]:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type") or "unknown")
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        delta = event.get("delta")
        text = event.get("text")
        if not isinstance(text, str) and event_type.startswith("response.content_part."):
            part_text = _codex_content_part_text(event.get("part"))
            text = part_text or None
        if not isinstance(text, str) and event_type == "response.output_item.done" and item:
            item_text = _codex_output_item_text(item)
            text = item_text or None
        raw_usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        _emit_codex_stream_event(
            stream_event_callback,
            event_type,
            elapsed_ms=int((time.monotonic() - started_at) * 1000),
            bytes_read=len(raw.encode("utf-8")),
            response_id=response.get("id") or event.get("id"),
            model=response.get("model"),
            status=response.get("status") or event.get("status"),
            item_id=event.get("item_id") or item.get("id"),
            item_type=item.get("type"),
            delta_chars=len(delta) if isinstance(delta, str) else None,
            text_chars=len(text) if isinstance(text, str) else None,
            input_tokens=raw_usage.get("input_tokens"),
            output_tokens=raw_usage.get("output_tokens"),
            error=_codex_error_message(event, response) if event_type in {"response.failed", "response.incomplete", "response.cancelled", "error"} else None,
        )
    return len(payloads)


def _emit_codex_stream_event(
    stream_event_callback: Callable[[dict[str, Any]], None] | None,
    stream_event: str,
    **fields: Any,
) -> None:
    if stream_event_callback is None:
        return
    payload = {
        "provider": "codex",
        "stream_event": stream_event,
        **{key: value for key, value in fields.items() if value is not None},
    }
    stream_event_callback(payload)


def _codex_stream_terminal(raw: str) -> bool:
    return _codex_stream_has_event(raw, {"response.completed", "response.failed", "response.incomplete", "response.cancelled", "error"})


def _codex_stream_has_event(raw: str, event_types: set[str]) -> bool:
    for payload in _sse_payloads(raw):
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            if any(f'"{event_type}"' in payload for event_type in event_types) and '"type"' in payload:
                return True
            continue
        if event.get("type") in event_types:
            return True
    return False


def _codex_response_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    output = _codex_normalize_output_items(response.get("output"))
    if not output:
        return _codex_pick_first_string(response.get("text"), response.get("content"), response.get("message")) or ""
    chunks: list[str] = []
    for item in output:
        chunks.append(_codex_output_item_text(item))
    return "".join(chunks)


def _codex_output_item_text(item: dict[str, Any]) -> str:
    item_type = item.get("type")
    if item_type == "message":
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return _codex_content_part_text(content)
        if isinstance(content, list):
            return "".join(_codex_content_part_text(part) for part in content)
        return ""
    if item_type in {"output_text", "text", "input_text"}:
        return _codex_pick_first_string(item.get("text"), item.get("value"), item.get("content")) or ""
    if item_type in {"output_json", "json"}:
        json_payload = _codex_first_json_payload(item)
        return json.dumps(json_payload) if json_payload is not None else ""
    if item_type == "refusal":
        return _codex_pick_first_string(item.get("refusal"), item.get("text")) or ""
    if item_type in {"function_call", "tool_call", "function_tool_call"}:
        return ""
    return _codex_pick_first_string(item.get("text"), item.get("value"), item.get("content")) or ""


def _codex_content_part_text(part: Any) -> str:
    if not isinstance(part, dict):
        return ""
    part_type = part.get("type")
    if part_type in {"output_json", "json"}:
        json_payload = _codex_first_json_payload(part)
        return json.dumps(json_payload) if json_payload is not None else ""
    if part_type == "refusal":
        return _codex_pick_first_string(part.get("refusal"), part.get("text"), part.get("value")) or ""
    content = part.get("content")
    if isinstance(content, list):
        content_text = "".join(_codex_content_part_text(item) for item in content)
        if content_text:
            return content_text
    return _codex_pick_first_string(part.get("text"), part.get("value"), content) or ""


def _codex_normalize_output_items(output: Any) -> list[dict[str, Any]]:
    if isinstance(output, list):
        return [item for item in output if isinstance(item, dict)]
    if isinstance(output, dict):
        return [output]
    return []


def _codex_first_json_payload(block: dict[str, Any]) -> Any:
    for key in ("json", "output", "parsed", "value"):
        value = block.get(key)
        if isinstance(value, (dict, list)):
            return value
    return None


def _codex_pick_first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str):
            return value
    return None


def _codex_error_message(event: dict[str, Any], response: dict[str, Any]) -> str | None:
    error = event.get("error") or response.get("error")
    if isinstance(error, dict):
        code = _nonempty_string(error.get("code"))
        message = _nonempty_string(error.get("message")) or _nonempty_string(error.get("type"))
        if code and message:
            return f"{code}: {message}"
        return message or code
    if isinstance(error, str):
        return error
    details = response.get("incomplete_details") or event.get("incomplete_details")
    if isinstance(details, dict):
        reason = _nonempty_string(details.get("reason"))
        if reason:
            return reason
    status = _nonempty_string(response.get("status")) or _nonempty_string(event.get("status"))
    return status


def _codex_stream_idle_timeout_seconds(timeout_seconds: float, explicit: float | None = None) -> float:
    if explicit is not None and explicit > 0:
        return explicit
    raw = os.getenv("CODEX_STREAM_IDLE_TIMEOUT_SECONDS")
    if raw is not None and raw.strip():
        parsed = float(raw.strip())
        if parsed <= 0:
            raise ProviderError("CODEX_STREAM_IDLE_TIMEOUT_SECONDS must be positive")
        return parsed
    return timeout_seconds


def _codex_auth_cache_file(source_file: Path) -> Path:
    raw = os.getenv("CODEX_AUTH_CACHE_FILE")
    if raw is not None and raw.strip():
        return Path(raw.strip()).expanduser()
    default_auth_file = Path.home() / ".codex" / "auth.json"
    if source_file.expanduser() != default_auth_file:
        return source_file.with_name(f"{source_file.stem}.cache{source_file.suffix or '.json'}")
    cache_root = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
    digest = hashlib.sha256(str(source_file).encode("utf-8")).hexdigest()[:12]
    return cache_root / "synth-data-pipeline-agents" / f"codex-auth-{digest}.json"


def _codex_auth_lock_file(cache_file: Path) -> Path:
    return cache_file.with_name(f"{cache_file.name}.lock")


@contextmanager
def _codex_auth_file_lock(lock_file: Path) -> Iterator[None]:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _is_likely_stale_refresh_token_error(exc: ProviderError) -> bool:
    message = str(exc).lower()
    return (
        "codex token refresh failed" in message
        and ("invalid_grant" in message or "refresh token" in message or "expired" in message)
    )


def _codex_select_auth_file(source_file: Path, cache_file: Path) -> Path:
    if not cache_file.exists():
        return source_file
    if not source_file.exists():
        return cache_file
    try:
        if cache_file.stat().st_mtime >= source_file.stat().st_mtime:
            return cache_file
    except OSError:
        return source_file
    return source_file
