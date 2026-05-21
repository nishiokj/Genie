from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from models import StageRecord, stable_hash

_event_callback: Callable[[str, dict[str, Any]], None] | None = None


def set_event_callback(callback: Callable[[str, dict[str, Any]], None] | None) -> None:
    global _event_callback
    _event_callback = callback


def emit_event(event: str, data: dict[str, Any] | None = None) -> bool:
    if _event_callback is None:
        return False
    _event_callback(event, data or {})
    return True


class StageLogWriter:
    def __init__(self, logs_dir: Path, run_id: str) -> None:
        self.run_dir = logs_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage_records_path = self.run_dir / "stage_records.jsonl"
        self.stage_io_path = self.run_dir / "stage_io.jsonl"
        self.stage_events_path = self.run_dir / "stage_events.jsonl"
        self._lock = threading.Lock()

    def append_event(self, event: str, value: dict[str, Any]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        payload = {
            "run_id": value.get("run_id"),
            "event_type": event,
            "ts": ts,
            "wallclock_ts": ts,
            **value,
        }
        self._append_jsonl(self.stage_events_path, _jsonable(payload))

    def write_stage_record(
        self,
        record: StageRecord,
        *,
        stage_input: Any | None = None,
        stage_output: Any | None = None,
    ) -> None:
        value = record.model_dump(mode="json")
        if stage_input is not None or stage_output is not None:
            trace = {
                "run_id": record.run_id,
                "stage_id": record.stage_id,
                "role": record.role,
                "stage_kind": record.stage_kind,
                "artifact_id": record.artifact_id,
                "parent_artifact_id": record.parent_artifact_id,
                "input_hash": record.input_hash,
                "output_hash": record.output_hash,
                "input": stage_input,
                "output": stage_output,
                "wallclock_ts": record.wallclock_ts,
            }
            self._append_jsonl(self.stage_io_path, _jsonable(trace))
        self._append_jsonl(self.stage_records_path, value)
        emit_event("stage_result", value)

    def append_validation(self, value: Any) -> None:
        self._append_jsonl(self.run_dir / "validation.jsonl", _jsonable(value))

    def append_rejection(self, value: Any) -> None:
        self._append_jsonl(self.run_dir / "rejections.jsonl", _jsonable(value))

    def append_adversary_report(self, value: Any) -> None:
        self._append_jsonl(self.run_dir / "adversary.jsonl", _jsonable(value))

    def append_candidate(self, value: Any) -> None:
        self._append_jsonl(self.run_dir / "candidates.jsonl", _jsonable(value))

    def append_generation_envelope(self, value: Any) -> None:
        self._append_jsonl(self.run_dir / "generation_envelopes.jsonl", _jsonable(value))

    def _append_jsonl(self, path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, sort_keys=True) + "\n")

def _jsonable(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    return {"value": value}


def _jsonable_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def trace_hash(value: Any) -> str:
    return stable_hash(_jsonable_value(value))
