from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize pipeline performance from run logs.")
    parser.add_argument("run_id", nargs="?", help="Run id to inspect. Omit with --all to scan every run.")
    parser.add_argument("--all", action="store_true", help="Scan all run directories under --logs-dir.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--top", type=int, default=8, help="Number of largest stages or rejection buckets to show.")
    args = parser.parse_args()

    root = Path(args.logs_dir)
    run_dirs = _select_run_dirs(root, args.run_id, all_runs=args.all)
    if not run_dirs:
        print(f"No stage_records.jsonl files found under {root}")
        return 1

    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        records.extend(_with_run(_read_jsonl(run_dir / "stage_records.jsonl"), run_dir.name))
        events.extend(_with_run(_read_jsonl(run_dir / "stage_events.jsonl"), run_dir.name))

    if not records:
        print("No stage records found.")
        return 1

    _print_summary(records, events, top=args.top)
    return 0


def _select_run_dirs(root: Path, run_id: str | None, *, all_runs: bool) -> list[Path]:
    if run_id:
        run_dir = root / run_id
        return [run_dir] if (run_dir / "stage_records.jsonl").exists() else []
    if not all_runs:
        return []
    return sorted(path for path in root.iterdir() if (path / "stage_records.jsonl").exists())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _with_run(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    return [{**row, "_run_id": run_id} for row in rows]


def _print_summary(records: list[dict[str, Any]], events: list[dict[str, Any]], *, top: int) -> None:
    total_latency_ms = sum(_latency_ms(row) for row in records)
    total_input = sum(_int(row.get("input_tokens")) for row in records)
    total_output = sum(_int(row.get("output_tokens")) for row in records)
    run_ids = sorted({str(row.get("_run_id")) for row in records})
    print(f"runs={len(run_ids)} stage_records={len(records)}")
    print(
        "model_latency="
        f"{_fmt_ms(total_latency_ms)} tokens={total_input:,}/{total_output:,} "
        f"commits={_count_commits(records)} rejects={_count_rejects(records)}"
    )
    print()

    print("By Role")
    print("-------")
    role_rows = _group_records(records, key="role")
    for label, rows in sorted(role_rows.items(), key=lambda item: _sum_latency(item[1]), reverse=True):
        print(_record_group_line(label, rows))
    print()

    request_rows = [event for event in events if event.get("stage_event") == "model_request_body"]
    if request_rows:
        print("Request Body Size")
        print("-----------------")
        for label, rows in sorted(_group_events(request_rows).items(), key=lambda item: _sum(item[1], "body_bytes"), reverse=True):
            sizes = [_int(row.get("body_bytes")) for row in rows]
            print(
                f"{label:36} n={len(rows):>3} total={_fmt_bytes(sum(sizes)):>9} "
                f"median={_fmt_bytes(int(median(sizes))):>8} max={_fmt_bytes(max(sizes)):>8}"
            )
        print()

    print(f"Slowest {top}")
    print("---------")
    for row in sorted(records, key=_latency_ms, reverse=True)[:top]:
        print(
            f"{_fmt_ms(_latency_ms(row)):>9} "
            f"{str(row.get('role')):34} "
            f"{str(row.get('verdict')):6} {str(row.get('route_code')):28} "
            f"tok={_int(row.get('input_tokens')):,}/{_int(row.get('output_tokens')):,} "
            f"run={row.get('_run_id')}"
        )
    print()

    rejected = [row for row in records if row.get("verdict") == "reject"]
    if rejected:
        print(f"Top Rejection Waste")
        print("-------------------")
        buckets: dict[tuple[str, str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
        for row in rejected:
            buckets[
                (
                    str(row.get("role")),
                    str(row.get("route_code")),
                    tuple(str(code) for code in row.get("subcodes", [])),
                )
            ].append(row)
        for (role, route, subcodes), rows in sorted(buckets.items(), key=lambda item: _sum_latency(item[1]), reverse=True)[:top]:
            code_text = ",".join(subcodes) or "-"
            print(
                f"{_fmt_ms(_sum_latency(rows)):>9} n={len(rows):>3} "
                f"{role:34} {route:28} codes={code_text}"
            )
        print()

    print("Routes")
    print("------")
    for (role, route), count in Counter((row.get("role"), row.get("route_code")) for row in records).most_common():
        print(f"{count:>4}  {role} -> {route}")


def _group_records(records: list[dict[str, Any]], *, key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key))].append(record)
    return groups


def _group_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[str(event.get("role") or event.get("stage") or "model")].append(event)
    return groups


def _record_group_line(label: str, rows: list[dict[str, Any]]) -> str:
    latencies = [_latency_ms(row) for row in rows]
    input_tokens = sum(_int(row.get("input_tokens")) for row in rows)
    output_tokens = sum(_int(row.get("output_tokens")) for row in rows)
    return (
        f"{label:36} n={len(rows):>3} total={_fmt_ms(sum(latencies)):>9} "
        f"median={_fmt_ms(int(median(latencies))):>9} "
        f"tok={input_tokens:,}/{output_tokens:,}"
    )


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return sum(_int(row.get(key)) for row in rows)


def _sum_latency(rows: list[dict[str, Any]]) -> int:
    return sum(_latency_ms(row) for row in rows)


def _latency_ms(row: dict[str, Any]) -> int:
    latency = _int(row.get("latency_ms"))
    if latency:
        return latency
    error = row.get("error")
    if not isinstance(error, str):
        return 0
    match = re.search(r"elapsed_ms=(\d+)", error)
    return int(match.group(1)) if match else 0


def _count_commits(records: list[dict[str, Any]]) -> int:
    return sum(1 for row in records if row.get("role") == "curate_committed_sample" and row.get("verdict") == "accept")


def _count_rejects(records: list[dict[str, Any]]) -> int:
    return sum(1 for row in records if row.get("verdict") == "reject")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _fmt_ms(value: int) -> str:
    if value >= 60_000:
        return f"{value / 60_000:.1f}m"
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{value}ms"


def _fmt_bytes(value: int) -> str:
    if value >= 1_048_576:
        return f"{value / 1_048_576:.1f}MiB"
    if value >= 1024:
        return f"{value / 1024:.1f}KiB"
    return f"{value}B"


if __name__ == "__main__":
    raise SystemExit(main())
