"""Runtime CLI graph for the synthetic data pipeline LangGraph run."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

from models import AgentRole, RouteCode, Verdict


NODE_ORDER = [
    "design",
    "validate_design_batch_det",
    "select_next_design",
    "audit_design",
    "generate",
    "validate_det",
    "adversary",
    "revise_from_adversary",
    "quality_gate",
    "rubric_gate",
    "join_gates",
    "curate",
]

STAGE_NODE_ORDER = [
    "design",
    "validate_design_batch_det",
    "select_next_design",
    "audit_design",
    "generate",
    "validate_det",
    "adversary",
    "revise_from_adversary",
    "quality_gate",
    "rubric_gate",
    "join_gates",
    "curate",
]

NODE_LABELS_FULL = {
    "design": "design",
    "validate_design_batch_det": "design_det",
    "select_next_design": "design_cursor",
    "audit_design": "design_audit",
    "generate": "generate",
    "validate_det": "validate_det",
    "adversary": "adversary",
    "revise_from_adversary": "revise_adv",
    "quality_gate": "quality_gate",
    "rubric_gate": "rubric_gate",
    "join_gates": "join_gates",
    "curate": "curate",
}

NODE_LABELS_COMPACT = {
    "design": "DG",
    "validate_design_batch_det": "DD",
    "select_next_design": "DC",
    "audit_design": "DA",
    "generate": "GN",
    "validate_det": "VD",
    "adversary": "AD",
    "revise_from_adversary": "RV",
    "quality_gate": "QG",
    "rubric_gate": "RG",
    "join_gates": "JG",
    "curate": "CU",
}

STAGE_TO_NODE = {
    "run": None,
    "design": "design",
    "design_det": "validate_design_batch_det",
    "design_cursor": "select_next_design",
    "design_audit": "audit_design",
    "generation": "generate",
    "validation_det": "validate_det",
    "adversary": "adversary",
    "quality_gate": "quality_gate",
    "rubric_gate": "rubric_gate",
    "join_gates": "join_gates",
    "curation": "curate",
}

ROLE_TO_NODE = {
    "design_batch": "design",
    "validate_design_batch_deterministically": "validate_design_batch_det",
    "audit_design": "audit_design",
    "generate_candidate_sample": "generate",
    "revise_candidate_from_adversary": "revise_from_adversary",
    "validate_candidate_deterministically": "validate_det",
    "adversary_attack_report": "adversary",
    "quality_gate_candidate": "quality_gate",
    "rubric_gate_candidate": "rubric_gate",
    "curate_committed_sample": "curate",
}

NODE_AGENT_NAMES = {
    "design": "Designer",
    "audit_design": "DesignAuditor",
    "generate": "SampleGenerator",
    "adversary": "Adversary",
    "revise_from_adversary": "SampleGenerator",
    "quality_gate": "QualityGate",
    "rubric_gate": "RubricGate",
}

AGENT_ROLE_NAMES = {
    AgentRole.DESIGNER.value: "Designer",
    AgentRole.DESIGN_AUDITOR.value: "DesignAuditor",
    AgentRole.SAMPLE_GENERATOR.value: "SampleGenerator",
    AgentRole.QUALITY_GATE.value: "QualityGate",
    AgentRole.RUBRIC_GATE.value: "RubricGate",
}

COLORS = {
    "pending": "\033[90m",
    "running": "\033[94m",
    "accept": "\033[92m",
    "reject": "\033[91m",
    "local": "\033[92m",
}
RESET = "\033[0m"
DIM = "\033[90m"

MIN_FULL_WIDTH = 92

_CX_FULL = {
    "design": 46,
    "validate_design_batch_det": 46,
    "select_next_design": 46,
    "audit_design": 46,
    "generate": 46,
    "validate_det": 46,
    "adversary": 46,
    "revise_from_adversary": 46,
    "quality_gate": 28,
    "rubric_gate": 64,
    "join_gates": 46,
    "curate": 46,
}

_CX_COMPACT = {
    "design": 24,
    "validate_design_batch_det": 24,
    "select_next_design": 24,
    "audit_design": 24,
    "generate": 24,
    "validate_det": 24,
    "adversary": 24,
    "revise_from_adversary": 24,
    "quality_gate": 12,
    "rubric_gate": 36,
    "join_gates": 24,
    "curate": 24,
}

_BY = {
    "design": 0,
    "validate_design_batch_det": 4,
    "select_next_design": 8,
    "audit_design": 12,
    "generate": 16,
    "validate_det": 20,
    "adversary": 24,
    "revise_from_adversary": 28,
    "quality_gate": 34,
    "rubric_gate": 34,
    "join_gates": 39,
    "curate": 43,
}

_CANVAS_H = 50


def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except (ValueError, OSError):
        return 120


def _is_compact() -> bool:
    return _term_width() < MIN_FULL_WIDTH


class _Canvas:
    __slots__ = ("w", "h", "chars", "colors")

    def __init__(self, w: int, h: int) -> None:
        self.w = w
        self.h = h
        self.chars = [[" "] * w for _ in range(h)]
        self.colors: list[list[str | None]] = [[None] * w for _ in range(h)]

    def put(self, x: int, y: int, text: str, color: str | None = None) -> None:
        for i, ch in enumerate(text):
            cx = x + i
            if 0 <= cx < self.w and 0 <= y < self.h:
                self.chars[y][cx] = ch
                self.colors[y][cx] = color

    def render(self) -> str:
        lines: list[str] = []
        for y in range(self.h):
            parts: list[str] = []
            current: str | None = None
            for x in range(self.w):
                color = self.colors[y][x]
                if color != current:
                    if current is not None:
                        parts.append(RESET)
                    if color is not None:
                        parts.append(color)
                    current = color
                parts.append(self.chars[y][x])
            if current is not None:
                parts.append(RESET)
            lines.append("".join(parts).rstrip())
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)


def _draw_box(c: _Canvas, cx: int, y: int, label: str, color: str | None, width: int | None = None) -> None:
    width = width or len(label) + 4
    x = cx - width // 2
    c.put(x, y, "┌" + "─" * (width - 2) + "┐", color)
    c.put(x, y + 1, "│" + label.center(width - 2) + "│", color)
    c.put(x, y + 2, "└" + "─" * (width - 2) + "┘", color)


def _vline(c: _Canvas, x: int, y1: int, y2: int) -> None:
    for y in range(y1, y2 + 1):
        c.put(x, y, "│")


def _hline(c: _Canvas, y: int, x1: int, x2: int) -> None:
    for x in range(min(x1, x2), max(x1, x2) + 1):
        c.put(x, y, "─")


def _status_color(status: str) -> str | None:
    return COLORS.get(status)


def _render_connectors(c: _Canvas, cx: dict[str, int], compact: bool, labels: dict[str, str]) -> None:
    center = cx["design"]
    serial_nodes = [
        "design",
        "validate_design_batch_det",
        "select_next_design",
        "audit_design",
        "generate",
        "validate_det",
        "adversary",
        "revise_from_adversary",
    ]
    for upper, lower in zip(serial_nodes, serial_nodes[1:]):
        c.put(center, _BY[upper] + 2, "┬")
        _vline(c, center, _BY[upper] + 3, _BY[lower] - 1)
        c.put(center, _BY[lower], "┴")

    left = 6 if compact else 16
    right = 42 if compact else 78

    def _roff(node: str) -> int:
        return (len(labels[node]) + 4) // 2

    design_l = center - _roff("design")
    c.put(left, 7, "┌")
    _vline(c, left, 2, 6)
    c.put(left, 1, "└")
    _hline(c, 7, left + 1, center)
    _hline(c, 1, left + 1, design_l)
    c.put(design_l - 1, 1, "→")
    c.put(left, 6, "retry design", DIM)

    generate_l = center - _roff("generate")
    c.put(left, 42, "┌")
    _vline(c, left, _BY["generate"] + 2, 41)
    c.put(left, _BY["generate"] + 1, "└")
    _hline(c, 42, left + 1, center)
    _hline(c, _BY["generate"] + 1, left + 1, generate_l)
    c.put(generate_l - 1, _BY["generate"] + 1, "→")
    c.put(left, 41, "retry sample", DIM)

    validate_r = center + _roff("validate_det")
    revise_r = center + _roff("revise_from_adversary")
    loop_right = 68 if not compact else 40
    c.put(loop_right, 30, "┘")
    _vline(c, loop_right, _BY["validate_det"] + 2, 30)
    c.put(loop_right, _BY["validate_det"] + 2, "┐")
    _hline(c, 30, revise_r, loop_right - 1)
    _hline(c, _BY["validate_det"] + 2, validate_r, loop_right - 1)
    c.put(validate_r + 1, _BY["validate_det"] + 2, "←")
    if not compact:
        c.put(loop_right - 7, 29, "recheck", DIM)

    qx = cx["quality_gate"]
    rx = cx["rubric_gate"]
    jy = _BY["join_gates"]
    c.put(center, _BY["revise_from_adversary"] + 2, "┬")
    _vline(c, center, _BY["revise_from_adversary"] + 3, 32)
    c.put(center, 33, "┬")
    _hline(c, 33, qx, rx)
    c.put(qx, 33, "┴")
    c.put(rx, 33, "┴")
    c.put(center - 5, 32, "parallel", DIM)
    _vline(c, qx, _BY["quality_gate"] + 3, jy - 1)
    _vline(c, rx, _BY["rubric_gate"] + 3, jy - 1)
    _hline(c, jy, qx, rx)
    c.put(qx, jy, "┼")
    c.put(rx, jy, "┼")
    c.put(center, jy, "┴")
    _vline(c, center, jy + 3, _BY["curate"] - 1)
    c.put(center, _BY["curate"], "┴")

    curate_r = center + _roff("curate")
    design_r = center + _roff("design")
    cursor_r = center + _roff("select_next_design")

    _hline(c, 46, curate_r, right)
    c.put(right, 46, "┘")
    _vline(c, right, 2, 46)
    c.put(right, _BY["select_next_design"] + 1, "┤")
    _hline(c, _BY["select_next_design"] + 1, cursor_r, right)
    c.put(cursor_r + 1, _BY["select_next_design"] + 1, "←")
    c.put(right - 11, _BY["select_next_design"] + 3, "next design", DIM)
    c.put(right, 2, "┤")
    _hline(c, 2, design_r, right)
    c.put(design_r + 1, 2, "←")
    c.put(right - 10, 1, "new design", DIM)

    c.put(center - 1, 48, "END", DIM)


def render_graph(
    node_status: dict[str, str],
    stats: dict[str, Any],
    recent: list[str] | None = None,
) -> str:
    compact = _is_compact()
    labels = NODE_LABELS_COMPACT if compact else NODE_LABELS_FULL
    cx = _CX_COMPACT if compact else _CX_FULL
    width = 50 if compact else MIN_FULL_WIDTH

    canvas = _Canvas(width, _CANVAS_H)
    _render_connectors(canvas, cx, compact, labels)
    for node in STAGE_NODE_ORDER:
        _draw_box(canvas, cx[node], _BY[node], labels[node], _status_color(node_status[node]))

    running_agents = [
        NODE_AGENT_NAMES[node]
        for node in NODE_ORDER
        if node_status[node] == "running" and node in NODE_AGENT_NAMES
    ]
    agent_list = "  ".join(f"{COLORS['running']}{name}{RESET}" for name in running_agents) or "—"

    lines = [
        canvas.render(),
        "",
        f"Active Agents: {agent_list}",
        (
            "Run: "
            f"{stats.get('run_id', '-')}"
            f"  target={stats.get('target', '-')}"
            f"  committed={stats.get('committed', 0)}"
            f"  dropped={stats.get('dropped', 0)}"
            f"  design={stats.get('design', '-')}"
        ),
        "Legend: blue=active  green=accepted/last-ok  red=rejected  gray=pending",
    ]
    if compact:
        lines.extend([
            "",
            " ".join(f"{NODE_LABELS_COMPACT[n]}={NODE_LABELS_FULL[n]}" for n in STAGE_NODE_ORDER),
        ])
    if recent:
        lines.extend(["", "─" * min(width, 80), *recent[-5:]])
    return "\n".join(lines)


def create_graph_callback(run_id: str) -> Callable[[str, dict[str, Any]], None]:
    node_status = {node: "pending" for node in NODE_ORDER}
    stats: dict[str, Any] = {"run_id": run_id, "committed": 0, "dropped": 0, "design": "-"}
    recent: list[str] = []
    previous_lines = 0
    printed_recent = 0
    is_tty = sys.stdout.isatty()

    def redraw() -> None:
        nonlocal previous_lines, printed_recent
        if not is_tty:
            for line in recent[printed_recent:]:
                sys.stdout.write(line + "\n")
            sys.stdout.flush()
            printed_recent = len(recent)
            return

        output = render_graph(node_status, stats, recent)
        lines = output.split("\n")
        if is_tty:
            try:
                term_h = os.get_terminal_size().lines
            except (ValueError, OSError):
                term_h = 40
            lines = lines[: max(term_h - 1, 1)]
            if previous_lines:
                sys.stdout.write(f"\033[{previous_lines}A\033[J")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        previous_lines = len(lines)

    redraw()

    def callback(event: str, data: dict[str, Any]) -> None:
        if event == "stage_progress":
            _handle_progress(data, node_status, stats, recent)
            redraw()
            return
        if event == "stage_result":
            _handle_result(data, node_status, stats, recent)
            redraw()

    return callback


def _handle_progress(
    data: dict[str, Any],
    node_status: dict[str, str],
    stats: dict[str, Any],
    recent: list[str],
) -> None:
    stage = str(data.get("stage", ""))
    progress_event = str(data.get("event", ""))
    if stage == "run" and progress_event == "start":
        stats["target"] = data.get("target", "-")
        recent.append(f"run start target={stats['target']} model={data.get('model', '-')}")
        return
    if stage == "candidate":
        recent.append(_candidate_line(progress_event, data))
        return

    node = STAGE_TO_NODE.get(stage)
    if stage == "generation" and progress_event == "revise":
        node = "revise_from_adversary"
    if node is None:
        return
    if node == "select_next_design" and data.get("id"):
        stats["design"] = data["id"]
    if progress_event in {"start", "select", "revise"}:
        parallel_nodes = {"quality_gate", "rubric_gate"}
        for existing, status in list(node_status.items()):
            if status == "running" and not (node in parallel_nodes and existing in parallel_nodes):
                node_status[existing] = "local" if existing in {"validate_design_batch_det", "validate_det", "curate"} else "pending"
        node_status[node] = "running"
        recent.append(_progress_line(stage, progress_event, data))


def _handle_result(
    data: dict[str, Any],
    node_status: dict[str, str],
    stats: dict[str, Any],
    recent: list[str],
) -> None:
    node = ROLE_TO_NODE.get(str(data.get("role", "")))
    if node is None:
        return

    verdict = _enum_value(data.get("verdict"))
    route = _enum_value(data.get("route_code"))
    status = "accept" if verdict == Verdict.ACCEPT.value else "reject"
    if str(data.get("provider", "local")) == "local" and status == "accept":
        status = "local"
    node_status[node] = status

    if node == "curate" and route == RouteCode.ACCEPT.value:
        stats["committed"] = int(stats.get("committed", 0)) + 1
    elif status == "reject" and (
        node in {"audit_design", "curate"} or route.startswith("drop_")
    ):
        stats["dropped"] = int(stats.get("dropped", 0)) + 1

    agent = AGENT_ROLE_NAMES.get(str(data.get("agent_role") or ""), "local")
    artifact = _short_id(str(data.get("artifact_id", "-")))
    codes = ",".join(str(code) for code in data.get("subcodes", [])[:4])
    suffix = f" codes={codes}" if codes else ""
    recent.append(f"{NODE_LABELS_FULL[node]} {verdict} route={route} agent={agent} artifact={artifact}{suffix}")


def _progress_line(stage: str, event: str, data: dict[str, Any]) -> str:
    bits = [event, stage]
    for key in ("round", "design", "id", "candidate", "attempt", "remaining", "retry"):
        value = data.get(key)
        if value is not None:
            bits.append(f"{key}={_short_id(str(_enum_value(value)))}")
    return " ".join(bits)


def _candidate_line(event: str, data: dict[str, Any]) -> str:
    route = data.get("route")
    codes = data.get("codes")
    bits = [event, "candidate"]
    for key in ("id", "case_type", "ability"):
        value = data.get(key)
        if value:
            bits.append(f"{key}={_short_id(str(value))}")
    if route:
        bits.append(f"route={_enum_value(route)}")
    if codes:
        bits.append(f"codes={_short_id(','.join(str(code) for code in codes))}")
    prompt = str(data.get("prompt") or "").replace("\n", " ")
    proxy = str(data.get("proxy") or "").replace("\n", " ")
    if prompt:
        bits.append(f"prompt=\"{_clip(prompt, 120)}\"")
    if proxy:
        bits.append(f"proxy=\"{_clip(proxy, 120)}\"")
    return " ".join(bits)


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _short_id(value: str) -> str:
    if len(value) <= 36:
        return value
    return f"{value[:16]}...{value[-16:]}"
