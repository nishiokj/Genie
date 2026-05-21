from __future__ import annotations

from typing import Any

from langgraph.graph import END

from models import StageKind, Verdict


def route_from_decision(state: dict[str, Any]) -> str:
    decision = state["last_decision"]
    if decision is None or decision.terminal:
        return END
    return {
        StageKind.DESIGN: "design",
        StageKind.DESIGN_AUDIT: "audit_design",
        StageKind.GENERATION: "generate",
        StageKind.VALIDATION: "validate_det",
        StageKind.CURATION: "curate",
    }[decision.next_stage]


def after_validate_design_batch_det(state: dict[str, Any]) -> str:
    decision = state["last_decision"]
    if decision and decision.verdict == Verdict.ACCEPT:
        return "select_next_design"
    return route_from_decision(state)


def after_curate(state: dict[str, Any]) -> str:
    if state["committed_count"] >= state["target_n"]:
        return END
    if state["designs_queue"]:
        return "select_next_design"
    if state["design_round"] > state["max_design_retries"]:
        return END
    return "design"


def after_terminal_design(state: dict[str, Any]) -> str:
    if state["committed_count"] >= state["target_n"]:
        return END
    if state["designs_queue"]:
        return "select_next_design"
    if state["design_round"] > state["max_design_retries"]:
        return END
    return "design"


def after_select_next_design(state: dict[str, Any]) -> str:
    return "audit_design" if state.get("design") else "design"


def after_audit_design(state: dict[str, Any]) -> str:
    decision = state["last_decision"]
    if decision and decision.verdict == Verdict.ACCEPT:
        return "generate"
    if state["designs_queue"]:
        return "select_next_design"
    if state["design_round"] > state["max_design_retries"]:
        return END
    return "design"


def after_validate_det(state: dict[str, Any]) -> str | list[str]:
    if state["det_accepted"]:
        if not state.get("adversary_done"):
            return "adversary"
        return ["quality_gate", "rubric_gate"]
    decision = state["last_decision"]
    if decision and decision.terminal:
        return after_terminal_design(state)
    return route_from_decision(state)


def after_generate(state: dict[str, Any]) -> str:
    decision = state["last_decision"]
    if decision and decision.terminal:
        return after_terminal_design(state)
    return route_from_decision(state)


def after_adversary(state: dict[str, Any]) -> str | list[str]:
    decision = state.get("last_decision")
    if decision:
        if decision.terminal:
            return after_terminal_design(state)
        return route_from_decision(state)
    if state.get("adversary_done"):
        return ["quality_gate", "rubric_gate"]
    return "revise_from_adversary"


def after_gate_join(state: dict[str, Any]) -> str:
    decision = state["last_decision"]
    if decision and decision.terminal:
        return after_terminal_design(state)
    return route_from_decision(state)


def after_generate_entrypoint(state: dict[str, Any]) -> str:
    return route_from_decision(state)


def after_validate_det_entrypoint(state: dict[str, Any]) -> str | list[str]:
    if state["det_accepted"]:
        if not state.get("adversary_done"):
            return "adversary"
        return ["quality_gate", "rubric_gate"]
    return route_from_decision(state)


def after_adversary_entrypoint(state: dict[str, Any]) -> str | list[str]:
    decision = state.get("last_decision")
    if decision:
        return route_from_decision(state)
    if state.get("adversary_done"):
        return ["quality_gate", "rubric_gate"]
    return "revise_from_adversary"


def after_gate_join_entrypoint(state: dict[str, Any]) -> str:
    return route_from_decision(state)


def after_curate_entrypoint(state: dict[str, Any]) -> str:
    return END
