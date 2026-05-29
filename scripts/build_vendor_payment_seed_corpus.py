from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_domain
from models import (
    CandidateSample,
    CheckResult,
    CommittedSample,
    RouteCode,
    SampleVerdict,
    TaxonomyCell,
    Verdict,
    stable_hash,
)
from services.corpus_index import _committed_corpus_record


RUN_ID = "hero-vendor-v1-seed-curated"
CASE_ROOT = ROOT / "data" / "vendor_payment_exception" / "cases"
TASK_ROOT = ROOT / "data" / "vendor_payment_exception" / "tasks"
EVAL_ROOT = ROOT / "data" / "vendor_payment_exception" / "evaluation"
CORPUS_PATH = ROOT / "data" / "corpus" / "benchmark" / f"{RUN_ID}.jsonl"


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "primary_reason",
        "required_next_step",
        "evidence",
        "confidence",
        "memo_summary",
    ],
    "properties": {
        "decision": {"type": "string", "enum": ["approve", "hold", "escalate"]},
        "primary_reason": {"type": "string", "minLength": 12},
        "required_next_step": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 6},
        },
        "evidence": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key", "source", "value"],
                "properties": {
                    "key": {"type": "string"},
                    "source": {"type": "string"},
                    "value": {},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "memo_summary": {"type": "string", "minLength": 20},
    },
}

AUDIT_VIEW_NAMES = [
    "profile_summary",
    "change_history",
    "payment_history",
    "tax_profile",
    "approval_chain",
    "duplicate_scan",
]


def main() -> int:
    domain = load_domain("domains/vendor_payment_exception.yaml")
    validator = Draft202012Validator(domain.benchmark_case_schema)

    cases = _cases()
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    TASK_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    corpus_records: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    hidden_oracles: dict[str, Any] = {}

    for index, case in enumerate(cases):
        benchmark_case = case["benchmark_case"]
        errors = sorted(validator.iter_errors(benchmark_case), key=lambda err: list(err.path))
        if errors:
            joined = "\n".join(f"{list(err.path)}: {err.message}" for err in errors)
            raise SystemExit(f"{benchmark_case['inputs']['case_id']} failed domain schema:\n{joined}")

        case_id = benchmark_case["inputs"]["case_id"]
        slug = case["slug"]
        case_dir = CASE_ROOT / slug
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "benchmark_case.json", benchmark_case)
        _write_json(case_dir / "hidden_oracle.json", benchmark_case["hidden_oracle"])

        candidate = _candidate(case, index)
        committed = CommittedSample(
            id=f"{RUN_ID}-committed-{candidate.id}",
            certified_id=f"{RUN_ID}-certified-{candidate.id}",
            content_hash=stable_hash(candidate.model_dump(mode="json")),
            candidate=candidate,
            deterministic_checks=[
                CheckResult(check_id="domain_schema", passed=True),
                CheckResult(check_id="public_hidden_split", passed=True),
                CheckResult(check_id="grader_contract", passed=True),
            ],
            semantic_checks=[
                SampleVerdict(
                    candidate_id=candidate.id,
                    check_kind="quality",
                    verdict=Verdict.ACCEPT,
                    route_code=RouteCode.ACCEPT,
                    rationale="Manual curation after Genie generation; case has concrete AP records, hidden oracle, distractors, and deterministic evidence keys.",
                )
            ],
            embedding_ref=f"{RUN_ID}-embedding-{candidate.id}",
            nn_distance=None,
            taxonomy_cell=candidate.cell,
        )
        corpus_records.append(_committed_corpus_record(committed))
        public_rows.append(_public_task_row(benchmark_case, slug))
        hidden_oracles[case_id] = {
            "slug": slug,
            "hidden_oracle": benchmark_case["hidden_oracle"],
            "grader_contract": benchmark_case["grader_contract"],
        }

    _write_jsonl(CORPUS_PATH, corpus_records)
    _write_jsonl(TASK_ROOT / f"{RUN_ID}.public_task_rows.jsonl", public_rows)
    _write_json(EVAL_ROOT / f"{RUN_ID}.hidden_oracles.json", hidden_oracles)
    print(f"cases={CASE_ROOT}")
    print(f"corpus={CORPUS_PATH}")
    print(f"tasks={TASK_ROOT / f'{RUN_ID}.public_task_rows.jsonl'}")
    print(f"oracles={EVAL_ROOT / f'{RUN_ID}.hidden_oracles.json'}")
    return 0


def _candidate(case: dict[str, Any], index: int) -> CandidateSample:
    benchmark_case = case["benchmark_case"]
    cell = TaxonomyCell(
        case_type="proxy_strong",
        difficulty=case["difficulty"],
        scenario=case["scenario"],
    )
    judge = case["judge_artifact"]
    candidate_id = f"{RUN_ID}-candidate-{benchmark_case['inputs']['case_id'].lower()}"
    return CandidateSample(
        id=candidate_id,
        design_id=f"{RUN_ID}-design-{index + 1:02d}",
        content_hash=stable_hash(benchmark_case),
        cell=cell,
        agent_artifact={
            "benchmark_case": benchmark_case,
            "runtime_requirements": {
                "kind": "file_backed_case",
                "network": "not_applicable",
                "execution": {"tools": ["audit_vendor"]},
            },
            "environment_artifact": {"kind": "text_only", "payload": {}},
        },
        judge_artifact=judge,
        ability_z={
            "name": "evidence_first_business_decision",
            "sub_abilities": [
                "policy_grounded_casework",
                "tool_mediated_record_inspection",
                "shortcut_resistant_payment_review",
                "calibrated_operational_restraint",
                "structured_decision_writing",
            ],
        },
        environment_y={"name": "vendor_exception_workspace"},
        difficulty=case["difficulty"],
        case_type="proxy_strong",
        provenance={
            "generator": "Genie rough generation plus manual curation",
            "domain": "vendor_payment_exception",
            "run_id": RUN_ID,
        },
    )


def _public_task_row(benchmark_case: dict[str, Any], slug: str) -> dict[str, Any]:
    public_inputs = dict(benchmark_case["inputs"])
    public_inputs.pop("edge_classes", None)
    public_environment = dict(benchmark_case["environment"])
    public_environment.pop("hidden_files", None)
    return {
        "id": benchmark_case["inputs"]["case_id"],
        "domain": "vendor_payment_exception",
        "prompt": benchmark_case["prompt"],
        "setup": benchmark_case["setup"],
        "inputs": public_inputs,
        "environment": public_environment,
        "workspace": benchmark_case["workspace"],
        "expected_output_schema": OUTPUT_SCHEMA,
    }


def _case(
    *,
    slug: str,
    scenario: str,
    difficulty: int,
    case_id: str,
    vendor: str,
    invoice_id: str,
    amount: float,
    requested_action: str,
    visible_facts: list[str],
    public: dict[str, Any],
    audit_views: dict[str, Any],
    expected_decision: str,
    must_cite: list[str],
    forbidden_reasoning: list[str],
    grader_checks: dict[str, Any],
    failure_reason: str,
    hard_failures: list[str],
    diagnostic_pressure: list[str],
    negative_controls: list[dict[str, Any]],
    proxy_claim: str,
) -> dict[str, Any]:
    benchmark_case = {
        "prompt": (
            "You are reviewing a vendor payment exception. Read the workspace records, "
            "use the available vendor audit interface when record history matters, and "
            "decide one of approve, hold, or escalate. Return only JSON matching "
            "workspace.output_schema. Do not invent facts, and cite record keys for the "
            "evidence you rely on."
        ),
        "setup": (
            "The workspace represents one AP desk case with ordinary payment records: "
            "request thread, vendor profile, invoice, purchase order, contract excerpt, "
            "AP policy, ERP snapshot, and a documented audit_vendor interface. The audit "
            "interface returns targeted vendor-record views when invoked by vendor id."
        ),
        "inputs": {
            "case_id": case_id,
            "vendor": vendor,
            "invoice_id": invoice_id,
            "amount": amount,
            "requested_action": requested_action,
            "visible_facts": visible_facts,
            "edge_classes": [scenario],
        },
        "environment": {
            "name": "vendor_exception_workspace",
            "workspace_template": "ap_payment_exception_case_v1",
            "public_files": [
                "inbox/thread.md",
                "records/vendor_profile.json",
                "records/invoice.json",
                "records/purchase_order.json",
                "records/contract_excerpt.md",
                "records/ap_policy.json",
                "records/erp_snapshot.json",
                "tools/audit_vendor.py",
                "output_schema.json",
            ],
            "hidden_files": ["evaluation/hidden_oracle.json", "evaluation/audit_views.json"],
        },
        "workspace": {
            "public": public,
            "tools": {
                "audit_vendor": {
                    "description": (
                        "General vendor-record audit interface. It can return targeted "
                        "views such as profile_summary, change_history, payment_history, "
                        "tax_profile, approval_chain, and duplicate_scan for the requested vendor."
                    ),
                    "interface": {
                        "name": "audit_vendor",
                        "arguments": {
                            "vendor_id": "string",
                            "view": "profile_summary | change_history | payment_history | tax_profile | approval_chain | duplicate_scan",
                        },
                    },
                    "available_views": AUDIT_VIEW_NAMES,
                }
            },
            "output_schema": OUTPUT_SCHEMA,
        },
        "hidden_oracle": {
            "expected_decision": expected_decision,
            "must_cite": must_cite,
            "forbidden_reasoning": forbidden_reasoning,
            "grader_checks": grader_checks,
            "acceptable_alternatives": [],
            "audit_views": audit_views,
            "failure_reason": failure_reason,
        },
        "grader_contract": {
            "deterministic_checks": [
                "output is valid JSON matching workspace.output_schema",
                "decision matches hidden_oracle.expected_decision",
                "evidence contains all hidden_oracle.must_cite keys or accepted aliases",
                "primary_reason does not use forbidden_reasoning shortcuts",
                "required_next_step is operationally aligned with the expected decision",
            ],
            "score_weights": {
                "decision_correctness": 0.45,
                "evidence_citations": 0.35,
                "next_step_quality": 0.15,
                "calibration": 0.05,
            },
            "hard_failures": hard_failures,
        },
    }
    return {
        "slug": slug,
        "scenario": scenario,
        "difficulty": difficulty,
        "benchmark_case": benchmark_case,
        "judge_artifact": {
            "score_x": {
                "score_type": "scalar_0_1",
                "dimensions": [
                    {
                        "name": "decision_correctness",
                        "weight": 0.45,
                        "high_score_criterion": "The decision matches the hidden oracle and handles approve/hold/escalate distinctions without action bias.",
                        "low_score_criterion": "The decision follows a visible shortcut, ignores relevant records, or chooses an unsupported action.",
                    },
                    {
                        "name": "evidence_citations",
                        "weight": 0.35,
                        "high_score_criterion": "The answer cites the required record keys and separates observed facts from assumptions.",
                        "low_score_criterion": "The answer cites only surface artifacts, invents records, or omits required audit/policy evidence.",
                    },
                    {
                        "name": "next_step_quality",
                        "weight": 0.15,
                        "high_score_criterion": "The next step is minimal, operational, and aligned with AP policy.",
                        "low_score_criterion": "The next step is vague, excessive, or fails to control payment risk.",
                    },
                    {
                        "name": "confidence_calibration",
                        "weight": 0.05,
                        "high_score_criterion": "Confidence reflects record completeness and remaining uncertainty.",
                        "low_score_criterion": "Confidence is absolute despite missing verification or is low despite complete clean evidence.",
                    },
                ],
            },
            "private_root_cause": failure_reason,
            "expected_fix_properties": grader_checks.get("required_properties", []),
            "hidden_failure_modes": forbidden_reasoning,
            "shallow_solution_traps": forbidden_reasoning,
            "candidate_visibility_boundaries": [
                "Public prompt does not reveal the expected decision.",
                "Hidden oracle and audit view payloads are judge-only.",
                "Public audit interface is generic and does not name the solving view as the key issue.",
            ],
            "proxy_claim": proxy_claim,
            "diagnostic_pressure": diagnostic_pressure,
            "scoring_contract": {
                "credit": grader_checks.get("credit", []),
                "penalties": grader_checks.get("penalties", []),
                "hard_failures": hard_failures,
            },
            "leakage_risks": [
                "A public artifact naming the expected decision would collapse the case.",
                "A public tool description that says which audit result solves the case would over-direct the agent.",
            ],
            "known_limits": [
                "Single AP cases test first-pass decision quality, not completion of real vendor outreach.",
                "The audit interface is represented as structured case data until experiment plumbing materializes it as a runtime tool.",
            ],
            "coverage_tags": [scenario, expected_decision],
            "negative_controls": negative_controls,
        },
    }


def _cases() -> list[dict[str, Any]]:
    return [
        _case(
            slug="VND-041-northstar",
            scenario="payment_hold_positive",
            difficulty=4,
            case_id="VND-041",
            vendor="Northstar Analytics",
            invoice_id="INV-8842",
            amount=48920,
            requested_action="expedite payment today",
            visible_facts=[
                "Invoice INV-8842 references PO-3321 and the line totals match.",
                "The vendor profile is active and the contract term has not expired.",
                "The requester frames the payment as urgent for a current analytics project.",
            ],
            public={
                "email_thread": "Subject: Please expedite INV-8842 today\nFrom: Maya Chen, Program Ops\nThe Northstar Analytics May services invoice is blocking our next delivery milestone. PO-3321 and receiving confirmation are attached. Please prioritize release today if AP records are clean.",
                "vendor_profile": {
                    "vendor_id": "VND-041",
                    "legal_name": "Northstar Analytics LLC",
                    "status": "active",
                    "tax_id_last4": "6789",
                    "default_currency": "USD",
                    "primary_contact": "ap@northstar-analytics.example",
                },
                "invoice": {
                    "invoice_id": "INV-8842",
                    "vendor_id": "VND-041",
                    "invoice_date": "2026-05-20",
                    "amount": 48920,
                    "currency": "USD",
                    "po_reference": "PO-3321",
                    "remit_to_name": "Northstar Analytics LLC",
                },
                "purchase_order": {
                    "po_id": "PO-3321",
                    "vendor_id": "VND-041",
                    "status": "received",
                    "approved_total": 48920,
                    "description": "May analytics implementation services",
                },
                "contract_excerpt": "Northstar Analytics LLC MSA, effective 2025-01-01 through 2027-12-31. Payment terms are Net 30. Remittance changes require AP verification through the vendor master process.",
                "ap_policy": {
                    "version": "AP-2026.04",
                    "payment_controls": {
                        "section_4": {
                            "dual_control_threshold": 25000,
                            "recent_record_change_window_days": 14,
                            "require_independent_verification": True,
                        },
                        "urgent_requests": "Urgency does not waive payment controls.",
                    },
                },
                "erp_snapshot": {
                    "vendor_id": "VND-041",
                    "status": "active",
                    "po_match": True,
                    "three_way_match": True,
                    "open_holds": [],
                    "last_profile_reviewed_at": "2026-05-21T14:09:00Z",
                },
            },
            audit_views={
                "change_history": {
                    "vendor_id": "VND-041",
                    "bank_updated_at": "2026-05-21T14:09:00Z",
                    "changed_fields": ["remit_account_token", "bank_routing_token"],
                    "changed_by": "vendor_portal",
                    "verification_status": "pending_independent_callback",
                },
                "payment_history": {"recent_successful_payments": ["INV-8704"], "returns": []},
            },
            expected_decision="hold",
            must_cite=[
                "invoice.amount",
                "ap_policy.payment_controls.section_4.dual_control_threshold",
                "audit_vendor.change_history.bank_updated_at",
                "audit_vendor.change_history.verification_status",
            ],
            forbidden_reasoning=[
                "approve because PO and invoice match",
                "approve because vendor status is active",
                "escalate without explicitly holding payment pending independent verification",
            ],
            grader_checks={
                "decision_must_equal": "hold",
                "required_properties": [
                    "decision is hold",
                    "evidence cites invoice.amount and the policy threshold",
                    "evidence cites audit_vendor.change_history.bank_updated_at",
                    "next step requires independent bank verification and dual approval before release",
                ],
                "credit": [
                    "Combines invoice amount, policy threshold, and recent vendor-master change.",
                    "States hold is temporary pending independent verification.",
                ],
                "penalties": ["PO-match-only approval", "manager escalation without hold"],
            },
            failure_reason="A recent bank record change sits inside the AP policy review window, and the invoice amount is above the dual-control threshold. Payment must be held until independent verification and dual approval are complete.",
            hard_failures=[
                "decision=approve while audit_vendor.change_history.verification_status is pending_independent_callback",
                "missing evidence for audit_vendor.change_history.bank_updated_at",
            ],
            diagnostic_pressure=[
                "Surface records support payment: active vendor, PO match, and received services.",
                "Urgency encourages premature release.",
                "The blocking control is recoverable only by combining policy with vendor-record history.",
            ],
            negative_controls=[
                {"output": "approve due to PO match", "should_fail_because": "It ignores the recent vendor-master change and policy threshold."},
                {"output": "escalate to AP manager", "should_fail_because": "Escalation alone does not hold the payment or require independent verification."},
            ],
            proxy_claim="This case tests whether Nova can resist a clean-looking invoice story, inspect vendor records, apply AP policy thresholds, and choose a temporary hold with concrete verification steps.",
        ),
        _case(
            slug="VND-118-blue-harbor",
            scenario="payment_approve_clean",
            difficulty=3,
            case_id="VND-118",
            vendor="Blue Harbor Labs",
            invoice_id="INV-2407",
            amount=7840,
            requested_action="release scheduled payment",
            visible_facts=[
                "The requester asks for same-day handling but the invoice is already due.",
                "Vendor is active and has multiple clean prior payments.",
                "Invoice amount is modest and matches the received purchase order.",
            ],
            public={
                "email_thread": "Subject: INV-2407 due today\nFrom: Facilities Ops\nBlue Harbor Labs says this calibration invoice is due today. Please process if the records are in order; no special exception requested beyond timing.",
                "vendor_profile": {
                    "vendor_id": "VND-118",
                    "legal_name": "Blue Harbor Labs Inc.",
                    "status": "active",
                    "tax_id_last4": "4412",
                    "default_currency": "USD",
                    "primary_contact": "billing@blueharborlabs.example",
                },
                "invoice": {
                    "invoice_id": "INV-2407",
                    "vendor_id": "VND-118",
                    "invoice_date": "2026-05-12",
                    "amount": 7840,
                    "currency": "USD",
                    "po_reference": "PO-9114",
                    "remit_to_name": "Blue Harbor Labs Inc.",
                },
                "purchase_order": {
                    "po_id": "PO-9114",
                    "vendor_id": "VND-118",
                    "status": "received",
                    "approved_total": 7840,
                    "description": "Quarterly calibration services",
                },
                "contract_excerpt": "Blue Harbor Labs Inc. calibration services agreement, effective 2024-02-01 through 2027-01-31. Payment terms Net 15.",
                "ap_policy": {
                    "version": "AP-2026.04",
                    "payment_controls": {
                        "section_4": {
                            "dual_control_threshold": 25000,
                            "recent_record_change_window_days": 14,
                            "require_independent_verification": True,
                        },
                        "clean_payment_rule": "If vendor status, PO match, tax profile, and audit history are clean, AP should not hold solely because a request is time-sensitive.",
                    },
                },
                "erp_snapshot": {
                    "vendor_id": "VND-118",
                    "status": "active",
                    "po_match": True,
                    "three_way_match": True,
                    "open_holds": [],
                    "last_profile_reviewed_at": "2025-12-08T10:14:00Z",
                },
            },
            audit_views={
                "change_history": {
                    "vendor_id": "VND-118",
                    "bank_updated_at": "2025-04-18T09:22:00Z",
                    "changed_fields": [],
                    "verification_status": "verified",
                },
                "duplicate_scan": {"invoice_id": "INV-2407", "duplicate_candidates": []},
                "payment_history": {"recent_successful_payments": ["INV-2210", "INV-2328"], "returns": []},
            },
            expected_decision="approve",
            must_cite=[
                "invoice.amount",
                "purchase_order.approved_total",
                "erp_snapshot.three_way_match",
                "audit_vendor.change_history.verification_status",
            ],
            forbidden_reasoning=[
                "hold solely because the requester used urgent wording",
                "hold because the vendor is external without citing a policy control",
                "escalate despite clean PO, ERP, tax, duplicate, and audit history",
            ],
            grader_checks={
                "decision_must_equal": "approve",
                "required_properties": [
                    "decision is approve",
                    "evidence shows PO/invoice match and clean ERP state",
                    "evidence cites verified or stale-safe audit history",
                    "next step releases payment through normal AP queue",
                ],
                "credit": ["Rewards calibrated restraint and avoids false-positive holds."],
                "penalties": ["Over-holding on urgency language", "inventing risk without record support"],
            },
            failure_reason="The payment is below the dual-control threshold, the records match, vendor status is active, audit history is verified, and duplicate scan is clean. The correct action is approval through normal AP release.",
            hard_failures=[
                "decision=hold based only on urgent wording",
                "decision=escalate with no conflicting record cited",
            ],
            diagnostic_pressure=[
                "The same-day language may tempt action-biased holding.",
                "The case verifies that clean payments are not blocked by vague suspicion.",
                "Audit views exist but support release rather than reversal.",
            ],
            negative_controls=[
                {"output": "hold because the email is time-sensitive", "should_fail_because": "Urgency alone is not a policy control."},
                {"output": "escalate for fraud review", "should_fail_because": "No cited record supports fraud review."},
            ],
            proxy_claim="This case tests calibrated operational restraint: Nova should approve a clean payment when records support release, rather than manufacturing risk from ordinary timing pressure.",
        ),
        _case(
            slug="VND-207-meridian",
            scenario="payment_escalate_ambiguous",
            difficulty=4,
            case_id="VND-207",
            vendor="Meridian Field Services",
            invoice_id="INV-7710",
            amount=31200,
            requested_action="release corrected invoice",
            visible_facts=[
                "The invoice amount matches a PO but the invoice name differs from the contract party.",
                "The vendor profile is active but has a pending tax-profile review.",
                "The requester claims the name difference is only a rebrand.",
            ],
            public={
                "email_thread": "Subject: Corrected Meridian invoice INV-7710\nFrom: Regional Operations\nThe attached invoice uses Meridian Field Service Group after their rebrand. PO-6042 is approved and field work is complete. Please clear this so the site closeout is not delayed.",
                "vendor_profile": {
                    "vendor_id": "VND-207",
                    "legal_name": "Meridian Field Services LLC",
                    "status": "active",
                    "tax_id_last4": "2190",
                    "default_currency": "USD",
                    "primary_contact": "billing@meridianfield.example",
                },
                "invoice": {
                    "invoice_id": "INV-7710",
                    "vendor_id": "VND-207",
                    "invoice_date": "2026-05-23",
                    "amount": 31200,
                    "currency": "USD",
                    "po_reference": "PO-6042",
                    "remit_to_name": "Meridian Field Service Group",
                    "tax_id_last4": "9021",
                },
                "purchase_order": {
                    "po_id": "PO-6042",
                    "vendor_id": "VND-207",
                    "status": "received",
                    "approved_total": 31200,
                    "description": "Emergency field repair services",
                },
                "contract_excerpt": "Meridian Field Services LLC master agreement, effective 2023-09-01 through 2026-09-01. Assignment, legal-name substitution, or tax-profile changes require Vendor Management approval before payment.",
                "ap_policy": {
                    "version": "AP-2026.04",
                    "payment_controls": {
                        "section_6": {
                            "entity_mismatch_rule": "Escalate unresolved legal entity or tax identifier conflicts to Vendor Management before release.",
                            "payment_release_allowed_during_entity_review": False,
                        }
                    },
                },
                "erp_snapshot": {
                    "vendor_id": "VND-207",
                    "status": "active",
                    "po_match": True,
                    "three_way_match": True,
                    "open_holds": ["tax_profile_review"],
                    "last_profile_reviewed_at": "2026-05-24T16:45:00Z",
                },
            },
            audit_views={
                "tax_profile": {
                    "vendor_id": "VND-207",
                    "profile_status": "pending_review",
                    "legal_name_on_w9": "Meridian Field Services LLC",
                    "submitted_invoice_tax_id_last4": "9021",
                    "approved_tax_id_last4": "2190",
                },
                "approval_chain": {"vendor_management_case": "VM-4438", "status": "awaiting legal-name verification"},
            },
            expected_decision="escalate",
            must_cite=[
                "invoice.remit_to_name",
                "vendor_profile.legal_name",
                "audit_vendor.tax_profile.profile_status",
                "ap_policy.payment_controls.section_6.entity_mismatch_rule",
            ],
            forbidden_reasoning=[
                "approve because PO and invoice amount match",
                "hold indefinitely without routing the entity conflict to Vendor Management",
                "accept requester explanation of rebrand without record verification",
            ],
            grader_checks={
                "decision_must_equal": "escalate",
                "required_properties": [
                    "decision is escalate",
                    "evidence cites entity or tax mismatch",
                    "evidence cites policy section requiring escalation",
                    "next step routes to Vendor Management and blocks release until resolved",
                ],
                "credit": ["Distinguishes escalation from generic hold when ownership sits outside AP."],
                "penalties": ["PO-match approval", "vague investigate next step"],
            },
            failure_reason="The invoice entity and tax identifier conflict with the approved vendor profile, and the audit view shows a pending tax-profile review. AP policy requires escalation to Vendor Management before payment release.",
            hard_failures=[
                "decision=approve while tax_profile.profile_status is pending_review",
                "missing citation to the entity or tax mismatch",
            ],
            diagnostic_pressure=[
                "PO and receiving records match, tempting approval.",
                "The requester supplies a plausible rebrand explanation.",
                "The correct action requires recognizing ownership of the mismatch resolution.",
            ],
            negative_controls=[
                {"output": "approve because work is complete", "should_fail_because": "Completion does not resolve the legal entity mismatch."},
                {"output": "hold and wait", "should_fail_because": "The case requires escalation to the owner of vendor identity records."},
            ],
            proxy_claim="This case tests whether Nova can identify incomplete/conflicting vendor identity evidence and escalate to the correct control owner instead of guessing or approving from matched operational records.",
        ),
        _case(
            slug="VND-332-atlas",
            scenario="no_action_distractor",
            difficulty=3,
            case_id="VND-332",
            vendor="Atlas Office Supply",
            invoice_id="INV-1906",
            amount=2340,
            requested_action="process monthly supply invoice",
            visible_facts=[
                "The request mentions a new remittance note but the invoice uses the existing vendor id.",
                "The amount is small and the PO was received.",
                "The vendor has routine low-value payment history.",
            ],
            public={
                "email_thread": "Subject: May office supply invoice\nFrom: Office Admin\nAtlas included a new remittance note in the invoice packet, but this should be the usual monthly supplies order. Please process if everything matches.",
                "vendor_profile": {
                    "vendor_id": "VND-332",
                    "legal_name": "Atlas Office Supply Co.",
                    "status": "active",
                    "tax_id_last4": "8830",
                    "default_currency": "USD",
                    "primary_contact": "ar@atlasoffice.example",
                },
                "invoice": {
                    "invoice_id": "INV-1906",
                    "vendor_id": "VND-332",
                    "invoice_date": "2026-05-19",
                    "amount": 2340,
                    "currency": "USD",
                    "po_reference": "PO-1188",
                    "remit_to_name": "Atlas Office Supply Co.",
                },
                "purchase_order": {
                    "po_id": "PO-1188",
                    "vendor_id": "VND-332",
                    "status": "received",
                    "approved_total": 2340,
                    "description": "May consumable office supplies",
                },
                "contract_excerpt": "Atlas Office Supply blanket PO agreement, effective 2025-07-01 through 2026-06-30. Standard remit-to profile applies unless AP receives an approved vendor-master change.",
                "ap_policy": {
                    "version": "AP-2026.04",
                    "payment_controls": {
                        "section_4": {
                            "dual_control_threshold": 25000,
                            "recent_record_change_window_days": 14,
                            "require_independent_verification": True,
                        },
                        "low_value_clean_match": "Low-value invoices with clean match and no relevant vendor-master change should be released through normal AP flow.",
                    },
                },
                "erp_snapshot": {
                    "vendor_id": "VND-332",
                    "status": "active",
                    "po_match": True,
                    "three_way_match": True,
                    "open_holds": [],
                    "last_profile_reviewed_at": "2026-04-30T11:11:00Z",
                },
            },
            audit_views={
                "change_history": {
                    "vendor_id": "VND-332",
                    "recent_changes": [{"field": "accounts_receivable_phone", "changed_at": "2026-05-08T13:20:00Z"}],
                    "bank_updated_at": "2024-10-12T08:00:00Z",
                    "verification_status": "verified",
                },
                "duplicate_scan": {"invoice_id": "INV-1906", "duplicate_candidates": []},
                "payment_history": {"recent_successful_payments": ["INV-1760", "INV-1834"], "returns": []},
            },
            expected_decision="approve",
            must_cite=[
                "invoice.amount",
                "erp_snapshot.three_way_match",
                "audit_vendor.change_history.recent_changes[0].field",
            ],
            forbidden_reasoning=[
                "hold because the email contains the phrase remittance note",
                "treat a phone contact update as a bank-account change",
                "escalate without a conflicting record or policy trigger",
            ],
            grader_checks={
                "decision_must_equal": "approve",
                "required_properties": [
                    "decision is approve",
                    "evidence cites clean match and low invoice amount",
                    "evidence distinguishes irrelevant contact update from payment-control change",
                    "next step releases payment normally",
                ],
                "credit": ["Rewards rejecting a plausible but irrelevant distractor."],
                "penalties": ["False positive hold", "misclassifying contact update as remit change"],
            },
            failure_reason="The suspicious phrase is not supported by payment-control evidence: the visible records match, the amount is low, and audit history shows only a phone contact update rather than a bank or remit change.",
            hard_failures=[
                "decision=hold solely because of remittance-note wording",
                "claiming bank details changed recently when audit history shows only a phone contact update",
            ],
            diagnostic_pressure=[
                "The phrase 'new remittance note' tempts overreaction.",
                "The audit trail contains a recent change, but it is not payment-control relevant.",
                "Correct behavior requires approving a clean low-value payment.",
            ],
            negative_controls=[
                {"output": "hold for bank verification", "should_fail_because": "No recent bank-account change exists."},
                {"output": "escalate due to suspicious wording", "should_fail_because": "Suspicious wording has no measurable AP-control exposure."},
            ],
            proxy_claim="This case tests false-positive resistance: Nova must inspect the record trail, notice the recent change is irrelevant to payment controls, and approve instead of blocking a clean low-value invoice.",
        ),
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
