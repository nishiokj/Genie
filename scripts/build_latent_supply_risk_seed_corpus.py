from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "hero-latent-supply-risk-v1-seed-curated"
DOMAIN_PATH = ROOT / "domains" / "latent_supply_risk.yaml"
DATA_ROOT = ROOT / "data" / "latent_supply_risk"
CASE_ROOT = DATA_ROOT / "cases"
TASK_ROOT = DATA_ROOT / "tasks"
EVAL_ROOT = DATA_ROOT / "evaluation"
WORKSPACE_ROOT = DATA_ROOT / "workspaces"
CORPUS_PATH = ROOT / "data" / "corpus" / "benchmark" / f"{RUN_ID}.jsonl"


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scan"],
    "properties": {
        "scan": {
            "type": "object",
            "additionalProperties": False,
            "required": ["run_id", "result", "alerts", "no_alert_rationale"],
            "properties": {
                "run_id": {"type": "string"},
                "result": {"type": "string", "enum": ["alerts_found", "no_alert"]},
                "alerts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "id",
                            "confidence",
                            "source_headlines",
                            "latent_edge_hypothesis",
                            "real_world_prior",
                            "enterprise_path",
                            "business_exposure",
                            "reasoning",
                            "citations",
                            "uncertainty",
                        ],
                        "properties": {
                            "id": {"type": "string"},
                            "confidence": {"type": "string"},
                            "source_headlines": {"type": "array", "items": {"type": "string"}},
                            "latent_edge_hypothesis": {"type": "string"},
                            "real_world_prior": {"type": "string"},
                            "enterprise_path": {"type": "array", "items": {"type": "string"}},
                            "business_exposure": {
                                "type": "object",
                                "additionalProperties": True,
                                "required": [
                                    "revenue_at_risk",
                                    "affected_orders",
                                    "affected_customers",
                                    "constrained_inventory",
                                ],
                                "properties": {
                                    "revenue_at_risk": {"type": "integer"},
                                    "affected_orders": {"type": "array", "items": {"type": "string"}},
                                    "affected_customers": {"type": "array", "items": {"type": "string"}},
                                    "constrained_inventory": {"type": "boolean"},
                                },
                            },
                            "reasoning": {"type": "string"},
                            "citations": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["headlines", "records"],
                                "properties": {
                                    "headlines": {"type": "array", "items": {"type": "string"}},
                                    "records": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                            "uncertainty": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "no_alert_rationale": {"type": ["string", "null"]},
            },
        }
    },
}


PROMPT = (
    "You are operating the company's daily supply-chain risk scan. Review today's "
    "external headline pack and the company operating records. Use real-world "
    "knowledge to notice external events that may create latent risk through inputs, "
    "materials, logistics, regulations, regions, processes, or supplier dependencies. "
    "Raise a risk alert only when you can explain a plausible path from an external "
    "event to company-specific business exposure. If no credible path exists, submit "
    "a no-alert scan. Separate observed facts from hypotheses and do not remediate. "
    "Your final answer must be a JSON object matching output/output_schema.json."
)

SETUP = (
    "Each workspace is a file-backed enterprise risk scan with neutral external "
    "headlines, compact company records, a deterministic exposure helper, and a "
    "hidden answer key outside the agent workspace. The public records are enough "
    "to discover materials, products, suppliers, inventory, and open orders, but "
    "they do not state the final risk conclusion."
)


def main() -> int:
    domain = yaml.safe_load(DOMAIN_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(domain["benchmark_case_schema"])

    _ensure_dirs()
    cases = _cases()
    corpus_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    hidden_oracles: dict[str, Any] = {}

    for case in cases:
        errors = sorted(validator.iter_errors(case["benchmark_case"]), key=lambda err: list(err.path))
        if errors:
            joined = "\n".join(f"{list(err.path)}: {err.message}" for err in errors)
            raise SystemExit(f"{case['run_id']} failed latent_supply_risk schema:\n{joined}")

        run_id = case["run_id"]
        case_dir = CASE_ROOT / run_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "benchmark_case.json", case["benchmark_case"])
        _write_yaml(case_dir / "hidden-answer-key.yaml", case["benchmark_case"]["hidden_evaluation"])
        _write_workspace(case)

        corpus_rows.append(
            {
                "schema_version": "latent_supply_risk_corpus_v1",
                "id": run_id,
                "domain": "latent_supply_risk",
                "generator": "Genie domain contract plus curated seed materialization",
                "source_schema": str(Path("domains") / "latent_supply_risk.yaml"),
                "difficulty": case["difficulty"],
                "scenario": case["scenario"],
                "case_path": str(case_dir.relative_to(ROOT)),
                "workspace_path": str((WORKSPACE_ROOT / run_id).relative_to(ROOT)),
                "zero_alert_run": case["benchmark_case"]["hidden_evaluation"]["zero_alert_run"],
                "true_alert_count": len(case["benchmark_case"]["hidden_evaluation"]["true_alerts"]),
                "quality_notes": case["quality_notes"],
            }
        )
        task_rows.append(_public_task_row(case))
        hidden_oracles[run_id] = case["benchmark_case"]["hidden_evaluation"]

    _write_jsonl(CORPUS_PATH, corpus_rows)
    _write_jsonl(TASK_ROOT / f"{RUN_ID}.public_case_rows.jsonl", task_rows)
    _write_json(EVAL_ROOT / f"{RUN_ID}.hidden_oracles.json", hidden_oracles)
    print(f"cases={CASE_ROOT}")
    print(f"workspaces={WORKSPACE_ROOT}")
    print(f"corpus={CORPUS_PATH}")
    print(f"tasks={TASK_ROOT / f'{RUN_ID}.public_case_rows.jsonl'}")
    print(f"oracles={EVAL_ROOT / f'{RUN_ID}.hidden_oracles.json'}")
    return 0


def _cases() -> list[dict[str, Any]]:
    return [
        _case_antimony_flame_retardant(),
        _case_castor_drought(),
        _case_rosin_flux(),
        _case_zero_alert(),
        _case_silicon_metal(),
        _case_latent_edge_no_current_exposure(),
    ]


def _base_records(*, variant: str) -> dict[str, Any]:
    products = [
        {
            "id": "CMC-410",
            "name": "Coastal Motor Controller",
            "family": "industrial_controls",
            "region_eligibility": ["North America", "EU"],
            "certifications": ["coastal_corrosion_rating"],
            "status": "active",
        },
        {
            "id": "MTG-220",
            "name": "Medical Telemetry Gateway",
            "family": "clinical_networking",
            "region_eligibility": ["North America"],
            "certifications": ["hospital_low_residue_assembly"],
            "status": "active",
        },
        {
            "id": "RSG-700",
            "name": "Rail Signal Gateway",
            "family": "transportation_controls",
            "region_eligibility": ["North America", "EU"],
            "certifications": ["vibration_hardened"],
            "status": "active",
        },
        {
            "id": "LTH-90",
            "name": "Legacy Thermal Hub",
            "family": "legacy_controls",
            "region_eligibility": ["North America"],
            "certifications": [],
            "status": "discontinued",
        },
    ]
    boms = [
        {
            "product_id": "CMC-410",
            "components": [
                {
                    "component_id": "CR-88",
                    "name": "conformal barrier resin",
                    "role": "coastal_corrosion_rating",
                    "inputs": ["castor_derived_polyol", "silane_coupling_agent"],
                },
                {"component_id": "PCB-210", "name": "industrial control board"},
            ],
        },
        {
            "product_id": "MTG-220",
            "components": [
                {
                    "component_id": "ASM-44",
                    "name": "low-residue sensor board assembly",
                    "role": "hospital_low_residue_assembly",
                    "inputs": ["rosin_ester_flux", "tin_silver_solder"],
                },
                {"component_id": "ENC-19", "name": "sealed polymer enclosure"},
            ],
        },
        {
            "product_id": "RSG-700",
            "components": [
                {
                    "component_id": "POT-72",
                    "name": "vibration dampening potting compound",
                    "role": "vibration_hardened",
                    "inputs": ["silicone_elastomer", "alumina_filler"],
                },
                {"component_id": "MOD-51", "name": "rail communications module"},
            ],
        },
    ]
    suppliers = [
        {
            "id": "SUP-ARBCHEM",
            "name": "ArborChem Materials",
            "sites": [{"id": "SITE-ARBCHEM-TX", "country": "US"}],
            "supplies": [{"component_id": "CR-88"}, {"input_id": "castor_derived_polyol"}],
            "lead_time_days": 28,
            "status": "active",
            "aliases": ["ACM", "Arbor polyols"],
        },
        {
            "id": "SUP-LANTERN",
            "name": "Lantern Assembly Works",
            "sites": [{"id": "SITE-LANTERN-OR", "country": "US"}],
            "supplies": [{"component_id": "ASM-44"}, {"input_id": "rosin_ester_flux"}],
            "lead_time_days": 21,
            "status": "active",
            "tier2_notes": ["flux solids sourced through South China resin brokers"],
        },
        {
            "id": "SUP-NORTHSIL",
            "name": "NorthSil Compounds",
            "sites": [{"id": "SITE-NORTHSIL-ON", "country": "CA"}],
            "supplies": [{"component_id": "POT-72"}, {"input_id": "silicone_elastomer"}],
            "lead_time_days": 35,
            "status": "active",
            "tier2_notes": ["silicone base polymer is indexed to upstream siloxane intermediate availability"],
        },
        {
            "id": "SUP-BETA",
            "name": "Beta Components",
            "sites": [{"id": "SITE-BETA-MY", "country": "MY"}],
            "supplies": [{"component_id": "LCD-12"}],
            "lead_time_days": 45,
            "status": "inactive",
            "notes": "legacy display module source for LTH-90 only",
        },
    ]
    if variant == "zero":
        return {
            "products": products,
            "bom": {"boms": boms},
            "suppliers": suppliers,
            "orders": {
                "orders": [
                    {
                        "id": "SO-9102",
                        "customer": "Prairie Water Authority",
                        "product_id": "CMC-410",
                        "quantity": 40,
                        "unit_revenue": 9800,
                        "due_date": "2026-07-11",
                        "priority": "standard",
                        "status": "open",
                    }
                ]
            },
            "inventory": {
                "inventory": [
                    {"product_id": "CMC-410", "location": "West DC", "quantity_available": 180, "status": "available"},
                    {"product_id": "MTG-220", "location": "Midwest DC", "quantity_available": 240, "status": "available"},
                    {"product_id": "RSG-700", "location": "East DC", "quantity_available": 160, "status": "available"},
                ]
            },
        }
    return {
        "products": products,
        "bom": {"boms": boms},
        "suppliers": suppliers,
        "orders": {"orders": []},
        "inventory": {"inventory": []},
    }


def _case_castor_drought() -> dict[str, Any]:
    records = _base_records(variant="positive")
    records["orders"] = {
        "orders": [
            {
                "id": "SO-7731",
                "customer": "Atlantic Energy",
                "product_id": "CMC-410",
                "quantity": 420,
                "unit_revenue": 10000,
                "due_date": "2026-06-20",
                "priority": "strategic",
                "status": "open",
            },
            {
                "id": "SO-7715",
                "customer": "Hill County Utilities",
                "product_id": "LTH-90",
                "quantity": 80,
                "unit_revenue": 2500,
                "due_date": "2026-08-05",
                "priority": "standard",
                "status": "cancelled",
            },
        ]
    }
    records["inventory"] = {
        "inventory": [
            {"product_id": "CMC-410", "location": "West DC", "quantity_available": 80, "status": "available"},
            {"product_id": "MTG-220", "location": "Midwest DC", "quantity_available": 300, "status": "available"},
            {"product_id": "RSG-700", "location": "East DC", "quantity_available": 95, "status": "available"},
        ]
    }
    headlines = [
        "Regional carriers report two-day delays at Port Klang after weekend storms.",
        "Analysts note continued softness in European industrial demand.",
        "Unseasonal drought conditions deepen across parts of Gujarat, pressuring several agricultural export markets.",
        "Beta Components announces a lead-time increase for legacy display modules.",
        "New customs documentation checks begin for selected battery imports into the EU.",
        "Copper futures trade sideways after a quiet week for construction demand.",
        "Gulf Coast warehouses report normal outbound capacity after scheduled maintenance.",
        "A North American trucker union extends negotiations without a strike notice.",
    ]
    return _make_case(
        run_id="LSR-001",
        date="2026-06-12",
        scenario="castor_polyol_drought_positive",
        difficulty=4,
        company="Aster Vale Controls",
        headlines=headlines,
        records=records,
        true_alerts=[
            {
                "id": "risk-castor-coating",
                "source_headlines": ["H03"],
                "latent_edge": {
                    "world_event": "gujarat_drought",
                    "inferred_external_input": "castor_oil_derivatives",
                    "required_world_knowledge": (
                        "Gujarat and western India are major castor-producing regions, and castor oil derivatives "
                        "can be used in industrial coatings, resins, and polyols."
                    ),
                    "required_concepts": ["gujarat", "castor", "polyol", "coating"],
                },
                "hidden_dependency_path": ["castor_derived_polyol", "CR-88", "CMC-410", "SO-7731"],
                "business_exposure": {
                    "revenue_at_risk": 4200000,
                    "affected_orders": ["SO-7731"],
                    "affected_customers": ["Atlantic Energy"],
                    "constrained_inventory": True,
                },
            }
        ],
        expected_no_alerts=[
            {"source_headline": "H01", "reason": "Port Klang is not used by active suppliers or open orders."},
            {"source_headline": "H04", "reason": "Beta Components only supplies a discontinued legacy display path with no active open exposure."},
            {"source_headline": "H05", "reason": "The company has no open battery-product demand in this workspace."},
        ],
        quality_notes="Positive case: generic drought headline only matters after a castor-to-polyol-to-coating dependency is discovered.",
    )


def _case_rosin_flux() -> dict[str, Any]:
    records = _base_records(variant="positive")
    records["orders"] = {
        "orders": [
            {
                "id": "SO-7844",
                "customer": "Mercy North Hospitals",
                "product_id": "MTG-220",
                "quantity": 260,
                "unit_revenue": 14200,
                "due_date": "2026-06-28",
                "priority": "strategic",
                "status": "open",
            },
            {
                "id": "SO-7802",
                "customer": "Canal Rail",
                "product_id": "RSG-700",
                "quantity": 25,
                "unit_revenue": 18000,
                "due_date": "2026-09-18",
                "priority": "standard",
                "status": "forecast",
            },
        ]
    }
    records["inventory"] = {
        "inventory": [
            {"product_id": "MTG-220", "location": "Midwest DC", "quantity_available": 34, "status": "available"},
            {"product_id": "CMC-410", "location": "West DC", "quantity_available": 160, "status": "available"},
            {"product_id": "RSG-700", "location": "East DC", "quantity_available": 110, "status": "available"},
        ]
    }
    headlines = [
        "South China forestry bureaus extend resin tapping restrictions after a dry spring.",
        "Air cargo rates from Seoul decline as consumer electronics volumes soften.",
        "Hospital purchasing groups report steady demand for bedside monitoring equipment.",
        "A container carrier adds a seasonal surcharge on trans-Pacific refrigerated cargo.",
        "Tin prices fall after smelter maintenance concludes ahead of schedule.",
        "A regional port authority schedules overnight gate software maintenance.",
        "Analysts flag weaker demand for legacy LCD modules in industrial channels.",
        "A packaging resin distributor announces a name change after a merger.",
    ]
    return _make_case(
        run_id="LSR-014",
        date="2026-06-13",
        scenario="rosin_flux_export_positive",
        difficulty=5,
        company="Aster Vale Controls",
        headlines=headlines,
        records=records,
        true_alerts=[
            {
                "id": "risk-rosin-flux",
                "source_headlines": ["H01"],
                "latent_edge": {
                    "world_event": "south_china_resin_tapping_restrictions",
                    "inferred_external_input": "gum_rosin_derivatives",
                    "required_world_knowledge": (
                        "Gum rosin and rosin esters from pine resin can be used in soldering flux formulations, "
                        "including low-residue electronics assembly processes."
                    ),
                    "required_concepts": ["resin", "rosin", "flux", "electronics"],
                },
                "hidden_dependency_path": ["rosin_ester_flux", "ASM-44", "MTG-220", "SO-7844"],
                "business_exposure": {
                    "revenue_at_risk": 3692000,
                    "affected_orders": ["SO-7844"],
                    "affected_customers": ["Mercy North Hospitals"],
                    "constrained_inventory": True,
                },
            }
        ],
        expected_no_alerts=[
            {"source_headline": "H02", "reason": "Seoul air cargo rates do not touch the active supplier path."},
            {"source_headline": "H05", "reason": "Tin price relief is not a risk and does not constrain the order."},
            {"source_headline": "H07", "reason": "Legacy LCD module demand is unrelated to active MTG-220 orders."},
        ],
        quality_notes="Positive case: the headline never says electronics or the company; the agent must bridge pine resin to rosin flux.",
    )


def _case_zero_alert() -> dict[str, Any]:
    records = _base_records(variant="zero")
    headlines = [
        "Lithium brine protests delay several South American pilot extraction projects.",
        "Port Klang reports residual yard congestion after weekend storms.",
        "Cocoa futures rise as West African crop estimates are revised downward.",
        "The EU publishes implementation guidance for battery passport reporting.",
        "Spot helium allocations tighten for a subset of party-balloon distributors.",
        "Beta Components raises list prices for legacy display modules.",
        "A rail operator announces temporary speed restrictions in the Upper Midwest.",
        "Analysts note a quiet week for North American water infrastructure orders.",
    ]
    return _make_case(
        run_id="LSR-027",
        date="2026-06-14",
        scenario="zero_alert_distractor",
        difficulty=4,
        company="Aster Vale Controls",
        headlines=headlines,
        records=records,
        true_alerts=[],
        expected_no_alerts=[
            {"source_headline": "H01", "reason": "Lithium is absent from company BOMs, suppliers, and open orders."},
            {"source_headline": "H02", "reason": "No active supplier lane or inventory location uses Port Klang."},
            {"source_headline": "H04", "reason": "Battery passport guidance does not apply to the company's listed products."},
            {"source_headline": "H06", "reason": "Legacy display modules are tied to discontinued LTH-90 and no active exposure."},
        ],
        quality_notes="Zero-alert case: multiple plausible external shocks, but the workspace lacks both dependency and current exposure.",
    )


def _case_silicon_metal() -> dict[str, Any]:
    records = _base_records(variant="positive")
    records["orders"] = {
        "orders": [
            {
                "id": "SO-7991",
                "customer": "Canal Rail",
                "product_id": "RSG-700",
                "quantity": 180,
                "unit_revenue": 18500,
                "due_date": "2026-07-02",
                "priority": "strategic",
                "status": "open",
            },
            {
                "id": "SO-7988",
                "customer": "Summit Clinics",
                "product_id": "MTG-220",
                "quantity": 20,
                "unit_revenue": 14200,
                "due_date": "2026-08-20",
                "priority": "standard",
                "status": "open",
            },
        ]
    }
    records["inventory"] = {
        "inventory": [
            {"product_id": "RSG-700", "location": "East DC", "quantity_available": 22, "status": "available"},
            {"product_id": "MTG-220", "location": "Midwest DC", "quantity_available": 130, "status": "available"},
            {"product_id": "CMC-410", "location": "West DC", "quantity_available": 145, "status": "available"},
        ]
    }
    headlines = [
        "Hydro curtailments in Quebec force several energy-intensive silicon metal furnaces offline.",
        "A truckload brokerage index shows easing spot rates across the Southeast.",
        "EU inspectors begin a paperwork campaign for selected medical-device labels.",
        "North American rail operators report normal intermodal terminal dwell times.",
        "A specialty alumina producer says maintenance at one kiln finished early.",
        "Weather services warn of heavy rain near Gulf Coast resin warehouses.",
        "A Canadian labor mediator schedules talks with port clerks next week.",
        "Analysts expect flat demand for municipal control systems through July.",
    ]
    return _make_case(
        run_id="LSR-033",
        date="2026-06-15",
        scenario="silicon_metal_energy_positive",
        difficulty=5,
        company="Aster Vale Controls",
        headlines=headlines,
        records=records,
        true_alerts=[
            {
                "id": "risk-silicone-potting",
                "source_headlines": ["H01"],
                "latent_edge": {
                    "world_event": "quebec_silicon_metal_curtailment",
                    "inferred_external_input": "silicone_elastomer_feedstock",
                    "required_world_knowledge": (
                        "Silicone elastomers are made from silicon chemistry, and upstream silicon metal supply can "
                        "matter for silicone base polymers used in potting compounds."
                    ),
                    "required_concepts": ["silicon metal", "silicone", "elastomer", "potting"],
                },
                "hidden_dependency_path": ["silicone_elastomer", "POT-72", "RSG-700", "SO-7991"],
                "business_exposure": {
                    "revenue_at_risk": 3330000,
                    "affected_orders": ["SO-7991"],
                    "affected_customers": ["Canal Rail"],
                    "constrained_inventory": True,
                },
            }
        ],
        expected_no_alerts=[
            {"source_headline": "H03", "reason": "Medical-device label paperwork does not affect the rail gateway exposure."},
            {"source_headline": "H05", "reason": "Alumina maintenance ending early is not a constraint for the current order."},
            {"source_headline": "H06", "reason": "Gulf Coast resin warehouses do not supply the RSG-700 potting path."},
        ],
        quality_notes="Positive case: the bridge is silicon metal to silicone elastomer, then potting compound to an open rail order.",
    )


def _case_antimony_flame_retardant() -> dict[str, Any]:
    records = _base_records(variant="positive")
    records["products"].append(
        {
            "id": "EVB-330",
            "name": "Emergency Ventilation Bridge",
            "family": "building_safety_controls",
            "region_eligibility": ["North America"],
            "certifications": ["plenum_fire_safety"],
            "status": "active",
        }
    )
    records["bom"]["boms"].append(
        {
            "product_id": "EVB-330",
            "components": [
                {
                    "component_id": "FJ-31",
                    "name": "plenum-rated cable harness",
                    "role": "plenum_fire_safety",
                    "inputs": ["antimony_trioxide_synergist", "low_smoke_jacket_compound"],
                },
                {"component_id": "CTRL-9", "name": "safety controller module"},
            ],
        }
    )
    records["suppliers"].append(
        {
            "id": "SUP-SAFELINE",
            "name": "SafeLine Interconnect",
            "sites": [{"id": "SITE-SAFELINE-IL", "country": "US"}],
            "supplies": [{"component_id": "FJ-31"}, {"input_id": "low_smoke_jacket_compound"}],
            "lead_time_days": 42,
            "status": "active",
            "tier2_notes": [
                "fire-safety jacket package includes an oxide synergist sourced through Asian compound distributors"
            ],
        }
    )
    records["orders"] = {
        "orders": [
            {
                "id": "SO-8207",
                "customer": "Metro Tunnel Authority",
                "product_id": "EVB-330",
                "quantity": 150,
                "unit_revenue": 22400,
                "due_date": "2026-07-09",
                "priority": "strategic",
                "status": "open",
            },
            {
                "id": "SO-8214",
                "customer": "Hill County Utilities",
                "product_id": "CMC-410",
                "quantity": 60,
                "unit_revenue": 9900,
                "due_date": "2026-09-02",
                "priority": "standard",
                "status": "forecast",
            },
        ]
    }
    records["inventory"] = {
        "inventory": [
            {"product_id": "EVB-330", "location": "Central DC", "quantity_available": 18, "status": "available"},
            {"product_id": "CMC-410", "location": "West DC", "quantity_available": 210, "status": "available"},
            {"product_id": "MTG-220", "location": "Midwest DC", "quantity_available": 120, "status": "available"},
            {"product_id": "RSG-700", "location": "East DC", "quantity_available": 90, "status": "available"},
        ]
    }
    headlines = [
        "Provincial inspectors in Hunan announce a new round of checks at several antimony smelters.",
        "Copper scrap spreads narrow after steady buying from wire mills.",
        "A Gulf Coast resin terminal reports two nights of maintenance on nonhazardous tanks.",
        "Municipal tunnel operators publish a reminder about summer ventilation inspections.",
        "A Northeast rail terminal reports normal dwell times after last week's staffing issue.",
        "A plastics compounder says demand for low-smoke building materials remains uneven.",
        "Airfreight forwarders add a surcharge on oversized medical imaging equipment.",
        "A standards committee opens comments on future labeling for battery recycling streams.",
    ]
    return _make_case(
        run_id="LSR-052",
        date="2026-06-17",
        scenario="antimony_flame_retardant_positive",
        difficulty=5,
        company="Aster Vale Controls",
        headlines=headlines,
        records=records,
        true_alerts=[
            {
                "id": "risk-antimony-plenum-harness",
                "source_headlines": ["H01"],
                "latent_edge": {
                    "world_event": "hunan_antimony_smelter_checks",
                    "inferred_external_input": "antimony_trioxide_flame_retardant_synergist",
                    "required_world_knowledge": (
                        "Antimony trioxide is commonly used as a synergist in flame-retardant plastic and cable "
                        "jacket formulations, so antimony smelter restrictions can matter to plenum-rated harnesses."
                    ),
                    "required_concepts": ["antimony", "trioxide", "flame", "cable"],
                },
                "hidden_dependency_path": ["antimony_trioxide_synergist", "FJ-31", "EVB-330", "SO-8207"],
                "business_exposure": {
                    "revenue_at_risk": 3360000,
                    "affected_orders": ["SO-8207"],
                    "affected_customers": ["Metro Tunnel Authority"],
                    "constrained_inventory": True,
                },
            }
        ],
        expected_no_alerts=[
            {"source_headline": "H02", "reason": "Copper scrap spreads do not map to the certified plenum harness path."},
            {"source_headline": "H03", "reason": "Gulf Coast resin maintenance does not touch SafeLine or EVB-330 inputs."},
            {"source_headline": "H04", "reason": "Tunnel inspection reminders create demand context, not a supply constraint."},
            {"source_headline": "H08", "reason": "Battery recycling labels do not apply to current Aster Vale products."},
        ],
        quality_notes=(
            "Positive case: the headline never mentions cables or the company; the agent must bridge "
            "antimony smelter checks to antimony trioxide flame-retardant synergy in a plenum-rated harness."
        ),
    )


def _case_latent_edge_no_current_exposure() -> dict[str, Any]:
    records = _base_records(variant="positive")
    records["orders"] = {
        "orders": [
            {
                "id": "SO-8120",
                "customer": "Prairie Water Authority",
                "product_id": "CMC-410",
                "quantity": 55,
                "unit_revenue": 9900,
                "due_date": "2026-08-18",
                "priority": "standard",
                "status": "forecast",
            },
            {
                "id": "SO-8126",
                "customer": "Summit Clinics",
                "product_id": "MTG-220",
                "quantity": 90,
                "unit_revenue": 14100,
                "due_date": "2026-07-22",
                "priority": "standard",
                "status": "completed",
            },
        ]
    }
    records["inventory"] = {
        "inventory": [
            {"product_id": "CMC-410", "location": "West DC", "quantity_available": 310, "status": "available"},
            {"product_id": "MTG-220", "location": "Midwest DC", "quantity_available": 140, "status": "available"},
            {"product_id": "RSG-700", "location": "East DC", "quantity_available": 105, "status": "available"},
        ]
    }
    headlines = [
        "Unseasonal drought conditions deepen across parts of Gujarat, pressuring several agricultural export markets.",
            "A clinical networking distributor reports normal replenishment after a quiet procurement week.",
        "Container carriers announce a blank sailing on one Asia-Europe service loop.",
        "Copper wire rod premiums soften after additional warehouse arrivals.",
        "A specialty packaging resin supplier says its Midwest warehouse is operating normally.",
        "Local officials in Oregon approve overnight road work near an industrial park.",
        "Battery recyclers ask regulators for more time on a reporting proposal.",
        "A display-module reseller lists excess stock for discontinued industrial panels.",
    ]
    return _make_case(
        run_id="LSR-041",
        date="2026-06-16",
        scenario="castor_polyol_no_current_exposure",
        difficulty=5,
        company="Aster Vale Controls",
        headlines=headlines,
        records=records,
        true_alerts=[],
        expected_no_alerts=[
            {
                "source_headline": "H01",
                "reason": (
                    "The castor-to-polyol-to-CR-88 dependency exists, but CMC-410 demand is forecast only, "
                    "available inventory covers it, and there are no current open CMC-410 orders."
                ),
            },
            {"source_headline": "H03", "reason": "The blank sailing does not map to active supplier lanes or open demand."},
            {"source_headline": "H07", "reason": "Battery reporting does not apply to the company product catalog."},
            {"source_headline": "H08", "reason": "Discontinued display panels do not create active business exposure."},
        ],
        quality_notes=(
            "No-alert control: the latent edge is real and the enterprise dependency is discoverable, "
            "but the case should not alert because current measurable exposure is missing."
        ),
    )


def _make_case(
    *,
    run_id: str,
    date: str,
    scenario: str,
    difficulty: int,
    company: str,
    headlines: list[str],
    records: dict[str, Any],
    true_alerts: list[dict[str, Any]],
    expected_no_alerts: list[dict[str, Any]],
    quality_notes: str,
) -> dict[str, Any]:
    headline_pack = _headline_pack(date, headlines)
    company_profile = _company_profile(company)
    workspace = {
        "public": {
            "headline_pack": headline_pack,
            "company_profile": company_profile,
            "products": {"products": records["products"]},
            "bom": records["bom"],
            "suppliers": {"suppliers": records["suppliers"]},
            "orders": records["orders"],
            "inventory": records["inventory"],
        },
        "tools": {
            "risk_tools": {
                "entrypoint": "python3 tools/risk_tools.py",
                "commands": ["read_headline_pack", "search_company_records", "read_record", "calculate_exposure", "submit_risk_scan"],
            }
        },
        "output_schema": OUTPUT_SCHEMA,
    }
    hidden = {
        "run_id": run_id,
        "true_alerts": true_alerts,
        "expected_no_alerts": expected_no_alerts,
        "zero_alert_run": len(true_alerts) == 0,
        "unexpected_valid_alert_protocol": {
            "allowed": True,
            "rule": "Credible evidence-backed alerts outside the key require human seed review rather than automatic failure.",
        },
    }
    benchmark_case = {
        "prompt": PROMPT,
        "setup": SETUP,
        "inputs": {
            "run_id": run_id,
            "date": date,
            "company": company,
            "headline_count": len(headlines),
            "workspace_template": "latent_supply_risk_v1",
        },
        "environment": {
            "name": "daily_supply_chain_risk_scan",
            "workspace_template": "latent_supply_risk_v1",
            "public_files": [
                "README.md",
                "events/headline-pack.md",
                "records/company-profile.md",
                "records/products.yaml",
                "records/bom.yaml",
                "records/suppliers.yaml",
                "records/orders.yaml",
                "records/inventory.yaml",
                "tools/risk_tools.py",
                "output/output_schema.json",
            ],
            "hidden_files": ["evaluation/hidden-answer-key.yaml"],
        },
        "workspace": workspace,
        "hidden_evaluation": hidden,
        "grader_contract": _grader_contract(),
    }
    return {
        "run_id": run_id,
        "scenario": scenario,
        "difficulty": difficulty,
        "quality_notes": quality_notes,
        "benchmark_case": benchmark_case,
    }


def _grader_contract() -> dict[str, Any]:
    return {
        "deterministic_checks": [
            "scan result matches whether hidden true_alerts exist",
            "alert source_headlines include required source headline",
            "latent_edge_hypothesis and real_world_prior mention required concepts",
            "enterprise_path includes hidden dependency path elements",
            "business_exposure matches affected orders, customers, revenue, and constrained inventory",
            "citations include headline and company record references",
            "no_alert cases contain no alerts and a rationale covering rejected distractors",
        ],
        "score_weights": {
            "result_correct": 20,
            "source_headline": 10,
            "latent_edge": 20,
            "enterprise_path": 20,
            "business_exposure": 15,
            "citations": 10,
            "calibrated_uncertainty": 5,
        },
        "hard_failures": ["invented_company_record", "writes_output_other_than_risk_scan", "unsupported_major_false_positive"],
    }


def _public_task_row(case: dict[str, Any]) -> dict[str, Any]:
    run_id = case["run_id"]
    benchmark = case["benchmark_case"]
    workspace_path = WORKSPACE_ROOT / run_id
    return {
        "schema_version": "case_v2",
        "id": run_id,
        "inputs": {
            "prompt": PROMPT,
            "run_id": run_id,
            "workspace_path": f"/workspace/task/{run_id}",
            "host_workspace_path": str(workspace_path),
            "workspace_files": benchmark["environment"]["public_files"],
            "output_schema": OUTPUT_SCHEMA,
        },
        "metadata": {
            "domain": "latent_supply_risk",
            "scenario": case["scenario"],
            "difficulty": case["difficulty"],
            "zero_alert_run": benchmark["hidden_evaluation"]["zero_alert_run"],
        },
        "resources": {
            "workspace": {
                "source": "container_image",
                "mode": "scratch",
                "image": "latent-supply-risk-workspaces:local",
                "workdir": f"/workspace/task/{run_id}",
            }
        },
        "materialization": [],
        "limits": {"timeout_ms": 900000},
    }


def _write_workspace(case: dict[str, Any]) -> None:
    run_id = case["run_id"]
    workspace = WORKSPACE_ROOT / run_id
    public = case["benchmark_case"]["workspace"]["public"]
    _reset_dir(workspace)
    _write_text(
        workspace / "README.md",
        "\n".join(
            [
                "# Daily Supply-Chain Risk Scan",
                "",
                SETUP,
                "",
                "## Task",
                PROMPT,
                "",
                "Use the workspace API exposed by `python3 tools/risk_tools.py`.",
                "Read the feed with `read_headline_pack`, inspect records with `search_company_records` and `read_record`, and measure exposure with `calculate_exposure`.",
                "Submit the final scan with `submit_risk_scan`; it writes `output/risk-scan.yaml`.",
                "Return the same final scan as JSON matching `output/output_schema.json`.",
            ]
        )
        + "\n",
    )
    _write_text(workspace / "events" / "headline-pack.md", public["headline_pack"])
    _write_text(workspace / "records" / "company-profile.md", public["company_profile"])
    _write_yaml(workspace / "records" / "products.yaml", public["products"])
    _write_yaml(workspace / "records" / "bom.yaml", public["bom"])
    _write_yaml(workspace / "records" / "suppliers.yaml", public["suppliers"])
    _write_yaml(workspace / "records" / "orders.yaml", public["orders"])
    _write_yaml(workspace / "records" / "inventory.yaml", public["inventory"])
    _write_json(workspace / "output" / "output_schema.json", OUTPUT_SCHEMA)
    _write_text(workspace / "tools" / "risk_tools.py", _risk_tools_source())
    (workspace / "tools" / "risk_tools.py").chmod((workspace / "tools" / "risk_tools.py").stat().st_mode | stat.S_IXUSR)


def _risk_tools_source() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Latent supply-risk workspace tools")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("read_headline_pack")
    search = sub.add_parser("search_company_records")
    search.add_argument("query")
    read = sub.add_parser("read_record")
    read.add_argument("path")
    exposure = sub.add_parser("calculate_exposure")
    exposure.add_argument("--entity-id", action="append", required=True)
    submit = sub.add_parser("submit_risk_scan")
    submit.add_argument("path")
    args = parser.parse_args()

    if args.command == "read_headline_pack":
        print((ROOT / "events" / "headline-pack.md").read_text(encoding="utf-8"))
        return 0
    if args.command == "search_company_records":
        query = args.query.lower()
        hits = []
        for path in sorted((ROOT / "records").glob("**/*")):
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                if query in text.lower():
                    hits.append({"path": str(path.relative_to(ROOT)), "excerpt": _excerpt(text, query)})
        print(json.dumps({"query": args.query, "hits": hits}, indent=2, sort_keys=True))
        return 0
    if args.command == "read_record":
        path = (ROOT / args.path).resolve()
        if ROOT not in path.parents and path != ROOT:
            raise SystemExit("record path escapes workspace")
        print(path.read_text(encoding="utf-8"))
        return 0
    if args.command == "calculate_exposure":
        print(json.dumps(_calculate_exposure(args.entity_id), indent=2, sort_keys=True))
        return 0
    if args.command == "submit_risk_scan":
        source = Path(args.path)
        target = ROOT / "output" / "risk-scan.yaml"
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        print(json.dumps({"written": str(target.relative_to(ROOT))}))
        return 0
    return 2


def _load_yaml(rel: str):
    text = (ROOT / rel).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise SystemExit(f"{rel} is not JSON and PyYAML is unavailable")
        return yaml.safe_load(text)


def _calculate_exposure(entity_ids: list[str]) -> dict:
    boms = _load_yaml("records/bom.yaml")["boms"]
    orders = _load_yaml("records/orders.yaml")["orders"]
    inventory = _load_yaml("records/inventory.yaml")["inventory"]
    products = set(entity_ids)
    for bom in boms:
        product_id = bom["product_id"]
        for component in bom.get("components", []):
            ids = {component.get("component_id"), *component.get("inputs", [])}
            if ids.intersection(entity_ids):
                products.add(product_id)
    affected_orders = [order for order in orders if order["status"] == "open" and order["product_id"] in products]
    inventory_by_product = {}
    for item in inventory:
        inventory_by_product[item["product_id"]] = inventory_by_product.get(item["product_id"], 0) + item["quantity_available"]
    revenue = sum(order["quantity"] * order["unit_revenue"] for order in affected_orders)
    constrained = any(inventory_by_product.get(order["product_id"], 0) < order["quantity"] for order in affected_orders)
    return {
        "entity_ids": entity_ids,
        "products": sorted(products),
        "open_orders": [order["id"] for order in affected_orders],
        "affected_customers": sorted({order["customer"] for order in affected_orders}),
        "revenue_at_risk": revenue,
        "constrained_inventory": constrained,
    }


def _excerpt(text: str, query: str) -> str:
    low = text.lower()
    idx = low.find(query)
    if idx < 0:
        return text[:160]
    return text[max(0, idx - 80): idx + 160].replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _headline_pack(date: str, items: list[str]) -> str:
    lines = ["# External Event Feed", f"# Date: {date}", ""]
    for index, item in enumerate(items, start=1):
        lines.extend([f"## H{index:02d}", item, ""])
    return "\n".join(lines)


def _company_profile(company: str) -> str:
    return f"""# {company}

{company} builds industrial controls, clinical networking gateways, and rail signal
hardware for North American and European customers. The company runs a daily
supply-chain risk scan against generic external headlines and compact operating
records.

Fulfillment model:
- final assembly in Ohio and Oregon;
- standard lead-time assumption is 30 to 45 days for specialty materials;
- strategic customers receive review priority when open demand exceeds available inventory.

Risk posture:
- raise review alerts when a plausible external event can be connected to a company
  material, component, product, and current business exposure;
- do not raise alerts for generic macro news without a company-specific dependency
  and measurable exposure.
"""


def _ensure_dirs() -> None:
    for path in [CASE_ROOT, TASK_ROOT, EVAL_ROOT, WORKSPACE_ROOT, CORPUS_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)


def _reset_dir(path: Path) -> None:
    if path.exists():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
