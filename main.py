from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from cli_graph import create_graph_callback
from config import DomainConfig, build_runtime_config
from models import DesignBrief, GenerationEnvelope, GenerationPipelineInput, TaxonomyCell
from observability import set_event_callback
from pipeline import PipelineRunner
from provider_errors import ProviderError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic data pipeline.")
    parser.add_argument("--domain", required=True, help="Path to domain YAML.")
    parser.add_argument("--target-stage", default="benchmark", choices=["benchmark"])
    parser.add_argument("--target-n", type=int, default=5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--auth-file", default=None, help="Provider auth file path. Codex defaults to ~/.codex/auth.json.")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--instruction", default=None, help="Steer the normal design stage toward a specific benchmark intent.")
    parser.add_argument(
        "--from-instruction",
        default=None,
        help="Skip design/audit and generate one benchmark case from this instruction within the selected domain.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="auto")
    parser.add_argument("--no-progress", action="store_true", help="Disable compact stdout progress lines.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing artifacts for this run id.")
    args = parser.parse_args()
    if args.instruction and args.from_instruction:
        parser.error("--instruction and --from-instruction are mutually exclusive")

    run_id = args.run_id
    if run_id == "auto":
        run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")

    existing = _existing_run_artifacts(run_id)
    if existing and not args.overwrite:
        print(f"run id already has artifacts: {run_id}", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        print("Use a fresh --run-id or pass --overwrite.", file=sys.stderr)
        return 1
    if existing and args.overwrite:
        _clear_run_artifacts(run_id)

    config = build_runtime_config(
        domain_path=args.domain,
        target_stage=args.target_stage,
        target_n=args.target_n,
        seed=args.seed,
        run_id=run_id,
        model=args.model,
        provider=args.provider,
        auth_file=args.auth_file,
        embedding_model=args.embedding_model,
        console_progress=not args.no_progress,
        instruction=args.instruction,
    )
    if config.console_progress:
        set_event_callback(create_graph_callback(run_id))
    try:
        runner = PipelineRunner(config)
        if args.from_instruction:
            result = runner.run_from_generation(
                GenerationPipelineInput(
                    envelope=GenerationEnvelope.from_design(
                        _design_from_instruction(args.from_instruction, config.domain, run_id=run_id),
                        envelope_id=f"{run_id}-instruction-envelope",
                        domain_ref=str(config.domain_path),
                        seed_context={"user_instruction": args.from_instruction},
                    ),
                    output_dir=config.logs_dir / run_id,
                )
            )
            print(f"run_id={result.run_id}")
            print(f"status={result.final_status}")
            print(f"committed={result.committed}")
            print(f"dropped={result.dropped}")
            print(f"corpus={result.corpus_path}")
            print(f"logs={result.logs_dir / 'stage_records.jsonl'}")
            print(f"result={result.result_path}")
            return 0 if result.committed >= 1 else 1
        summary = runner.run()
    except ProviderError as exc:
        set_event_callback(None)
        print(f"provider error: {exc}", file=sys.stderr)
        return 2
    finally:
        set_event_callback(None)

    print(f"run_id={summary['run_id']}")
    print(f"committed={summary['committed']}")
    print(f"dropped={summary['dropped']}")
    print(f"corpus=data/corpus/benchmark/{summary['run_id']}.jsonl")
    print(f"logs=logs/{summary['run_id']}/stage_records.jsonl")
    return 0 if summary["committed"] >= args.target_n else 1


def _design_from_instruction(instruction: str, domain: DomainConfig, *, run_id: str) -> DesignBrief:
    return DesignBrief.create(
        design_id=f"{run_id}-instruction-design",
        cell=TaxonomyCell(
            case_type=_first(domain.case_types, "proxy_strong"),
            difficulty=_middle_difficulty(domain.difficulties),
            scenario=_first(domain.scenarios, "nominal"),
        ),
        target_ability=_first(domain.abilities, "domain_target_ability"),
        target_environment=_first(domain.environments, "single_turn"),
        design_intent=instruction,
        environment_premise={"user_instruction": instruction},
        runtime_requirements={
            "kind": "text_only",
            "execution": {"mode": "none", "os": "none", "arch": "none"},
            "language": {"name": "none", "version": "none"},
            "dependencies": {"policy": "none", "packages": []},
            "commands": {},
            "network": "not_applicable",
        },
        failure_mode_family="user-specified benchmark intent with shallow compliance risk",
        diagnostic_pressure=[instruction],
        why_weak_agents_fail=["they satisfy surface wording without demonstrating the requested capability"],
        tempting_shallow_solutions=["generic answer that matches the topic but not the benchmark intent"],
        success_evidence_required=["observable output behavior that satisfies the user instruction and domain scoring rules"],
        minimum_depth_requirements=["preserve the user instruction while creating a valid benchmark case"],
        forbidden_shortcuts=["surface-level compliance", "generic template"],
        non_goals=["benchmarking abilities outside the selected domain"],
    )


def _first(values: list[str], fallback: str) -> str:
    return values[0] if values else fallback


def _middle_difficulty(values: list[int]) -> int:
    if not values:
        return 3
    return sorted(values)[len(values) // 2]


def _existing_run_artifacts(run_id: str) -> list[Path]:
    paths = [
        Path("logs") / run_id,
        Path("data") / "corpus" / "benchmark" / f"{run_id}.jsonl",
    ]
    return [path for path in paths if path.exists()]


def _clear_run_artifacts(run_id: str) -> None:
    for path in _existing_run_artifacts(run_id):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
