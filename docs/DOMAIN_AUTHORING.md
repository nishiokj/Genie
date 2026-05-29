# Authoring A New Domain

A domain is a single YAML file in `domains/`. It is the only thing a user brings to spin up a new environment — there is no domain-specific Python to write unless the domain needs code execution.

The pipeline reads the YAML at startup and feeds its contents into the design, audit, generation, and gating prompts. If a section is weak, every case the pipeline produces in that domain will be weak in the same way.

## Quickstart

```bash
cp domains/benchmark_haiku.yaml domains/<your_domain>.yaml
# edit the sections listed below
python3 main.py --domain domains/<your_domain>.yaml --target-n 1 --run-id probe-1
```

Open `logs/probe-1/rejections.jsonl` after the run. The rejection codes and judge evidence tell you which YAML section is underspecified. Iterate the YAML, bump `--run-id`, repeat.

Do not add top-level keys that are not in the template — the loader will ignore them.

## What The YAML Contains

### Mechanical scaffolding (copy from the template, rename)

- `domain_id` — unique slug; must match the filename stem.
- `case_types`, `difficulties`, `scenarios` — enums the building agents constrain themselves to.
- `route_codes`, `subcodes` — only extend if the domain has a failure mode the router must distinguish from existing ones.
- `novelty_threshold`, `max_design_retries`, `max_generation_retries` — retry and dedup knobs.

### Content the author must write

1. **`abilities`** — the capabilities the benchmark proxies. This is the `ability_z` the whole pipeline reasons about.
2. **`environments`** — the world the agent operates in. This is `environment_y`.
3. **`diagnostic_pressure_types`** — the *kinds* of pressure cases may apply. The single highest-leverage field; weak entries here cap the quality of every generated case.
4. **`deterministic_rules`** — hard checks the router runs without an LLM (min lengths, required arrays, banned tokens). Cheap floor.
5. **`semantic_rules`** — natural-language constraints the PlanAuditor enforces. Where the author encodes what counts as a real case in this domain.
6. **`general_probe_principles`** — definitions plus good/bad examples for things like `meaningful_constraint_pressure` and `shallow_strategy_resistance`. Pasted into every audit and gate prompt; this is how the author teaches judges what excellence looks like in the domain.
7. **`generator_guidance`** — the goal, `scoring_contract_bar`, `proxy_claim_bar`, and `common_rejection_patterns`. The generator's spec; without it, prompts collapse to generic shapes.
8. **`quality_gate_rules`** and **`rubric_gate_rules`** — what the two gates accept and reject. The split is load-bearing: QualityGate judges proxy validity, RubricGate judges scoring reliability. Do not collapse them.
9. **`benchmark_case_schema`** — JSON Schema for what a case looks like in this environment. For haiku it is `{prompt: str}`. For a more structured domain it may include emails, ledgers, code snippets, etc.
10. **`output_schema_path`** — points at the JSON schema for the full agent+judge artifact. The shared `schemas/benchmark_output.schema.json` covers most domains; only override when the judge needs different fields.

### Execution-backed domains only

If cases must be executed (compiled, run, diffed), set `runtime_requirements.kind` in generated cases and provide a harness via the substrate executioner. `domains/benchmark_code_debug.yaml` is the reference. Skip this section entirely for non-execution domains.

## The 80/20

`diagnostic_pressure_types`, `generator_guidance`, and `semantic_rules` are where the domain actually lives. Everything else is plumbing copied from the template. Spend authoring time on those three.
