# Synthetic Data Pipeline Agents

A staged LangGraph pipeline that generates benchmark cases, validates them, and writes auditable run artifacts.

## Requirements

- Python 3.10+
- A model provider for live runs: OpenAI, Gemini, xAI/Grok, or Codex subscription auth
- The sibling `../substrate` repo, because `requirements.txt` installs `../substrate/packages/executioner-python`
- The `../substrate/target/debug/executioner` binary when a domain uses execution-backed validation

## Install

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create local settings:

   ```bash
   cp .env.example .env
   ```

4. Edit `.env` with one model provider. For OpenAI, the smallest useful setup is:

   ```text
   OPENAI_API_KEY=sk-...
   MODEL_PROVIDER=openai
   MODEL_NAME=gpt-5.5
   ```

   `.env.example` also shows Gemini, xAI/Grok, Codex auth, model timeout, base URL, and embedding options. Shell environment values override `.env`.

## Check The Install

Run the deterministic tests first. They do not need model provider credentials.

```bash
pytest
```

## Run The Pipeline

Run a small live pipeline with a fresh run id:

```bash
python3 main.py \
  --domain domains/benchmark_haiku.yaml \
  --target-stage benchmark \
  --target-n 5 \
  --seed 42 \
  --run-id smoke-run
```

Use a different domain by changing `--domain`:

```bash
python3 main.py \
  --domain domains/benchmark_code_debug.yaml \
  --target-stage benchmark \
  --target-n 3 \
  --seed 42 \
  --run-id code-smoke
```

Useful flags:

- `--provider` and `--model` override `.env` for one run.
- `--no-progress` hides the terminal progress graph.
- `--overwrite` replaces artifacts for an existing run id.
- `--auth-file` points Codex subscription auth at a specific auth file.

## Review A Run

Each run writes a corpus file and a log directory:

```text
data/corpus/benchmark/<run-id>.jsonl
logs/<run-id>/stage_records.jsonl
logs/<run-id>/validation.jsonl
logs/<run-id>/rejections.jsonl
logs/<run-id>/metrics.json
data/outputs/<run-id>.jsonl
```

Common review commands:

```bash
python3 analyze.py --run-id smoke-run
python3 run_report.py smoke-run
python3 sample_outputs.py smoke-run --limit 1
```

`main.py` will not reuse a run id if matching artifacts already exist. Pick a new run id, or use `--overwrite` when replacing an old run intentionally.

## How The Pipeline Works

The router owns pipeline state. Agents produce or judge artifacts, but they do not choose state transitions.

The main stages are:

1. Design benchmark case candidates.
2. Audit the design.
3. Generate a sample.
4. Run deterministic validation.
5. Search for adversarial failures.
6. Apply quality and rubric gates.
7. Curate the corpus.
8. Commit accepted samples.

For deeper details, see:

- [Pipeline state machine](docs/PIPELINE_STATE_MACHINE.md)
- [Pipeline artifact reference](docs/PIPELINE_REFERENCE.md)

## Project Map

```text
main.py                  CLI entrypoint
pipeline.py              Pipeline nodes, edges, and retry policy
router.py                Route table and routing context
agents.py                Agent role implementations
rules.py                 Deterministic benchmark checks
models.py                Pydantic artifact and event schemas
config.py                CLI, environment, and domain config
observability.py         Stage run log writer
analyze.py               Offline metrics
run_report.py            Human-readable run report
sample_outputs.py        Sample outputs for committed prompts
domains/                 Domain contracts
services/                Workspace, corpus, coverage, and validation services
tests/                   Deterministic tests
```
