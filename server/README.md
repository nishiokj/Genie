# Domain Author (Nova daemon + substrate)

Two-service interview app that drives a **Nova daemon over TCP** via the
official Python client. Nova has substrate as its tool-execution backend, so
every file the model writes via `apply_patch` lands in substrate's effects
ledger AND on disk through the shared workspace volume.

## Architecture

- **substrate executioner host** (port 8765) — long-running. One *shared*
  environment (`interview_shared`) bound to a host directory at
  `/tmp/nova-substrate-shared`. Pre-created by `dev.sh` so Nova can attach.
- **nova-service** (docker, port 9556) — the Nova daemon. Started by `dev.sh`
  with env vars `NOVA_TOOL_EXECUTION_BACKEND=substrate`,
  `NOVA_SUBSTRATE_HOST_BASE_URL=http://127.0.0.1:8765/`,
  `NOVA_SUBSTRATE_ENVIRONMENT_ID=interview_shared`. The Nova daemon's
  attach-over-HTTP branch picks these up at startup and routes its tool
  executors (Read/Write/Edit/Bash/Glob/Grep/apply_patch) through substrate.
- **agent** (FastAPI, port 8001) — owns interview sessions. Each session
  reserves a subdirectory under the shared workspace (`/workspace/<sid>`)
  and a Nova session key. Per user turn it calls
  `nova_client.run_to_completion(prompt, session_key=..., working_dir=...)`.
  Auto-approves permission requests via the event callback.
- **app** (FastAPI, port 8000) — proxies turns to the agent, serves the UI,
  and on Generate copies the finalized `draft.yaml` to `domains/<id>.yaml`.

### How a turn flows

1. UI → `POST /api/sessions/<sid>/message` → app proxies to agent.
2. Agent invokes the shared `NovaClient` with the kickoff/turn prompt.
3. Nova picks the `apply_patch` tool (in the standard agent's tool list).
4. The Nova patch we shipped in `apply_patch.ts` detects the substrate session
   on the tool execution context and dispatches each Add op as
   `env.submit({toolName:'Write', arguments:{path, content}})`.
5. Substrate executes the Write, records a `file.write` effect, and lands
   the bytes on disk in `/tmp/nova-substrate-shared/<sid>/`.
6. Agent reads `draft.yaml` from that path and returns it to the UI.

## Prerequisites

- `substrate-sdk` from PyPI. `server/dev.sh` resolves the bundled
  `substrate-runtime` binary from the Python package unless `SUBSTRATE_BIN` is
  set explicitly.
- Nova source repo at `../agent` with the substrate-attach + apply_patch-bridge
  patches applied. Build:
  ```bash
  cd ../agent
  bun run --cwd packages/core/tools build
  bun run --cwd packages/core/llm   build
  bun run --cwd packages/core/agent build
  bun run --cwd packages/infra/harness-daemon build
  docker build -f Dockerfile.nova-service -t nova-service:local .
  ```
- Docker (OrbStack works) for the Nova daemon container.
- `~/.config/nova/codex-auth.json` populated for codex provider access.

## Run

```bash
./server/dev.sh
# open http://127.0.0.1:8000
```

### Env knobs

| var | default | purpose |
|-----|---------|---------|
| `SUBSTRATE_HOST_URL` | `http://127.0.0.1:8765/` | executioner host URL |
| `SUBSTRATE_ENVIRONMENT_ID` | `interview_shared` | shared substrate env id |
| `SUBSTRATE_WORKSPACE_ROOT` | `/tmp/nova-substrate-shared` | host dir bound as substrate `/workspace` |
| `NOVA_HOST` / `NOVA_PORT` | `127.0.0.1` / `9556` | nova daemon TCP coords |
| `NOVA_IMAGE` | `nova-service:local` | docker image tag |
| `NOVA_REPO` | `../agent` | nova source repo (for nova-client install) |
| `NOVA_TURN_TIMEOUT_S` | `300` | per-turn timeout against Nova |
| `DOMAINS_DIR` | `<repo>/domains` | where finalized YAMLs land |

## Layout

```
server/
  dev.sh              # launcher (substrate host + Nova container + Python services)
  agent/
    main.py           # FastAPI: sessions, message, get
    interview.py      # NovaClient driver — run_to_completion + permission auto-approve
    state.py          # Session + per-session workspace subdir + nova_session_key
    substrate.py      # ensure_shared_environment() + host_healthy()
    prompts.py        # kickoff + per-turn prompts (apply_patch instructions)
    requirements.txt
    Dockerfile        # (parked; dev runs python services on host)
  app/
    main.py           # proxy + UI + generate
    ui/index.html
    requirements.txt
```

## Known limits / TODO

- The Nova patch routes apply_patch **Add** ops through substrate. Update and
  Delete still hit native `fs` (substrate's existing-mode binding means they
  still land in the right host dir, just without an effect entry). Worth
  finishing once we need diff-based edits.
- Auto-approve sends `decision='always_allow'`. Fine for the headless backend;
  not a security boundary on its own — substrate's policy is.
- GraphD-leakage / `session_not_found` showing up as tool results is a Nova
  bug we haven't fixed yet (see earlier notes).
- Pipeline execution after Generate is still manual (writes the YAML, prints
  the `python3 main.py ...` command).
