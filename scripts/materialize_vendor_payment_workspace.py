from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "data/vendor_payment_exception/tasks/hero-vendor-v1-seed-curated.public_task_rows.jsonl"
DEFAULT_ORACLES = ROOT / "data/vendor_payment_exception/evaluation/hero-vendor-v1-seed-curated.hidden_oracles.json"
DEFAULT_OUT = ROOT / "data/vendor_payment_exception/workspaces"
DEFAULT_AUDIT_DIR = ROOT / "data/vendor_payment_exception/audit_views"


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize vendor payment benchmark rows as runnable workspaces.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--oracles", default=str(DEFAULT_ORACLES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--case-id", default=None)
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.tasks))
    if args.case_id:
        rows = [row for row in rows if row["id"] == args.case_id]
    if not rows:
        raise SystemExit("no matching task rows")

    oracles = json.loads(Path(args.oracles).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    for row in rows:
        case_id = row["id"]
        workspace = out_dir / case_id
        _reset_dir(workspace)
        _write_workspace(row, oracles[case_id]["hidden_oracle"], workspace)
        print(workspace)
    return 0


def _write_workspace(row: dict[str, Any], oracle: dict[str, Any], workspace: Path) -> None:
    public = row["workspace"]["public"]
    _write_text(
        workspace / "README.md",
        "\n".join(
            [
                "# Vendor Payment Exception Case",
                "",
                row["setup"],
                "",
                "## Task",
                row["prompt"],
                "",
                "Write your final answer to `decision.json` in this directory.",
                "The answer must match `output_schema.json`.",
                "",
                "Use `python3 tools/audit_vendor.py --vendor-id <id> --view <view>` for targeted vendor-record checks.",
                "Available audit views: profile_summary, change_history, payment_history, tax_profile, approval_chain, duplicate_scan.",
            ]
        )
        + "\n",
    )
    _write_text(workspace / "inbox/thread.md", str(public["email_thread"]) + "\n")
    _write_json(workspace / "records/vendor_profile.json", public["vendor_profile"])
    _write_json(workspace / "records/invoice.json", public["invoice"])
    _write_json(workspace / "records/purchase_order.json", public["purchase_order"])
    _write_text(workspace / "records/contract_excerpt.md", str(public["contract_excerpt"]) + "\n")
    _write_json(workspace / "records/ap_policy.json", public["ap_policy"])
    _write_json(workspace / "records/erp_snapshot.json", public["erp_snapshot"])
    _write_json(workspace / "output_schema.json", row["workspace"]["output_schema"])
    _write_json(DEFAULT_AUDIT_DIR / f"{row['id']}.json", oracle["audit_views"])
    _write_text(workspace / "tools/audit_vendor.py", _audit_tool_source())
    (workspace / "tools/audit_vendor.py").chmod(
        (workspace / "tools/audit_vendor.py").stat().st_mode | stat.S_IXUSR
    )
    _write_json(workspace / "trial_input.json", _trial_input(row))


def _trial_input(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "prompt": (
            "You are inside a materialized AP case workspace. Read README.md and the records directory, "
            "run the audit tool when vendor-record history matters, and return only the final structured "
            "answer JSON. If file writing is available, also save that same JSON to decision.json."
        ),
        "case": {
            "id": row["id"],
            "vendor": row["inputs"]["vendor"],
            "invoice_id": row["inputs"]["invoice_id"],
            "amount": row["inputs"]["amount"],
            "requested_action": row["inputs"]["requested_action"],
        },
        "output_schema_path": "output_schema.json",
        "workspace_files": [
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
    }


def _audit_tool_source() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Return targeted vendor-record audit views.")
    parser.add_argument("--vendor-id", required=True)
    parser.add_argument("--view", required=True)
    args = parser.parse_args()

    audit_path = Path(__file__).resolve().parents[3] / "audit_views" / f"{args.vendor_id}.json"
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    view = data.get(args.view)
    if view is None:
        print(json.dumps({"error": "unknown_view", "available_views": sorted(data)}))
        return 0
    if isinstance(view, dict) and view.get("vendor_id") not in (None, args.vendor_id):
        print(json.dumps({"error": "vendor_mismatch", "vendor_id": args.vendor_id}))
        return 3
    print(json.dumps({"vendor_id": args.vendor_id, "view": args.view, "result": view}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
