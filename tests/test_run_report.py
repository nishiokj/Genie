from __future__ import annotations

from run_report import _case_type, _prompt, _semantic_check_lines


def test_report_reads_current_benchmark_shape_only() -> None:
    candidate = {
        "case_type": "proxy_strong",
        "benchmark_case": {"prompt": "Debug the execution workspace."},
    }

    assert _prompt(candidate) == "Debug the execution workspace."
    assert _case_type(candidate) == "proxy_strong"


def test_semantic_check_lines_use_subcodes_only() -> None:
    lines = _semantic_check_lines(
        [
            {
                "check_kind": "quality",
                "verdict": "reject",
                "subcodes": ["weak_proxy_validity"],
                "rationale": "Too thin.",
            }
        ]
    )

    assert lines == ['quality: reject codes=weak_proxy_validity rationale="Too thin."']
