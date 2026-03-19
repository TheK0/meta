from __future__ import annotations

from fsmol_cliff.reports import render_markdown_report


def test_render_markdown_report_includes_metrics_and_hypothesis_verdicts() -> None:
    report = render_markdown_report(
        benchmark_version="v3.0",
        aggregate_rows=[
            {
                "split_type": "standard",
                "metric": "q_psr",
                "score": 0.61,
                "ci_low": 0.55,
                "ci_high": 0.67,
                "num_tasks": 30,
            }
        ],
        hypothesis_results={
            "h1": {"accepted": True},
            "h2": {"accepted": False},
        },
    )

    assert "# FS-Mol-Cliff v3.0 Report" in report
    assert "| standard | q_psr | 0.6100 | 0.5500 | 0.6700 | 30 |" in report
    assert "| h1 | accepted |" in report
    assert "| h2 | rejected |" in report
