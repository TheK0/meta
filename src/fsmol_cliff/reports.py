from __future__ import annotations

from typing import Mapping, Sequence


def render_markdown_report(
    *,
    benchmark_version: str,
    aggregate_rows: Sequence[Mapping],
    hypothesis_results: Mapping[str, Mapping],
) -> str:
    lines = [f"# FS-Mol-Cliff {benchmark_version} Report", "", "## Aggregate Metrics", ""]
    lines.append("| Split | Metric | Score | CI Low | CI High | Tasks |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in aggregate_rows:
        lines.append(
            "| {split_type} | {metric} | {score:.4f} | {ci_low:.4f} | {ci_high:.4f} | {num_tasks} |".format(
                **row
            )
        )
    lines.extend(["", "## Hypotheses", "", "| Hypothesis | Verdict |", "| --- | --- |"])
    for hypothesis, result in hypothesis_results.items():
        verdict = "accepted" if result.get("accepted") else "rejected"
        lines.append(f"| {hypothesis} | {verdict} |")
    lines.append("")
    return "\n".join(lines)


def render_hypothesis_markdown(
    *,
    benchmark_version: str,
    hypothesis_results: Mapping[str, Mapping],
) -> str:
    return render_markdown_report(
        benchmark_version=benchmark_version,
        aggregate_rows=[],
        hypothesis_results=hypothesis_results,
    )
