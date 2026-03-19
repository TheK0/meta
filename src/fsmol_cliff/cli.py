from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .aggregate import aggregate_task_result_rows, macro_mean
from .adapters import diagnose_official_adapter_availability
from .episodes import build_adversarial_episode
from .hypotheses import validate_h1, validate_h2, validate_h3
from .io import write_json
from .metrics import c_bacc, nc_bacc, nc_psr, q_psr, scr, sq_psr, ss_q_psr, ss_scr, ss_sq_psr
from .models import PairRecord
from .pipeline import build_assay_asset_bundle
from .fetch import write_source_manifest
from .release import build_release_bundle
from .reports import render_markdown_report
from .runner import evaluate_release_with_sklearn_baseline
from .constants import EpisodeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fsmol-cliff")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch-fsmol")
    fetch_parser.add_argument("--data-dir", required=True)
    fetch_parser.add_argument("--source-url")
    fetch_parser.add_argument("--task-list-file")
    fetch_parser.add_argument("--fsmol-data-version", default="fsmol-0.1")
    fetch_parser.add_argument(
        "--output",
        default="benchmark_manifest.source.json",
        help="Path for the captured FS-Mol source manifest.",
    )
    adapter_status_parser = subparsers.add_parser("adapter-status")
    adapter_status_parser.add_argument("--output", required=True)
    build_assets_parser = subparsers.add_parser("build-assets")
    build_assets_parser.add_argument("--task-file", required=True)
    build_assets_parser.add_argument("--output-dir", required=True)

    build_release_parser = subparsers.add_parser("build-release")
    build_release_parser.add_argument("--data-dir", required=True)
    build_release_parser.add_argument("--output-dir", required=True)
    build_release_parser.add_argument("--task-list-file")
    build_release_parser.add_argument("--fsmol-data-version", default="<fixed_version>")
    build_release_parser.add_argument("--support-per-class", type=int, default=16)
    build_release_parser.add_argument("--query-per-class", type=int, default=16)
    build_release_parser.add_argument("--episodes-per-split", type=int, default=400)
    build_release_parser.add_argument("--seeds", default="[0, 1, 2, 3, 4]")

    build_episodes_parser = subparsers.add_parser("build-episodes")
    build_episodes_parser.add_argument("--input", required=True)
    build_episodes_parser.add_argument("--output", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--input")
    evaluate_parser.add_argument("--release-dir")
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--split-types", default='["standard", "adversarial"]')
    evaluate_parser.add_argument("--model-name", default="kNN")
    evaluate_parser.add_argument("--model-params", default="{}")
    evaluate_parser.add_argument("--backend", choices=["local", "official"], default="local")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--input", required=True)
    aggregate_parser.add_argument("--output", required=True)

    validate_parser = subparsers.add_parser("validate-hypotheses")
    validate_parser.add_argument("--input", required=True)
    validate_parser.add_argument("--output", required=True)
    validate_parser.add_argument("--report")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "fetch-fsmol":
        write_source_manifest(
            output_path=Path(args.output),
            data_dir=Path(args.data_dir),
            source_url=args.source_url,
            task_list_file=Path(args.task_list_file) if args.task_list_file else None,
            fsmol_data_version=args.fsmol_data_version,
        )
    elif args.command == "adapter-status":
        write_json(Path(args.output), diagnose_official_adapter_availability())
    elif args.command == "build-assets":
        build_assay_asset_bundle(
            task_file=Path(args.task_file),
            output_dir=Path(args.output_dir),
        )
    elif args.command == "build-release":
        build_release_bundle(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            task_list_file=Path(args.task_list_file) if args.task_list_file else None,
            episode_config=EpisodeConfig(
                support_per_class=args.support_per_class,
                query_per_class=args.query_per_class,
            ),
            seeds=json.loads(args.seeds),
            episodes_per_split=args.episodes_per_split,
            fsmol_data_version=args.fsmol_data_version,
        )
    elif args.command == "build-episodes":
        payload = json.loads(Path(args.input).read_text())
        episode = build_adversarial_episode(
            support_pos_ids=payload["support_pos_ids"],
            support_neg_ids=payload["support_neg_ids"],
            query_pos_ids=payload["query_pos_ids"],
            query_neg_ids=payload["query_neg_ids"],
            cliff_pairs=[PairRecord(**pair) for pair in payload["cliff_pairs"]],
            anchor_to_hardnegs=payload["anchor_to_hardnegs"],
        )
        write_json(Path(args.output), None if episode is None else episode.to_dict())
    elif args.command == "evaluate":
        if args.release_dir:
            evaluate_release_with_sklearn_baseline(
                release_dir=Path(args.release_dir),
                output_path=Path(args.output),
                split_types=json.loads(args.split_types),
                model_name=args.model_name,
                model_params=json.loads(args.model_params),
                backend=args.backend,
            )
        else:
            payload = json.loads(Path(args.input).read_text())
            query_pairs = [PairRecord(**pair) for pair in payload.get("query_pairs", [])]
            noncliff_pairs = [PairRecord(**pair) for pair in payload.get("noncliff_pairs", [])]
            support_query_pairs = [PairRecord(**pair) for pair in payload.get("support_query_pairs", [])]
            labels = payload["labels"]
            scores = payload["scores"]
            predictions = payload["predictions"]
            summary = {
                "c_bacc": c_bacc(None, labels, predictions, payload.get("cliff_query_ids", [])),
                "nc_bacc": nc_bacc(None, labels, predictions, payload.get("noncliff_query_ids", [])),
                "q_psr": q_psr(None, query_pairs, scores),
                "nc_psr": nc_psr(None, noncliff_pairs, scores),
                "sq_psr": sq_psr(None, support_query_pairs, scores),
                "scr": scr(None, query_pairs + noncliff_pairs, predictions),
                "ss_q_psr": ss_q_psr(None, query_pairs, scores),
                "ss_scr": ss_scr(None, query_pairs + noncliff_pairs, predictions),
                "ss_sq_psr": ss_sq_psr(None, support_query_pairs, scores),
            }
            write_json(Path(args.output), summary)
    elif args.command == "aggregate":
        input_path = Path(args.input)
        if input_path.suffix == ".parquet":
            payload = pd.read_parquet(input_path).to_dict(orient="records")
            write_json(Path(args.output), aggregate_task_result_rows(payload))
        else:
            payload = json.loads(input_path.read_text())
            write_json(Path(args.output), macro_mean(payload))
    elif args.command == "validate-hypotheses":
        payload = json.loads(Path(args.input).read_text())
        results = {
            "h1": validate_h1(payload),
            "h2": validate_h2(payload),
            "h3": validate_h3(payload),
        }
        write_json(Path(args.output), results)
        if args.report:
            Path(args.report).write_text(
                render_markdown_report(
                    benchmark_version="v3.0",
                    aggregate_rows=[],
                    hypothesis_results=results,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
