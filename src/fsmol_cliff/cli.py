from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fsmol-cliff")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fetch-fsmol")
    subparsers.add_parser("build-assets")
    subparsers.add_parser("build-episodes")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("aggregate")
    subparsers.add_parser("validate-hypotheses")
    return parser


def main() -> int:
    parser = build_parser()
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
