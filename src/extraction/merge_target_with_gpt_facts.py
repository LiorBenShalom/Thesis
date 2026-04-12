#!/usr/bin/env python3

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


DEFAULT_BASE_DIR = Path(
    "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/drugs"
)
DEFAULT_PRIMARY_FACTS = Path("gpt/processed_verdicts_with_gpt.csv")
DEFAULT_SUPPLEMENTAL_FACTS = Path("n/gpt/processed_verdicts_with_gpt.csv")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Merge verdict similarity pairs with GPT indictment facts."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory containing target.csv and GPT outputs (defaults to the drugs dataset).",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        help="Path to CSV with verdict pairs and similarity scores (defaults to base-dir/target.csv).",
    )
    parser.add_argument(
        "--facts",
        type=Path,
        nargs="+",
        help=(
            "One or more CSV files with GPT indictment facts. "
            "If omitted, looks for gpt/processed_verdicts_with_gpt.csv and n/gpt/processed_verdicts_with_gpt.csv under base-dir."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (defaults to base-dir/target_with_gpt_facts.csv).",
    )
    return parser


def resolve_pairs_path(base_dir: Path, pairs_arg: Path | None) -> Path:
    pairs_path = pairs_arg or (base_dir / "target.csv")
    if not pairs_path.is_absolute():
        pairs_path = base_dir / pairs_path
    return pairs_path


def resolve_fact_paths(base_dir: Path, facts_arg: list[Path] | None) -> list[Path]:
    default_candidates = [DEFAULT_PRIMARY_FACTS, DEFAULT_SUPPLEMENTAL_FACTS]
    candidate_paths = facts_arg or default_candidates

    resolved_paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidate_paths:
        path = candidate if candidate.is_absolute() else base_dir / candidate
        path = path.resolve()
        if path in seen:
            continue
        resolved_paths.append(path)
        seen.add(path)
    return resolved_paths


def resolve_output_path(base_dir: Path, output_arg: Path | None) -> Path:
    output_path = output_arg or (base_dir / "target_with_gpt_facts.csv")
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    return output_path


def load_facts_csv(path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(path, usecols=["verdict", "extracted_gpt_facts"])
        .dropna(subset=["verdict", "extracted_gpt_facts"])
    )


def merge_pairs_with_facts(
    pairs_path: Path,
    fact_paths: list[Path],
    output_path: Path,
) -> None:
    if not pairs_path.exists():
        sys.exit(f"Pairs file not found: {pairs_path}")

    available_facts_paths = [path for path in fact_paths if path.exists()]
    missing_facts_paths = [path for path in fact_paths if not path.exists()]

    for missing_path in missing_facts_paths:
        print(f"Warning: facts file not found -> {missing_path}")

    if not available_facts_paths:
        sys.exit("No indictment facts files could be loaded.")

    pairs_df = pd.read_csv(pairs_path)
    facts_frames = [load_facts_csv(path) for path in available_facts_paths]

    combined_facts_df = (
        pd.concat(facts_frames, ignore_index=True)
        .drop_duplicates(subset="verdict", keep="first")
    )

    facts_series = combined_facts_df.set_index("verdict")["extracted_gpt_facts"]

    pairs_df["verdict_1_gpt_facts"] = pairs_df["verdict_1"].map(facts_series)
    pairs_df["verdict_2_gpt_facts"] = pairs_df["verdict_2"].map(facts_series)

    missing_v1 = pairs_df["verdict_1_gpt_facts"].isna().sum()
    missing_v2 = pairs_df["verdict_2_gpt_facts"].isna().sum()

    pairs_df.to_csv(output_path, index=False)

    print(f"Saved merged dataset to {output_path}")
    print(f"Missing GPT facts -> verdict_1: {missing_v1}, verdict_2: {missing_v2}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    pairs_path = resolve_pairs_path(base_dir, args.pairs)
    fact_paths = resolve_fact_paths(base_dir, args.facts)
    output_path = resolve_output_path(base_dir, args.output)

    merge_pairs_with_facts(pairs_path, fact_paths, output_path)


if __name__ == "__main__":
    main()

