from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


# FEATURE_ID_COLUMN = "שם קובץ התיק"
FEATURE_ID_COLUMN = "מספר תיק"


def load_feature_map(features_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the feature schema CSV and build a mapping from verdict id to features."""
    features_df = pd.read_csv(features_path)

    if FEATURE_ID_COLUMN not in features_df.columns:
        raise ValueError(
            f"Could not find the feature id column '{FEATURE_ID_COLUMN}' in {features_path}"
        )

    features_df = features_df.drop_duplicates(subset=FEATURE_ID_COLUMN, keep="first")
    feature_columns = [col for col in features_df.columns if col != FEATURE_ID_COLUMN]

    feature_map: Dict[str, Dict[str, Any]] = {}
    for _, row in features_df.iterrows():
        verdict_id = str(row[FEATURE_ID_COLUMN]).strip()
        cleaned_features = {
            col: row[col]
            for col in feature_columns
            if pd.notna(row[col]) and str(row[col]).strip() != ""
        }
        feature_map[verdict_id] = cleaned_features

    return feature_map


def feature_vector_as_json(feature_map: Dict[str, Dict[str, Any]], verdict_id: str) -> str:
    """Return a JSON string representation of the feature vector for the given verdict."""
    features = feature_map.get(str(verdict_id).strip(), {})
    return json.dumps(features, ensure_ascii=False, sort_keys=True)


def ensure_directory(path: Path) -> None:
    """Create parent directories for the given path if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_similarity_frame(target_path: Path) -> pd.DataFrame:
    """Load the target CSV and enrich it with derived similarity columns."""
    target_df = pd.read_csv(target_path)

    required_columns = {"verdict_1", "verdict_2", "similarity"}
    missing_columns = required_columns.difference(target_df.columns)
    if missing_columns:
        raise ValueError(
            f"Target CSV {target_path} is missing required columns: {sorted(missing_columns)}"
        )

    target_df["verdict_1"] = target_df["verdict_1"].astype(str).str.strip()
    target_df["verdict_2"] = target_df["verdict_2"].astype(str).str.strip()

    target_df["similarity_scale"] = target_df["similarity"]

    # Build ordinal-friendly binary targets that stay strictly in {0, 1}.
    binary_0_map = {1: 0, 2: 0, 3: 1}
    binary_1_map = {1: 0, 2: 1, 3: 1}

    if invalid := sorted(
        {value for value in target_df["similarity_scale"].unique() if value not in binary_0_map}
    ):
        raise ValueError(
            f"Unexpected similarity values {invalid} in {target_path}. "
            "Update the binary mapping logic to reflect the scale."
        )

    target_df["similarity_binary_0"] = target_df["similarity_scale"].map(binary_0_map).astype(int)
    target_df["similarity_binary_1"] = target_df["similarity_scale"].map(binary_1_map).astype(int)

    return target_df


def build_feature_vector_dataset(
    base_df: pd.DataFrame,
    feature_map: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> pd.DataFrame:
    """Create a dataset with feature vectors for both verdicts in each pair."""
    output_columns = [
        "verdict_1",
        "verdict_2",
        "similarity_scale",
        "similarity_binary_0",
        "similarity_binary_1",
    ]

    output_df = base_df[output_columns].copy()
    output_df["feature_vector_1"] = output_df["verdict_1"].map(
        lambda verdict_id: feature_vector_as_json(feature_map, verdict_id)
    )
    output_df["feature_vector_2"] = output_df["verdict_2"].map(
        lambda verdict_id: feature_vector_as_json(feature_map, verdict_id)
    )

    ensure_directory(output_path)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    missing_v1 = sorted(
        {v for v, vec in zip(output_df["verdict_1"], output_df["feature_vector_1"]) if vec == "{}"}
    )
    missing_v2 = sorted(
        {v for v, vec in zip(output_df["verdict_2"], output_df["feature_vector_2"]) if vec == "{}"}
    )

    if missing_v1 or missing_v2:
        print("Warning: missing feature vectors for some verdicts.")
        if missing_v1:
            print(f"  verdict_1 without features: {missing_v1}")
        if missing_v2:
            print(f"  verdict_2 without features: {missing_v2}")

    print(f"Saved feature vector dataset to {output_path}")

    return output_df


def resolve_fact_paths(
    target_path: Path,
    facts_paths: Iterable[Path],
) -> List[Path]:
    """Resolve indictment fact CSV paths relative to the target file location."""
    base_dir = target_path.parent
    return [
        path if path.is_absolute() else (base_dir / path).resolve()
        for path in facts_paths
    ]


def load_indictment_facts(facts_paths: Iterable[Path]) -> pd.Series:
    """Load GPT indictment facts from the provided CSV files."""
    frames: List[pd.DataFrame] = []
    missing_files: List[Path] = []

    for path in facts_paths:
        if not path.exists():
            missing_files.append(path)
            continue

        df = pd.read_csv(path, usecols=["verdict", "extracted_gpt_facts"])
        df = df.dropna(subset=["verdict", "extracted_gpt_facts"])
        df["verdict"] = df["verdict"].astype(str).str.strip()
        df["extracted_gpt_facts"] = df["extracted_gpt_facts"].astype(str).str.strip()
        df = df[df["verdict"] != ""]
        frames.append(df)

    if missing_files:
        print("Warning: some indictment facts files were not found.")
        for missing in missing_files:
            print(f"  missing facts file -> {missing}")

    if not frames:
        raise FileNotFoundError("No indictment facts files could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="verdict", keep="first")
    return combined.set_index("verdict")["extracted_gpt_facts"]


def build_indictment_facts_dataset(
    base_df: pd.DataFrame,
    facts_series: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    """Create a dataset that pairs verdicts with their indictment facts."""
    output_columns = [
        "verdict_1",
        "verdict_2",
        "similarity_scale",
        "similarity_binary_0",
        "similarity_binary_1",
    ]

    output_df = base_df[output_columns].copy()
    output_df["indicment_facts_1"] = output_df["verdict_1"].map(facts_series)
    output_df["indicment_facts_2"] = output_df["verdict_2"].map(facts_series)

    missing_v1 = output_df["indicment_facts_1"].isna().sum()
    missing_v2 = output_df["indicment_facts_2"].isna().sum()

    output_df["indicment_facts_1"] = output_df["indicment_facts_1"].fillna("")
    output_df["indicment_facts_2"] = output_df["indicment_facts_2"].fillna("")

    ensure_directory(output_path)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved indictment facts dataset to {output_path}")
    print(f"Missing indictment facts -> verdict_1: {missing_v1}, verdict_2: {missing_v2}")

    return output_df


def build_similarity_database(
    target_path: Path,
    features_path: Optional[Path],
    output_path: Path,
    *,
    facts_paths: Optional[Iterable[Path]] = None,
    facts_output_path: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Create the enriched similarity datasets (feature vectors and optional indictment facts)."""
    base_df = prepare_similarity_frame(target_path)

    datasets: Dict[str, pd.DataFrame] = {}

    if features_path:
        resolved_features = features_path if features_path.is_absolute() else (target_path.parent / features_path)
        resolved_features = resolved_features.resolve()
        if resolved_features.exists():
            feature_map = load_feature_map(resolved_features)
            feature_output_path = output_path if output_path.is_absolute() else (target_path.parent / output_path)
            feature_output_path = feature_output_path.resolve()
            datasets["features"] = build_feature_vector_dataset(
                base_df,
                feature_map,
                feature_output_path,
            )
        else:
            print(f"Warning: features file not found -> {resolved_features}. Skipping feature vector dataset.")

    if facts_paths:
        resolved_facts_paths = resolve_fact_paths(target_path, facts_paths)
        try:
            facts_series = load_indictment_facts(resolved_facts_paths)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}. Skipping indictment facts dataset.")
        else:
            output_path_candidate = (
                facts_output_path if facts_output_path is not None else target_path.with_name("similarity_database_with_indicment_facts.csv")
            )
            if not output_path_candidate.is_absolute():
                output_path_candidate = (target_path.parent / output_path_candidate).resolve()
            datasets["facts"] = build_indictment_facts_dataset(
                base_df,
                facts_series,
                output_path_candidate,
            )
    elif facts_output_path:
        print("Warning: --facts-output was provided without --facts; skipping indictment facts dataset.")

    if "features" not in datasets and "facts" not in datasets:
        raise RuntimeError(
            "No datasets were generated. Provide a valid features CSV, indictment facts CSVs, or both."
        )

    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build similarity datasets that augment verdict pairs with indictment facts "
            "and/or structured feature vectors."
        )
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(
            "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/weapon/target.csv"
        ),
        help="Path to the target CSV with verdict pairs and similarity scores.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "/Users/liorb/Library/CloudStorage/GoogleDrive-liorkob@post.bgu.ac.il/.shortcut-targets-by-id/1f5AVMhCLkfM_ZoGYf_oDNiTxa2jVgL2B/חיזוי מתחמי ענישה/lior.maxim/weapon/manual_feature_schema_extraction.csv"
        ),
        help=(
            "Path to the manual feature schema extraction CSV. "
            "If the file is missing, the feature vector dataset will be skipped."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/weapon/similarity_database_fe.csv"
        ),
        help="Path where the feature vector dataset will be written.",
    )
    parser.add_argument(
        "--facts",
        type=Path,
        nargs="+",
        help=(
            "One or more CSV files with GPT indictment facts to merge into the output. "
            "If omitted, the indictment facts dataset is not generated."
        ),
    )
    parser.add_argument(
        "--facts-output",
        type=Path,
        help=(
            "Output path for the indictment facts dataset. "
            "Defaults to <target_dir>/similarity_database_with_indicment_facts.csv when --facts is supplied."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_similarity_database(
        args.target.resolve(),
        args.features,
        args.output,
        facts_paths=args.facts,
        facts_output_path=args.facts_output,
    )


if __name__ == "__main__":
    main()

