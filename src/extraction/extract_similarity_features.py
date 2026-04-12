from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

try:
    from openai import OpenAI
    from openai.error import OpenAIError  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - library may not be installed at authoring time
    raise SystemExit(
        "The 'openai' package is required to run this script. Install it via 'pip install openai'."
    ) from exc


SYSTEM_PROMPT = (
    "You are an expert legal analyst. "
    "You read indictment facts from Israeli criminal cases and extract structured features "
    "that help compare cases for sentencing similarity."
)

USER_PROMPT_TEMPLATE = """\
Extract a concise JSON object capturing the key features of this case that influence sentencing.

Guidelines:
- Respond with a single JSON object only (no explanations, no markdown code fences).
- Use snake_case keys.
- Focus on attributes that influence similarity: drug type, quantities, role of the defendant, number of counts,
  aggravating factors (e.g., use of weapons, prior convictions), mitigating factors, plea deals, cooperation, etc.
- Provide scalar values or short strings; use numbers where available.
- If information is missing, omit the key (do not fabricate values).

Case identifier: {verdict_id}
Indictment facts:
{facts}
"""


@dataclass(frozen=True)
class VerdictFacts:
    verdict: str
    indictment_facts: str


def load_unique_verdict_facts(csv_path: Path) -> list[VerdictFacts]:
    df = pd.read_csv(csv_path)

    column_options = {
        "verdict_1": ["verdict_1"],
        "verdict_2": ["verdict_2"],
        "facts_1": [
            "verdict_1_indictment_facts",
            "indicment_facts_1",
            "indictment_facts_1",
        ],
        "facts_2": [
            "verdict_2_indictment_facts",
            "indicment_facts_2",
            "indictment_facts_2",
        ],
    }

    resolved_columns: dict[str, str] = {}
    for key, candidates in column_options.items():
        for column in candidates:
            if column in df.columns:
                resolved_columns[key] = column
                break
        else:
            raise ValueError(
                f"Input CSV {csv_path} is missing required column(s): {candidates}"
            )

    records: list[pd.DataFrame] = []
    for prefix in ("verdict_1", "verdict_2"):
        verdict_col = resolved_columns[prefix]
        facts_key = "facts_1" if prefix == "verdict_1" else "facts_2"
        facts_col = resolved_columns[facts_key]
        subset = (
            df[[verdict_col, facts_col]]
            .rename(columns={verdict_col: "verdict", facts_col: "indictment_facts"})
            .copy()
        )
        records.append(subset)

    combined = pd.concat(records, ignore_index=True)
    combined["verdict"] = combined["verdict"].astype(str).str.strip()
    combined["indictment_facts"] = combined["indictment_facts"].fillna("").astype(str).str.strip()
    combined = combined[combined["verdict"] != ""]

    combined = combined.drop_duplicates(subset="verdict", keep="first")

    verdicts = [
        VerdictFacts(verdict=row["verdict"], indictment_facts=row["indictment_facts"])
        for _, row in combined.iterrows()
    ]
    verdicts.sort(key=lambda item: item.verdict)
    return verdicts


def sanitise_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "").replace("JSON\n", "")
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    # Attempt to locate the first JSON object in the text
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_feature_json(text: str) -> dict[str, object]:
    cleaned = sanitise_json_text(text)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object with key/value pairs")
    return parsed


def build_user_prompt(verdict_facts: VerdictFacts) -> str:
    return USER_PROMPT_TEMPLATE.format(
        verdict_id=verdict_facts.verdict,
        facts=verdict_facts.indictment_facts,
    )


def request_features(
    client: OpenAI,
    verdict_facts: VerdictFacts,
    *,
    model: str,
    max_retries: int,
    retry_delay: float,
) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(verdict_facts)},
                ],
            )
            content = completion.choices[0].message.content or ""
            return parse_feature_json(content)
        except (OpenAIError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_delay * attempt)
    assert last_error is not None
    raise RuntimeError(
        f"Failed to extract features for verdict {verdict_facts.verdict!r} after {max_retries} attempts"
    ) from last_error


def iter_pending_verdicts(
    all_verdicts: Iterable[VerdictFacts],
    processed: set[str],
) -> Iterator[VerdictFacts]:
    for item in all_verdicts:
        if item.verdict not in processed:
            yield item


def load_existing_output(output_path: Path) -> pd.DataFrame | None:
    if not output_path.exists():
        return None
    return pd.read_csv(output_path)


def save_results(results: list[dict[str, object]], output_path: Path) -> None:
    df = pd.DataFrame(results)
    df = df[["verdict", "indictment_facts", "feature_vector"]]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract similarity feature vectors from indictment facts.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/drugs/similarity_database_with_indicment_facts.csv"
        ),
        help="Path to the similarity database CSV containing indictment facts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/drugs/indicment_facts_feature_vectors.csv"
        ),
        help="Destination CSV for verdict, indictment facts, and GPT-generated feature vectors.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name to use for feature extraction.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls to avoid rate limits.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed API calls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output CSV (skips already processed verdicts).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of verdicts to process (for testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI()

    all_verdicts = load_unique_verdict_facts(args.input)

    existing_df = load_existing_output(args.output) if args.resume else None
    processed_verdicts: set[str] = set()
    results: list[dict[str, object]] = []

    if existing_df is not None:
        processed_verdicts = set(existing_df["verdict"].astype(str))
        results.extend(existing_df.to_dict(orient="records"))
        print(f"Resuming with {len(processed_verdicts)} verdicts already processed.")

    pending = list(iter_pending_verdicts(all_verdicts, processed_verdicts))
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Processing {len(pending)} verdicts...")

    for idx, verdict_facts in enumerate(pending, start=1):
        if not verdict_facts.indictment_facts:
            features = {}
        else:
            features = request_features(
                client,
                verdict_facts,
                model=args.model,
                max_retries=args.max_retries,
                retry_delay=args.sleep,
            )
        results.append(
            {
                "verdict": verdict_facts.verdict,
                "indictment_facts": verdict_facts.indictment_facts,
                "feature_vector": json.dumps(features, ensure_ascii=False, sort_keys=True),
            }
        )
        if idx % 5 == 0:
            print(f"Processed {idx}/{len(pending)} verdicts...")
        if args.sleep > 0:
            time.sleep(args.sleep)

    save_results(results, args.output)
    print(f"Saved feature vectors to {args.output}")


if __name__ == "__main__":
    main()

