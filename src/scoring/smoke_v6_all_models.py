#!/usr/bin/env python3
"""
Smoke test: two v6-style API calls per model in MAIN_MODELS_DEFAULT.
Exits 0 only if every model returns two parseable scores in 0–100.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from structured_llm_comparison_experiment import (
    SYSTEM_PROMPT_V6_SCORE_RAW,
    USER_TEMPLATE_SCORE_RAW,
    parse_score,
)
from v6_score_multimodel_experiment import (
    MAIN_MODELS_DEFAULT,
    call_model_backend,
    validate_score,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=BASE.parent / "drugs" / "similarity_database_hybrid_full_gpt.csv",
        help="Use first 2 rows for prompts",
    )
    ap.add_argument("--sleep", type=float, default=0.25, help="Pause after each API call")
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of backends (default: MAIN_MODELS_DEFAULT from v6)",
    )
    args = ap.parse_args()

    models = args.models if args.models else list(MAIN_MODELS_DEFAULT)
    unknown = [m for m in models if m not in MAIN_MODELS_DEFAULT]
    if unknown:
        print(f"Unknown model(s): {unknown}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.csv).iloc[:2]
    if len(df) < 2:
        print("Need at least 2 rows in CSV", file=sys.stderr)
        sys.exit(2)

    user_prompts: list[str] = []
    for _, row in df.iterrows():
        user_prompts.append(
            USER_TEMPLATE_SCORE_RAW.format(
                fv1=row["feature_vector_1"],
                fv2=row["feature_vector_2"],
            )
        )

    system = SYSTEM_PROMPT_V6_SCORE_RAW
    failed: list[str] = []

    print(f"Models to test: {len(models)} (expect 2 OK calls each)\n")

    for model in models:
        ok_count = 0
        errs: list[str] = []
        for i, user in enumerate(user_prompts):
            raw = call_model_backend(model, system, user)
            if raw is None:
                errs.append(f"c{i + 1}=None")
            else:
                parsed = parse_score(raw)
                if validate_score(parsed):
                    ok_count += 1
                else:
                    snippet = raw.replace("\n", " ")[:120]
                    errs.append(f"c{i + 1}_bad={parsed!r} text={snippet!r}")
            if args.sleep > 0:
                time.sleep(args.sleep)

        line = f"{model:22} {ok_count}/2"
        if ok_count >= 2:
            print(f"{line}  OK")
        else:
            print(f"{line}  FAIL  {errs}")
            failed.append(model)
        if args.sleep > 0:
            time.sleep(0.2)

    print()
    if failed:
        print("FAILED models:", ", ".join(failed))
        sys.exit(1)
    print("All models: 2/2 valid SIMILARITY_SCORE parses.")
    sys.exit(0)


if __name__ == "__main__":
    main()
