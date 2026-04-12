#!/usr/bin/env python3
"""
Convert downloaded .docx files to verdict CSV format.
Extracts text from each paragraph with section headers.
Output: one CSV per verdict with columns (verdict, text, part).
"""

import os
import re
import sys
import pandas as pd
from pathlib import Path
from docx import Document
from docx.text.paragraph import Paragraph
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table

DOWNLOADED_DOCX_DIR = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/innovation_submission/downloaded_verdicts_docx")
OUTPUT_CSV_DIR = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/innovation_submission/downloaded_verdict_csv")


def iterate_block_items(parent):
    if hasattr(parent, "element") and hasattr(parent.element, "body"):
        parent_element = parent.element.body
    elif hasattr(parent, "_tc"):
        parent_element = parent._tc
    else:
        return
    for child in parent_element.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            table = Table(child, parent)
            for row in table.rows:
                for cell in row.cells:
                    yield from iterate_block_items(cell)


def is_run_bold(run):
    if run.bold is not None:
        return run.bold
    if run.font and run.font.bold is not None:
        return run.font.bold
    if run.font and hasattr(run.font, 'cs_bold') and run.font.cs_bold is not None:
        return run.font.cs_bold
    return False


def is_block_styled(block):
    if not hasattr(block, "runs") or not block.runs:
        return False
    text = " ".join(r.text.strip() for r in block.runs if r.text.strip()).strip()
    if not text:
        return False
    if len(text.split()) < 4:
        return True
    meaningful_runs = [r for r in block.runs if r.text.strip() and any(c.isalnum() for c in r.text)]
    if not meaningful_runs:
        return False
    bold_count = sum(1 for r in meaningful_runs if is_run_bold(r))
    if bold_count >= len(meaningful_runs) * 0.7:
        return True
    if block.style and block.style.name in ["כותרת", "Heading", "Title", "Heading 1", "Heading 2", "Heading 3"]:
        return True
    return False


def docx_to_csv(doc_path):
    verdict_id = os.path.splitext(os.path.basename(doc_path))[0]
    doc = Document(doc_path)
    rows = []
    current_part = "nothing"

    for block in iterate_block_items(doc):
        text = block.text.strip() if hasattr(block, 'text') else ""
        if not text:
            continue

        if is_block_styled(block) and len(text.split()) < 10:
            title = re.sub(r'^(?:\d+[.)]|[\u0590-\u05FF][.)])', '', text).strip()
            if title:
                current_part = title
            continue

        rows.append({
            'verdict': verdict_id,
            'text': text,
            'part': current_part,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.drop_duplicates(subset=['text', 'part'], inplace=True)
        df = df[df['text'].str.strip().astype(bool)]
    return df


def main():
    OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

    docx_files = [f for f in os.listdir(DOWNLOADED_DOCX_DIR) if f.endswith('.docx')]
    print(f"Found {len(docx_files)} docx files in {DOWNLOADED_DOCX_DIR}")

    converted = 0
    skipped = 0
    errors = 0

    for filename in sorted(docx_files):
        stem = filename.rsplit('.', 1)[0]
        output_path = OUTPUT_CSV_DIR / f"{stem}.csv"

        if output_path.exists() and output_path.stat().st_size > 0:
            skipped += 1
            continue

        try:
            df = docx_to_csv(str(DOWNLOADED_DOCX_DIR / filename))
            if not df.empty:
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                converted += 1
            else:
                errors += 1
                print(f"  Empty: {filename}")
        except Exception as e:
            errors += 1
            print(f"  Error: {filename}: {e}")

    print(f"\nConverted: {converted}, Skipped: {skipped}, Errors: {errors}")
    print(f"Output: {OUTPUT_CSV_DIR}")


if __name__ == "__main__":
    main()
