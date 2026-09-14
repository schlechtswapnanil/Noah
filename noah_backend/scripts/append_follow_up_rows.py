"""Append the follow-up rewrite rows to the existing corpus without rebuilding it.

``build_dataset.py`` reshuffles the whole corpus on every run, which turns a
few dozen new rows into a 10,000-line diff and a model trained on a different
sample.  This appends only the rows from ``follow_up_rewrite_rows`` that are
not already present, as raw CSV lines, so every existing line stays byte for
byte as it was and the diff is purely additive.

    python -m scripts.append_follow_up_rows
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_dataset import BASE_DIR, COLUMNS, TARGET, RouteFactory, follow_up_rewrite_rows


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def main() -> int:
    corpus = pd.read_csv(TARGET)
    corpus["planner_actions"] = corpus["planner_actions"].fillna("NULL")
    rows = follow_up_rewrite_rows(RouteFactory(corpus), set(corpus.instruction))
    if not rows:
        print("nothing to add")
        return 0
    with open(TARGET, "rb") as fh:
        fh.seek(-1, 2)
        ends_with_newline = fh.read(1) == b"\n"
    with open(TARGET, "a", newline="", encoding="utf-8") as fh:
        if not ends_with_newline:
            fh.write("\n")
        writer = csv.writer(fh, lineterminator="\n")
        for row in rows:
            writer.writerow([_cell(row[column]) for column in COLUMNS])
    print(f"appended {len(rows)} rows to {TARGET.relative_to(BASE_DIR)} "
          f"({len(corpus) + len(rows)} total)")
    return len(rows)


if __name__ == "__main__":
    main()
