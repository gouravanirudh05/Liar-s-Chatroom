"""Validate a submission CSV against sample_submission.csv (ids, header, labels). Usage: check_submission.py FILE"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sub = pd.read_csv(sys.argv[1], dtype=str)
ref = pd.read_csv(ROOT / "Data" / "sample_submission.csv", dtype=str)
assert list(sub.columns) == ["id", "label"], f"bad header {list(sub.columns)}"
assert not sub["id"].duplicated().any(), "duplicate ids"
assert set(sub["id"]) == set(ref["id"]), f"id mismatch: missing {len(set(ref.id) - set(sub.id))}, extra {len(set(sub.id) - set(ref.id))}"
assert sub["label"].isin(list("ABCD")).all(), "labels must be A/B/C/D"
print(f"OK: {len(sub)} rows; label counts {sub['label'].value_counts().to_dict()}")
