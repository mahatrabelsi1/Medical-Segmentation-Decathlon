"""Validate a submission's metadata.json.

    python scoring/validate_metadata.py submissions/inbox/<team>/<run_id>/metadata.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ALLOWED_AUTHOR_TYPES, NAME_RE, SubmissionError  # noqa: E402

REQUIRED = ("team", "run_id", "author_type", "model")
MAX_LEN = {"model": 120, "notes": 500}


def validate_metadata(path: Path, expect_team: str | None = None, expect_run: str | None = None) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SubmissionError("metadata.json is not valid UTF-8 JSON.")
    if not isinstance(data, dict):
        raise SubmissionError("metadata.json must be a JSON object.")

    missing = [k for k in REQUIRED if not isinstance(data.get(k), str) or not data[k].strip()]
    if missing:
        raise SubmissionError(f"metadata.json missing/empty string field(s): {', '.join(missing)}")

    meta = {k: str(data.get(k) or "").strip().replace("\r", " ").replace("\n", " ") for k in (*REQUIRED, "notes")}
    meta["author_type"] = meta["author_type"].lower()

    for k in ("team", "run_id"):
        if not NAME_RE.match(meta[k]):
            raise SubmissionError(f"'{k}' must be 2-40 chars of letters, digits, '_', '-', '.' (got '{meta[k]}')")
    if meta["author_type"] not in ALLOWED_AUTHOR_TYPES:
        raise SubmissionError(f"author_type must be one of {sorted(ALLOWED_AUTHOR_TYPES)}")
    for k, n in MAX_LEN.items():
        if len(meta[k]) > n:
            raise SubmissionError(f"'{k}' is longer than {n} characters")
    if expect_team is not None and meta["team"] != expect_team:
        raise SubmissionError(f"metadata team '{meta['team']}' must match folder name '{expect_team}'")
    if expect_run is not None and meta["run_id"] != expect_run:
        raise SubmissionError(f"metadata run_id '{meta['run_id']}' must match folder name '{expect_run}'")
    return meta


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    p = Path(sys.argv[1])
    try:
        validate_metadata(p, p.parent.parent.name, p.parent.name)
    except SubmissionError as e:
        raise SystemExit(f"Invalid: {e}")
    print("metadata.json is valid")
