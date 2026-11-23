"""Shared helpers: config, case lists, safe loading of predictions and labels."""
from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "competition.json"
PUBLIC_DIR = ROOT / "data" / "public"
HIDDEN_LABELS_ENC = ROOT / "data" / "hidden" / "test_labels.npz.enc"

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,39}$")
ALLOWED_AUTHOR_TYPES = {"human", "llm", "hybrid"}


class SubmissionError(Exception):
    """A problem with the participant's submission (safe to show in a PR comment)."""


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def foreground_labels(cfg: dict) -> dict[int, str]:
    return {int(k): v for k, v in cfg["labels"].items() if int(k) != 0}


def read_case_ids(csv_path: Path) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [row["case_id"] for row in csv.DictReader(f)]


def read_test_cases(csv_path: Path = PUBLIC_DIR / "test_cases.csv") -> dict[str, dict]:
    """case_id -> {"shape": (x, y, z), "spacing": (sx, sy, sz)}"""
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["case_id"]] = {
                "shape": tuple(int(row[k]) for k in ("shape_x", "shape_y", "shape_z")),
                "spacing": tuple(float(row[k]) for k in ("spacing_x", "spacing_y", "spacing_z")),
            }
    return out


def load_npz_masks(
    path: Path,
    expected: dict[str, tuple[int, ...]],
    allowed_values: set[int],
    max_uncompressed_bytes: int,
) -> dict[str, np.ndarray]:
    """Load an .npz of label maps defensively (no pickle, size-capped, shape/value checked)."""
    try:
        with zipfile.ZipFile(path) as zf:
            total = sum(i.file_size for i in zf.infolist())
    except zipfile.BadZipFile:
        raise SubmissionError("Predictions are not a valid .npz archive (np.savez_compressed).")
    if total > max_uncompressed_bytes:
        raise SubmissionError(f"Uncompressed predictions too large ({total} bytes).")

    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected))
        if missing:
            raise SubmissionError(f"Missing {len(missing)} case(s), e.g. {missing[:5]}")
        if extra:
            raise SubmissionError(f"Unknown case id(s) in predictions, e.g. {extra[:5]}")

        masks = {}
        for cid, shape in expected.items():
            try:
                arr = z[cid]
            except ValueError:
                raise SubmissionError(f"{cid}: object/pickled arrays are not allowed; save plain uint8 arrays")
            if not np.issubdtype(arr.dtype, np.integer) and arr.dtype != np.bool_:
                raise SubmissionError(f"{cid}: dtype must be integer (uint8 recommended), got {arr.dtype}")
            if tuple(arr.shape) != tuple(shape):
                raise SubmissionError(f"{cid}: shape {tuple(arr.shape)} != expected {tuple(shape)}")
            vals = set(np.unique(arr).tolist())
            if not vals <= allowed_values:
                raise SubmissionError(f"{cid}: invalid label values {sorted(vals - allowed_values)}")
            masks[cid] = arr.astype(np.uint8)
    return masks
