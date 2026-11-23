"""Re-score every archived submission and regenerate the leaderboard (organizers / CI).

    python scoring/rebuild_leaderboard.py --private-key data/private/submission_private_key.pem

Useful after changing a metric. Reads submissions/scored/<team>/<run_id>/ and
keeps each run's original submitted_at from its receipt.json.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "encryption"))
from common import HIDDEN_LABELS_ENC, ROOT, foreground_labels, load_config, load_npz_masks, read_test_cases  # noqa
from crypto import decrypt_bytes  # noqa: E402
from leaderboard import write_all  # noqa: E402
from metrics import score_cases  # noqa: E402
from validate_metadata import validate_metadata  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private-key", type=Path, required=True)
    a = ap.parse_args()
    key = a.private_key.read_bytes()

    cfg = load_config()
    cases = read_test_cases()
    labels = foreground_labels(cfg)
    spacings = {c: v["spacing"] for c, v in cases.items()}
    with np.load(io.BytesIO(decrypt_bytes(HIDDEN_LABELS_ENC.read_bytes(), key)), allow_pickle=False) as z:
        gts = {c: z[c] for c in cases}

    rows = []
    for run_dir in sorted((ROOT / "submissions" / "scored").glob("*/*/")):
        try:
            meta = validate_metadata(run_dir / "metadata.json")
            receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "pred.npz"
                p.write_bytes(decrypt_bytes((run_dir / "predictions.npz.enc").read_bytes(), key))
                preds = load_npz_masks(p, {c: v["shape"] for c, v in cases.items()},
                                       {int(k) for k in cfg["labels"]}, cfg["max_uncompressed_bytes"])
            scores = score_cases(preds, gts, spacings, labels, cfg["metric"]["nsd_tolerance_mm"])
        except Exception as e:  # keep going; report broken archives
            print(f"SKIP {run_dir}: {e}")
            continue
        rows.append({**meta, **{k: f"{v:.6f}" for k, v in scores.items() if k != "n_cases"},
                     "submitted_at": receipt["submitted_at"]})
        print(f"{meta['team']}/{meta['run_id']}: dice={scores['dice_mean']:.4f} nsd={scores['nsd_mean']:.4f}")

    write_all(rows)
    print(f"Leaderboard rebuilt with {len(rows)} run(s).")


if __name__ == "__main__":
    main()
