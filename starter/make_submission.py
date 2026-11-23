"""Package test-set predictions into an encrypted submission.

    python starter/make_submission.py --pred-dir runs/my_model/test \
        --team my_team --run-id unet_v1 --author-type human \
        --model "3D U-Net" --notes "CE + Dice, 200 epochs"

Input: one label map per test case, named <case_id>.nii.gz (test_001.nii.gz, ...),
with the same shape as the matching image in data/public/imagesTs and integer
labels {0: background, 1: anterior, 2: posterior}.

Output (ready to commit and open a PR with):
    submissions/inbox/<team>/<run_id>/predictions.npz.enc
    submissions/inbox/<team>/<run_id>/metadata.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scoring"))
sys.path.insert(0, str(ROOT / "encryption"))
from common import SubmissionError, load_config, load_npz_masks, read_test_cases  # noqa: E402
from crypto import encrypt_bytes  # noqa: E402
from validate_metadata import validate_metadata  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred-dir", type=Path, help="folder of <case_id>.nii.gz label maps")
    g.add_argument("--npz", type=Path, help="or: an .npz with one uint8 array per case id")
    ap.add_argument("--team", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--author-type", required=True, choices=["human", "llm", "hybrid"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    cfg = load_config()
    cases = read_test_cases()

    if a.pred_dir:
        arrays = {}
        for cid in cases:
            f = a.pred_dir / f"{cid}.nii.gz"
            if not f.exists():
                raise SystemExit(f"Missing prediction: {f}")
            arrays[cid] = np.rint(np.asarray(nib.load(str(f)).dataobj)).astype(np.uint8)
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        raw = buf.getvalue()
    else:
        raw = a.npz.read_bytes()

    # Validate exactly like the official scorer before encrypting.
    out_dir = ROOT / "submissions" / "inbox" / a.team / a.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / ".check.npz"
    tmp.write_bytes(raw)
    try:
        load_npz_masks(tmp, {c: v["shape"] for c, v in cases.items()},
                       {int(k) for k in cfg["labels"]}, cfg["max_uncompressed_bytes"])
    except SubmissionError as e:
        raise SystemExit(f"Predictions invalid: {e}")
    finally:
        tmp.unlink(missing_ok=True)

    meta = {"team": a.team, "run_id": a.run_id, "author_type": a.author_type, "model": a.model, "notes": a.notes}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    try:
        validate_metadata(out_dir / "metadata.json", a.team, a.run_id)
    except SubmissionError as e:
        raise SystemExit(f"Metadata invalid: {e}")

    enc = encrypt_bytes(raw, (ROOT / "encryption" / "public_key.pem").read_bytes())
    if len(enc) > cfg["max_submission_bytes"]:
        raise SystemExit(f"Encrypted submission is {len(enc)} bytes (limit {cfg['max_submission_bytes']}).")
    (out_dir / "predictions.npz.enc").write_bytes(enc)

    rel = out_dir.relative_to(ROOT).as_posix()
    print(f"Submission ready in {rel}/ ({len(cases)} cases, {len(enc) / 1024:.0f} KB encrypted)")
    print("Next: commit these 2 files on a branch of your fork and open a Pull Request to main.")


if __name__ == "__main__":
    main()
