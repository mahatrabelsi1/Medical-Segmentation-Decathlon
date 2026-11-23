"""Score one encrypted submission against the hidden test labels (run by GitHub Actions).

    python scoring/evaluate_submission.py \
        --run-dir <checkout>/submissions/inbox/<team>/<run_id> \
        --private-key data/private/submission_private_key.pem \
        --result result.json

On success: appends to the leaderboard, archives the encrypted files under
submissions/scored/<team>/<run_id>/ and writes result.json (status=ok).
On a submission problem: writes result.json (status=error, message) and exits 1.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "encryption"))
from common import (HIDDEN_LABELS_ENC, ROOT, SubmissionError, foreground_labels,  # noqa: E402
                    load_config, load_npz_masks, read_test_cases)
from crypto import decrypt_bytes, normalize_pem  # noqa: E402
from leaderboard import best_per_team, read_rows, write_all  # noqa: E402
from metrics import score_cases  # noqa: E402
from validate_metadata import validate_metadata  # noqa: E402

PRED_NAME, META_NAME = "predictions.npz.enc", "metadata.json"


def _check_file(p: Path, max_bytes: int) -> None:
    if p.is_symlink() or not p.is_file():
        raise SubmissionError(f"{p.name} must be a regular file")
    if p.stat().st_size > max_bytes:
        raise SubmissionError(f"{p.name} is larger than {max_bytes} bytes")


def check_server_key(private_key: bytes) -> None:
    """Fail loudly (as an organizer problem) if the secret is not the pair of the committed public key."""
    from cryptography.hazmat.primitives import serialization
    try:
        pub = serialization.load_pem_private_key(normalize_pem(private_key), password=None).public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    except Exception:
        raise RuntimeError("SUBMISSION_PRIVATE_KEY secret is not a valid unencrypted PEM private key")
    if pub.strip() != (ROOT / "encryption" / "public_key.pem").read_bytes().strip():
        raise RuntimeError("SUBMISSION_PRIVATE_KEY secret does not match encryption/public_key.pem")


def evaluate(run_dir: Path, private_key: bytes, archive: bool = True) -> dict:
    check_server_key(private_key)
    cfg = load_config()
    if run_dir.is_symlink() or run_dir.parent.is_symlink():
        raise SubmissionError("Submission folders must not be symlinks")
    pred_enc, meta_path = run_dir / PRED_NAME, run_dir / META_NAME
    _check_file(pred_enc, cfg["max_submission_bytes"])
    _check_file(meta_path, 64 * 1024)

    meta = validate_metadata(meta_path, expect_team=run_dir.parent.name, expect_run=run_dir.name)

    rows = read_rows()
    team_rows = [r for r in rows if r["team"].lower() == meta["team"].lower()]
    if any(r["run_id"].lower() == meta["run_id"].lower() for r in team_rows):
        raise SubmissionError(f"Run '{meta['run_id']}' was already scored for team '{meta['team']}'. Use a new run_id.")
    if len(team_rows) >= cfg["max_submissions_per_team"]:
        raise SubmissionError(f"Team '{meta['team']}' has used all {cfg['max_submissions_per_team']} submissions.")

    try:
        pred_bytes = decrypt_bytes(pred_enc.read_bytes(), private_key)
    except Exception:
        raise SubmissionError("Could not decrypt predictions. Encrypt with encryption/public_key.pem "
                              "(see starter/make_submission.py).")

    cases = read_test_cases()
    labels = foreground_labels(cfg)
    allowed = {int(k) for k in cfg["labels"]}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "pred.npz"
        p.write_bytes(pred_bytes)
        preds = load_npz_masks(p, {c: v["shape"] for c, v in cases.items()}, allowed, cfg["max_uncompressed_bytes"])

    with np.load(io.BytesIO(decrypt_bytes(HIDDEN_LABELS_ENC.read_bytes(), private_key)), allow_pickle=False) as z:
        gts = {c: z[c] for c in cases}

    scores = score_cases(preds, gts, {c: v["spacing"] for c, v in cases.items()},
                         labels, cfg["metric"]["nsd_tolerance_mm"])

    submitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    row = {**meta, **{k: f"{v:.6f}" for k, v in scores.items() if k != "n_cases"}, "submitted_at": submitted_at}
    rows.append(row)
    write_all(rows)

    if archive:
        dst = ROOT / "submissions" / "scored" / meta["team"] / meta["run_id"]
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pred_enc, dst / PRED_NAME)
        shutil.copyfile(meta_path, dst / META_NAME)
        (dst / "receipt.json").write_text(json.dumps({"submitted_at": submitted_at, **scores}, indent=2),
                                          encoding="utf-8")

    rank = next(r["rank"] for r in best_per_team(rows) if r["team"].lower() == meta["team"].lower())
    used = len(team_rows) + 1
    return {"status": "ok", "team": meta["team"], "run_id": meta["run_id"], "scores": scores,
            "team_rank": rank, "n_teams": len(best_per_team(rows)),
            "submissions_used": used, "submissions_max": cfg["max_submissions_per_team"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--private-key", type=Path, required=True)
    ap.add_argument("--result", type=Path, default=Path("result.json"))
    ap.add_argument("--no-archive", action="store_true")
    a = ap.parse_args()

    try:
        result = evaluate(a.run_dir, a.private_key.read_bytes(), archive=not a.no_archive)
        code = 0
    except SubmissionError as e:
        result, code = {"status": "error", "message": str(e)}, 1
    except RuntimeError as e:
        print(f"ORGANIZER CONFIG ERROR: {e}")
        result, code = {"status": "error", "message": "Scoring server misconfigured (organizer key problem), "
                        "not your submission. An organizer will fix it and re-run."}, 1
    except Exception:
        traceback.print_exc()
        result, code = {"status": "error", "message": "Internal scoring error - an organizer will take a look."}, 1

    a.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    sys.exit(code)


if __name__ == "__main__":
    main()
