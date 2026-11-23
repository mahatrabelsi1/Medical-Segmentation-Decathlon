"""Score your predictions on the public VALIDATION split, exactly like the official scorer.

    python scoring/score_local.py --pred-dir runs/my_model/val

`--pred-dir` holds one NIfTI label map per case named <case_id>.nii.gz
(e.g. hippocampus_003.nii.gz). Alternatively pass `--npz preds.npz` with
one array per case id.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import PUBLIC_DIR, foreground_labels, load_config, read_case_ids  # noqa: E402
from metrics import score_cases  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred-dir", type=Path)
    g.add_argument("--npz", type=Path)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    a = ap.parse_args()

    cfg = load_config()
    ids = read_case_ids(PUBLIC_DIR / f"{a.split}.csv")
    gts, spacings, preds = {}, {}, {}
    npz = np.load(a.npz, allow_pickle=False) if a.npz else None
    for cid in ids:
        lab = nib.load(str(PUBLIC_DIR / "labelsTr" / f"{cid}.nii.gz"))
        gts[cid] = np.asarray(lab.dataobj).astype(np.uint8)
        spacings[cid] = tuple(float(z) for z in lab.header.get_zooms()[:3])
        if npz is not None:
            if cid not in npz.files:
                raise SystemExit(f"missing case {cid} in {a.npz}")
            preds[cid] = npz[cid]
        else:
            f = a.pred_dir / f"{cid}.nii.gz"
            if not f.exists():
                raise SystemExit(f"missing prediction {f}")
            preds[cid] = np.asarray(nib.load(str(f)).dataobj)
        if preds[cid].shape != gts[cid].shape:
            raise SystemExit(f"{cid}: shape {preds[cid].shape} != {gts[cid].shape}")

    s = score_cases(preds, gts, spacings, foreground_labels(cfg), cfg["metric"]["nsd_tolerance_mm"])
    print(f"{a.split} split, {s['n_cases']} cases")
    print(f"  Dice (mean) : {s['dice_mean']:.4f}   <- leaderboard metric")
    print(f"  NSD  (mean) : {s['nsd_mean']:.4f}   (tolerance {cfg['metric']['nsd_tolerance_mm']} mm)")
    for name in foreground_labels(cfg).values():
        print(f"  {name:<12}: Dice {s[f'dice_{name}']:.4f}  NSD {s[f'nsd_{name}']:.4f}")


if __name__ == "__main__":
    main()
