"""Segmentation metrics: Dice and Normalised Surface Distance (NSD), as used by the MSD."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

_STRUCT = generate_binary_structure(3, 1)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice of two boolean masks. Both empty -> 1.0."""
    ps, gs = int(pred.sum()), int(gt.sum())
    if ps + gs == 0:
        return 1.0
    return 2.0 * int(np.logical_and(pred, gt).sum()) / (ps + gs)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, structure=_STRUCT, border_value=0)


def nsd(pred: np.ndarray, gt: np.ndarray, spacing, tolerance_mm: float) -> float:
    """Normalised Surface Distance (voxel-surface approximation).

    Fraction of boundary voxels of each mask that lie within `tolerance_mm`
    of the other mask's boundary. Both empty -> 1.0, one empty -> 0.0.
    """
    sp, sg = _surface(pred), _surface(gt)
    n_p, n_g = int(sp.sum()), int(sg.sum())
    if n_p + n_g == 0:
        return 1.0
    if n_p == 0 or n_g == 0:
        return 0.0
    dist_to_g = distance_transform_edt(~sg, sampling=spacing)
    dist_to_p = distance_transform_edt(~sp, sampling=spacing)
    ok = int((dist_to_g[sp] <= tolerance_mm).sum()) + int((dist_to_p[sg] <= tolerance_mm).sum())
    return ok / (n_p + n_g)


def score_cases(
    preds: dict[str, np.ndarray],
    gts: dict[str, np.ndarray],
    spacings: dict[str, tuple],
    labels: dict[int, str],
    tolerance_mm: float,
) -> dict[str, float]:
    """Mean over cases of per-class Dice/NSD, plus class-averaged summaries."""
    per = {f"{m}_{name}": [] for m in ("dice", "nsd") for name in labels.values()}
    for cid, gt in gts.items():
        pred = preds[cid]
        for lab, name in labels.items():
            p, g = pred == lab, gt == lab
            per[f"dice_{name}"].append(dice(p, g))
            per[f"nsd_{name}"].append(nsd(p, g, spacings[cid], tolerance_mm))

    out = {k: float(np.mean(v)) for k, v in per.items()}
    out["dice_mean"] = float(np.mean([out[f"dice_{n}"] for n in labels.values()]))
    out["nsd_mean"] = float(np.mean([out[f"nsd_{n}"] for n in labels.values()]))
    out["n_cases"] = len(gts)
    return out
