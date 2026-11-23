"""Noise models used to turn MSD Task04 into a robustness benchmark.

Image noise (train, val and test):
  * multiplicative MRI bias field: smooth low-order polynomial, exp-scaled
  * Rician noise: the magnitude of complex Gaussian noise, as in real MR magnitude images,
    with a per-case sigma drawn from a range, so noise levels vary between cases

Label noise (TRAIN only; val and test labels stay clean):
  * a fraction of training cases gets each foreground class randomly eroded or dilated by 1 voxel,
    which simulates boundary disagreement between annotators
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

_STRUCT = generate_binary_structure(3, 1)


def bias_field(shape, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Smooth multiplicative field with values in [exp(-strength), exp(strength)]."""
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    f = np.zeros(shape, dtype=np.float64)
    for i in range(3):
        for j in range(3 - i):
            for k in range(3 - i - j):
                if i + j + k:
                    f += rng.normal() * x**i * y**j * z**k
    f *= strength / (np.abs(f).max() + 1e-8)
    return np.exp(f)


def noisy_image(img: np.ndarray, rng: np.random.Generator, sigma_range, bias_strength: float):
    """Return (noisy uint8 image, sigma_fraction used)."""
    x = img.astype(np.float64) * bias_field(img.shape, rng, bias_strength)
    frac = float(rng.uniform(*sigma_range))
    sigma = frac * np.percentile(img, 99)
    x = np.sqrt((x + rng.normal(0, sigma, x.shape)) ** 2 + rng.normal(0, sigma, x.shape) ** 2)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8), frac


def noisy_label(lab: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Randomly erode or dilate each foreground class by one voxel."""
    out, ops = lab.copy(), []
    for c in sorted(int(v) for v in np.unique(lab) if v != 0):
        m = lab == c
        if rng.random() < 0.5:
            out[m & ~binary_erosion(m, _STRUCT)] = 0
            ops.append(f"{c}:erode")
        else:
            out[binary_dilation(m, _STRUCT) & (out == 0)] = c
            ops.append(f"{c}:dilate")
    return out, " ".join(ops)
