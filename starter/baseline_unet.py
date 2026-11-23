"""Minimal 3D U-Net baseline (PyTorch). Runs on CPU in a few minutes.

    pip install torch
    python starter/baseline_unet.py --epochs 40 --out runs/baseline

Writes:
    runs/baseline/val/<case_id>.nii.gz   -> score with scoring/score_local.py
    runs/baseline/test/<case_id>.nii.gz  -> package with starter/make_submission.py
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scoring"))
from common import PUBLIC_DIR, foreground_labels, load_config, read_case_ids, read_test_cases  # noqa: E402
from metrics import dice  # noqa: E402


# ---------------------------------------------------------------- data
def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32), img.affine


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def pad_to(x: np.ndarray, shape) -> np.ndarray:
    return np.pad(x, [(0, s - d) for s, d in zip(shape, x.shape)])


# ---------------------------------------------------------------- model
def block(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout), nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1), nn.InstanceNorm3d(cout), nn.LeakyReLU(0.01, inplace=True))


class UNet3D(nn.Module):
    def __init__(self, n_classes: int, base: int = 16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3, self.b = block(1, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])
        self.u3, self.d3 = nn.ConvTranspose3d(c[3], c[2], 2, 2), block(c[3], c[2])
        self.u2, self.d2 = nn.ConvTranspose3d(c[2], c[1], 2, 2), block(c[2], c[1])
        self.u1, self.d1 = nn.ConvTranspose3d(c[1], c[0], 2, 2), block(c[1], c[0])
        self.head = nn.Conv3d(c[0], n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool3d(e1, 2))
        e3 = self.e3(F.max_pool3d(e2, 2))
        b = self.b(F.max_pool3d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


def soft_dice_loss(logits, target, n_classes):
    p = logits.softmax(1)
    t = F.one_hot(target, n_classes).permute(0, 4, 1, 2, 3).float()
    inter = (p * t).sum((2, 3, 4))
    den = p.sum((2, 3, 4)) + t.sum((2, 3, 4))
    return 1 - ((2 * inter + 1e-5) / (den + 1e-5))[:, 1:].mean()


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "baseline")
    a = ap.parse_args()

    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    n_classes = len(foreground_labels(cfg)) + 1

    train_ids = read_case_ids(PUBLIC_DIR / "train.csv")
    val_ids = read_case_ids(PUBLIC_DIR / "val.csv")
    test_cases = read_test_cases()

    def read_pair(cid):
        x, aff = load(PUBLIC_DIR / "imagesTr" / f"{cid}.nii.gz")
        y, _ = load(PUBLIC_DIR / "labelsTr" / f"{cid}.nii.gz")
        return normalize(x), y.astype(np.int64), aff

    train = {c: read_pair(c) for c in train_ids}
    val = {c: read_pair(c) for c in val_ids}
    test = {c: (lambda x, aff: (normalize(x), aff))(*load(PUBLIC_DIR / "imagesTs" / f"{c}.nii.gz")) for c in test_cases}

    shapes = [v[0].shape for v in (*train.values(), *val.values(), *test.values())]
    canvas = tuple(int(np.ceil(max(s[i] for s in shapes) / 8) * 8) for i in range(3))
    print(f"device={dev} train={len(train)} val={len(val)} test={len(test)} canvas={canvas}")

    X = torch.from_numpy(np.stack([pad_to(v[0], canvas) for v in train.values()]))[:, None]
    Y = torch.from_numpy(np.stack([pad_to(v[1], canvas) for v in train.values()]))

    model = UNet3D(n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    @torch.no_grad()
    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        t = torch.from_numpy(pad_to(x, canvas))[None, None].to(dev)
        logits = model(t) + model(t.flip(2)).flip(2)  # light test-time augmentation
        return logits.argmax(1)[0].cpu().numpy()[tuple(slice(0, s) for s in x.shape)].astype(np.uint8)

    def val_dice() -> float:
        scores = [np.mean([dice(predict(x) == k, y == k) for k in foreground_labels(cfg)]) for x, y, _ in val.values()]
        return float(np.mean(scores))

    best, best_state = -1.0, None
    for ep in range(1, a.epochs + 1):
        model.train()
        perm, total = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), a.batch_size):
            idx = perm[i:i + a.batch_size]
            xb, yb = X[idx].to(dev), Y[idx].to(dev)
            if random.random() < 0.5:
                xb, yb = xb.flip(2), yb.flip(1)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb) + soft_dice_loss(logits, yb, n_classes)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if ep % 5 == 0 or ep == a.epochs:
            d = val_dice()
            if d > best:
                best, best_state = d, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"epoch {ep:3d}  loss {total / len(X):.4f}  val dice {d:.4f}  (best {best:.4f})")
        else:
            print(f"epoch {ep:3d}  loss {total / len(X):.4f}")

    model.load_state_dict(best_state)
    for split, data in (("val", {c: (v[0], v[2]) for c, v in val.items()}), ("test", test)):
        out = a.out / split
        out.mkdir(parents=True, exist_ok=True)
        for cid, (x, aff) in data.items():
            nib.save(nib.Nifti1Image(predict(x), aff), str(out / f"{cid}.nii.gz"))
    print(f"Best val Dice {best:.4f}. Predictions written to {a.out}/val and {a.out}/test")


if __name__ == "__main__":
    main()
