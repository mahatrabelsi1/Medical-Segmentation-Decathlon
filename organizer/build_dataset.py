"""Build the competition data from a raw MSD task folder (organizers only).

    python organizer/build_dataset.py --raw-dir data/raw/Task04_Hippocampus

What it does
  1. Splits the labelled MSD training cases (imagesTr/labelsTr) into
     train / val / test with the seed and fractions in competition.json.
     (Official MSD test labels were never released, so the hidden test set
     is carved out of the labelled data.)
  2. Copies train + val images and labels to data/public/imagesTr, labelsTr.
  3. Re-saves test images under anonymous ids (test_001, ...) in a shuffled
     order to data/public/imagesTs, and writes data/public/test_cases.csv.
  4. Packs the test labels into an .npz, encrypts it with the public key and
     writes data/hidden/test_labels.npz.enc (safe to commit: only the
     private key in GitHub Secrets can open it).
  5. Keeps plaintext copies + the id mapping in data/private/ (gitignored).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "encryption"))
sys.path.insert(0, str(ROOT / "scoring"))
from crypto import encrypt_bytes  # noqa: E402
from common import load_config  # noqa: E402
from noise import noisy_image, noisy_label  # noqa: E402


def case_name(rel: str) -> str:
    return Path(rel).name.replace(".nii.gz", "")


def write_ids(path: Path, ids: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id"])
        w.writerows([[i] for i in ids])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True, help="Extracted MSD task folder (has dataset.json)")
    ap.add_argument("--public-key", type=Path, default=ROOT / "encryption" / "public_key.pem")
    ap.add_argument("--force", action="store_true", help="Overwrite existing data/public + data/hidden")
    a = ap.parse_args()

    cfg = load_config()
    public, hidden, private = ROOT / "data" / "public", ROOT / "data" / "hidden", ROOT / "data" / "private"
    if (public / "test_cases.csv").exists() and not a.force:
        raise SystemExit("data/public already built. Re-run with --force to rebuild (changes the test set!).")
    for d in (public, hidden, private):
        if a.force and d.exists() and d != private:
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    raw_meta = json.loads((a.raw_dir / "dataset.json").read_text(encoding="utf-8"))
    cases = sorted(case_name(e["image"]) for e in raw_meta["training"])

    rng = random.Random(cfg["split"]["seed"])
    rng.shuffle(cases)
    n_test = round(len(cases) * cfg["split"]["test_fraction"])
    n_val = round(len(cases) * cfg["split"]["val_fraction"])
    test, val, train = cases[:n_test], cases[n_test:n_test + n_val], sorted(cases[n_test + n_val:])
    val = sorted(val)

    # Noise and test-id order use OS randomness: not reproducible from the public seed.
    secret_rng = random.SystemRandom()
    nrng = np.random.default_rng(secret_rng.getrandbits(64))
    ncfg = cfg["noise"]
    noise_log = []

    def make_noisy(img):
        data, frac = noisy_image(np.asarray(img.dataobj), nrng, ncfg["rician_sigma_range"], ncfg["bias_field_strength"])
        return data, frac

    # --- train + val: noisy images; noisy labels for a fraction of TRAIN only ----------
    for sub in ("imagesTr", "labelsTr"):
        (public / sub).mkdir(exist_ok=True)
    n_lab_noise = round(len(train) * ncfg["train_label_noise_fraction"])
    label_noise_cases = set(secret_rng.sample(train, n_lab_noise))
    for cid in train + val:
        img = nib.load(str(a.raw_dir / "imagesTr" / f"{cid}.nii.gz"))
        lab = nib.load(str(a.raw_dir / "labelsTr" / f"{cid}.nii.gz"))
        data, frac = make_noisy(img)
        nib.save(nib.Nifti1Image(data, img.affine, img.header), str(public / "imagesTr" / f"{cid}.nii.gz"))
        ldata, op = np.asarray(lab.dataobj).astype(np.uint8), ""
        if cid in label_noise_cases:
            ldata, op = noisy_label(ldata, nrng)
        nib.save(nib.Nifti1Image(ldata, lab.affine, lab.header), str(public / "labelsTr" / f"{cid}.nii.gz"))
        noise_log.append([cid, "train" if cid in train else "val", f"{frac:.4f}", op])
    write_ids(public / "train.csv", train)
    write_ids(public / "val.csv", val)

    # --- test: anonymise ids, noisy images, clean hidden labels ------------------------
    (public / "imagesTs").mkdir(exist_ok=True)
    test_shuffled = test[:]
    secret_rng.shuffle(test_shuffled)

    labels, rows, mapping = {}, [], []
    for i, cid in enumerate(test_shuffled, start=1):
        anon = f"test_{i:03d}"
        img = nib.load(str(a.raw_dir / "imagesTr" / f"{cid}.nii.gz"))
        lab = nib.load(str(a.raw_dir / "labelsTr" / f"{cid}.nii.gz"))
        data, frac = make_noisy(img)
        noise_log.append([anon, "test", f"{frac:.4f}", ""])
        out = nib.Nifti1Image(data, img.affine)
        out.header.set_zooms(img.header.get_zooms()[:3])
        nib.save(out, str(public / "imagesTs" / f"{anon}.nii.gz"))

        labels[anon] = np.asarray(lab.dataobj).astype(np.uint8)
        sx, sy, sz = (float(z) for z in img.header.get_zooms()[:3])
        rows.append([anon, f"{anon}.nii.gz", *data.shape[:3], sx, sy, sz])
        mapping.append([anon, cid])

    rows.sort()
    with open(public / "test_cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "image", "shape_x", "shape_y", "shape_z", "spacing_x", "spacing_y", "spacing_z"])
        w.writerows(rows)

    buf = io.BytesIO()
    np.savez_compressed(buf, **labels)
    (hidden / "test_labels.npz.enc").write_bytes(encrypt_bytes(buf.getvalue(), a.public_key.read_bytes()))
    (private / "test_labels.npz").write_bytes(buf.getvalue())
    with open(private / "test_mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anon_id", "msd_case"])
        w.writerows(sorted(mapping))
    with open(private / "noise_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "split", "rician_sigma_frac", "label_noise"])
        w.writerows(noise_log)

    (public / "dataset.json").write_text(json.dumps({
        "name": cfg["name"],
        "source": f"Medical Segmentation Decathlon - {cfg['task']} (http://medicaldecathlon.com)",
        "description": raw_meta.get("description", ""),
        "reference": raw_meta.get("reference", ""),
        "licence": raw_meta.get("licence", "CC-BY-SA 4.0"),
        "modality": raw_meta.get("modality", {}),
        "labels": cfg["labels"],
        "numTrain": len(train), "numVal": len(val), "numTest": len(test),
        "noise": {
            "images": "all images (train/val/test): MRI bias field + Rician noise, per-case noise level",
            "labels": f"{n_lab_noise} of {len(train)} TRAIN labels randomly eroded/dilated by 1 voxel; val/test labels clean",
            **ncfg,
        },
    }, indent=2), encoding="utf-8")

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    print("Public data   -> data/public/")
    print("Hidden labels -> data/hidden/test_labels.npz.enc (commit)")
    print("Plaintext     -> data/private/ (gitignored, keep safe)")


if __name__ == "__main__":
    main()
