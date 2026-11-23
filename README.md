# 🏆 [View the Leaderboard](https://mahatrabelsi1.github.io/Medical-Segmentation-Decathlon/#leaderboard)

# HippoSeg Decathlon: Hippocampus Segmentation under Noise

A Kaggle-style competition that runs entirely on GitHub, built on Task04 of the
[Medical Segmentation Decathlon](http://medicaldecathlon.com).

**Train a model → predict the test set → open a Pull Request → a bot scores your submission and updates the leaderboard.**

🌐 Competition site: <https://mahatrabelsi1.github.io/Medical-Segmentation-Decathlon/>

---

## The task

Label every voxel of a 3D brain MRI as **background (0)**, **anterior hippocampus (1)** or **posterior hippocampus (2)**.

| | |
|---|---|
| Data | 182 train · 26 validation · 52 hidden test volumes (about 35×50×35 voxels, 1 mm) |
| Metric | **Mean Dice** (higher is better). Ties are broken by NSD at 1 mm |
| Submissions | **3 per team**. Your best one counts |

### ⚠️ The twist: noise

- **All images** have MRI noise: a bias field plus Rician noise, at a different level per case.
- **25% of training labels** are wrong at the edges (eroded or dilated by 1 voxel). You aren't told which.
- **Validation and test labels are clean**, so scores measure the real error.

---

## How to submit (5 steps)

**1. Fork this repo and install**
```bash
git clone https://github.com/<you>/Medical-Segmentation-Decathlon.git
cd Medical-Segmentation-Decathlon
pip install -r requirements.txt
```

**2. Train your model** on `data/public/imagesTr` + `labelsTr` (case lists are in `train.csv` and `val.csv`).
Want a starting point? There's a ready-made 3D U-Net:
```bash
pip install torch
python starter/baseline_unet.py --epochs 40 --out runs/baseline
```

**3. Check your score on validation**
```bash
python scoring/score_local.py --pred-dir runs/baseline/val
```

**4. Predict the test set and package it.** Save one mask per test case as `test_001.nii.gz`, `test_002.nii.gz`, …, then run:
```bash
python starter/make_submission.py --pred-dir runs/baseline/test \
    --team my_team --run-id run_01 --author-type human --model "3D U-Net"
```
This checks your masks and creates 2 encrypted files in `submissions/inbox/my_team/run_01/`.

**5. Open a Pull Request** with only those 2 files:
```bash
git checkout -b submit/my_team-run_01
git add submissions/inbox/my_team/run_01
git commit -m "Submission: my_team/run_01"
git push origin submit/my_team-run_01
```
Then click **Create pull request** on GitHub. Within a few minutes the bot will:
✅ score your submission · 💬 comment your scores on the PR · 🏆 update the leaderboard · 🔒 close the PR (you don't need to merge it)

If something is wrong, the bot comments the reason. Fix it and push again to the same branch.

---

## ⏱️ How long does training take?

| Hardware | Baseline, 40 epochs |
|---|---|
| CPU (laptop or desktop) | about 15-30 min (estimate) |
| NVIDIA GPU | about 1-2 min (estimate) |

For a quick test, use `--epochs 10`.

---

## Rules

- At most **3 submissions per team**. Each one needs a new `--run-id`.
- Use only the data in this repo. **Don't use the original MSD download** in any way.
- Pre-trained weights are OK if you mention them in `--notes`.
- Set `--author-type` honestly: `human`, `llm` or `hybrid`.
- A submission PR may only contain your 2 submission files.

---

<details>
<summary><b>More details: data files, metric, security, repo layout</b></summary>

### Data files
```text
data/public/imagesTr/   labelsTr/     train + val (hippocampus_XXX.nii.gz)
data/public/imagesTs/                 test images (test_XXX.nii.gz), labels hidden
data/public/train.csv  val.csv        case ids
data/public/test_cases.csv            test ids + expected shape + spacing
data/hidden/test_labels.npz.enc       encrypted test labels (only the bot can read them)
```

### Metric
For each test case and each class: Dice = 2|P∩G| / (|P|+|G|). Results are averaged over cases, then over the 2 classes.
NSD is the fraction of boundary voxels within 1 mm of the other mask's boundary. Code: [scoring/metrics.py](scoring/metrics.py).

### Security
Your masks are encrypted with `encryption/public_key.pem` (RSA-4096 + Fernet). Only the bot, which holds the
private key in a GitHub secret, can decrypt them. The bot never runs code from a PR; it only reads your 2 files.

### Repo layout
```text
competition.json     settings (labels, noise, limits)
starter/             baseline U-Net + make_submission.py
scoring/             metrics, scorer, local validation
encryption/          public key + encrypt tool
leaderboard/         leaderboard.csv + leaderboard.md
docs/                competition website
organizer/           dataset builder + setup guide (organizers only)
```

</details>

---

**Data licence:** CC-BY-SA 4.0. **Please cite:** Antonelli et al., *The Medical Segmentation Decathlon*,
Nature Communications 13, 4128 (2022).
