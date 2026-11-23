# HippoSeg Decathlon: 3D Hippocampus Segmentation Challenge

**HippoSeg Decathlon** is a prediction-only, Kaggle-style competition that runs entirely on GitHub.
It is built on **Task04_Hippocampus** of the [Medical Segmentation Decathlon](http://medicaldecathlon.com).

Participants train models on their own machines and submit **encrypted segmentation masks** in a Pull Request.
GitHub Actions scores each submission against hidden labels, comments the result on the PR and updates the
live leaderboard.

**Live site:** <https://mahatrabelsi1.github.io/Medical-Segmentation-Decathlon/>
**Leaderboard:** <https://mahatrabelsi1.github.io/Medical-Segmentation-Decathlon/#leaderboard>
([Markdown version](leaderboard/leaderboard.md))

---

## Task

| | |
|---|---|
| Problem | Supervised **3D semantic segmentation** of MRI volumes |
| Input | Cropped T1-weighted MRI around one hippocampus (about 35×50×35 voxels, 1 mm isotropic) |
| Output | A label map with one value per voxel: `0` background, `1` anterior, `2` posterior |
| Train / Val / Test | 182 / 26 / 52 volumes (the test labels are hidden) |
| Metric | **Mean Dice** over cases and the two classes. Ties are broken by **NSD @ 1 mm** |
| Twist | **Noisy images + noisy training labels.** Clean labels are used only for scoring |

Each case is a cropped T1-weighted MRI volume around one hippocampus.

### Noise: what makes this harder than plain MSD

The organizers corrupted the data on purpose to make this a robustness benchmark. The noise settings are in
`competition.json` under `"noise"`.

| What | Where | How |
|---|---|---|
| **MRI bias field** | every image (train, val, test) | smooth multiplicative intensity drift, up to about ±25% |
| **Rician noise** | every image (train, val, test) | real-MRI magnitude noise. Sigma is 2-10% of the 99th-percentile intensity and **varies per case** |
| **Label noise** | **25% of training labels** (46 of 182 cases, not disclosed which) | each class randomly **eroded or dilated by 1 voxel**, as if annotators disagreed about boundaries |
| Clean | validation labels, hidden test labels | so your local val score and the leaderboard measure the true error |

Ideas that tend to help: intensity normalisation and bias-field correction, denoising, augmentation with noise,
and noise-robust losses. You can also detect and down-weight suspicious training labels.

---

## Repository layout

```text
competition.json                 # task, labels, split seed, metric settings, submission limits
data/public/                     # everything participants train and predict on
├── imagesTr/  labelsTr/         #   train + val cases (hippocampus_XXX.nii.gz)
├── imagesTs/                    #   test images, anonymised (test_XXX.nii.gz)
├── train.csv  val.csv           #   case-id splits
├── test_cases.csv               #   test ids, expected shapes, voxel spacing
└── dataset.json
data/hidden/test_labels.npz.enc  # encrypted test ground truth (only CI can decrypt)
encryption/                      # public key + RSA/Fernet encrypt/decrypt tools
scoring/                         # metrics, validators, CI scorer, leaderboard writer, local scorer
starter/                         # 3D U-Net baseline + submission packager
submissions/inbox/               # where your PR adds files
submissions/scored/              # archive of scored (still encrypted) submissions, written by the bot
leaderboard/                     # leaderboard.csv + leaderboard.md
docs/                            # GitHub Pages site (overview, data, evaluation, leaderboard, rules)
organizer/                       # dataset builder + setup guide (organizers only)
.github/workflows/               # score_submission.yml, rebuild_leaderboard.yml
```

---

## How to participate

```bash
# 1. Fork this repo on GitHub, then:
git clone https://github.com/<you>/<repo>.git && cd <repo>
pip install -r requirements.txt

# 2. Train. Optional: start from the baseline (see "Baseline training time" below)
pip install torch
python starter/baseline_unet.py --epochs 40 --out runs/baseline

# 3. Check yourself on the validation split, using the official metric code
python scoring/score_local.py --pred-dir runs/baseline/val

# 4. Package and encrypt the test predictions
python starter/make_submission.py --pred-dir runs/baseline/test \
    --team my_team --run-id unet_v1 --author-type human \
    --model "3D U-Net" --notes "CE + Dice, flip TTA"

# 5. Commit the 2 generated files on a branch and open a PR to main
git checkout -b submit/my_team-unet_v1
git add submissions/inbox/my_team/unet_v1
git commit -m "Submission: my_team/unet_v1"
git push origin submit/my_team-unet_v1
```

### Baseline training time

`starter/baseline_unet.py` is a small 3D U-Net (about 1.4M parameters) trained on all 182 training volumes padded to 48×64×48.

| Hardware | 1 epoch | Default run (`--epochs 40`) |
|---|---|---|
| Laptop / desktop CPU | about 30-45 s | **about 15-30 min (estimated)** |
| Any recent NVIDIA GPU | about 1-2 s | about 1-2 min |

These are estimates, not benchmarks. A 2-epoch smoke test on CPU took about 1-2 min including data loading
and already reached a val Dice of about 0.80 on the clean data. The 40-epoch run was not timed to completion.
For a quick check, use `--epochs 10`. The script uses the GPU automatically when one is available.

### Prediction format

- One label map per test case named `<case_id>.nii.gz` (for example `test_001.nii.gz`). Every id in
  `data/public/test_cases.csv` must be present.
- Same shape as the matching image in `data/public/imagesTs/`. Integer values in `{0, 1, 2}`.
- `make_submission.py` packs the masks into one compressed `.npz` (one `uint8` array per case id). It checks
  them with the **same validator the CI uses** and encrypts the result with `encryption/public_key.pem`.
  If you already have an `.npz`, pass `--npz preds.npz` instead of `--pred-dir`.

### What happens after you open the PR

1. The bot checks that the PR only adds `submissions/inbox/<team>/<run_id>/{predictions.npz.enc,metadata.json}`.
2. It validates `metadata.json` and your remaining submission quota.
3. It decrypts and validates your masks (case ids, shapes, label values).
4. It scores them with Dice and NSD against the hidden labels.
5. It commits the leaderboard update to `main` and archives your encrypted run in `submissions/scored/`.
6. It **comments your scores on the PR and closes it**. You do not need to merge.

If something is wrong, the bot comments the reason and leaves the PR open. Push a fix to the same branch and it
re-runs.

---

## Evaluation

For each test case and each foreground class `c`:

- **Dice** = `2|P∩G| / (|P|+|G|)`. It is 1 if both masks are empty.
- **NSD** is the fraction of surface voxels of P and G that lie within **1 mm** of the other surface.

Per-class scores are averaged over cases and then over the two classes. The ranking uses `dice_mean`, and ties
are broken by `nsd_mean`. The implementation is in [scoring/metrics.py](scoring/metrics.py).

---

## Rules

- At most **3 scored runs per team** (`max_submissions_per_team` in `competition.json`). Each run needs a new
  `run_id`. The leaderboard shows each team's best run.
- Train only on `data/public/`. **Do not use the original (clean) MSD release in any way.** Do not use it to look
  up test cases, to get their labels, or to replace the noisy training data. The test set is carved from the
  public MSD training labels, so doing this is cheating.
- Pre-trained weights are allowed if you disclose them in `notes`.
- Set `author_type` honestly: `human`, `llm` or `hybrid`.
- Submission PRs may only contain the two submission files.

---

## Security model

- Masks are encrypted locally with a hybrid scheme: a random Fernet key encrypts the file, and RSA-OAEP-4096
  encrypts that key. Only the encrypted file is in the PR.
- The hidden test labels are committed in the same encrypted form (`data/hidden/test_labels.npz.enc`).
- The only secret is the RSA private key, stored as the GitHub secret `SUBMISSION_PRIVATE_KEY`. Decryption
  happens only on the Actions runner.
- The workflow runs on `pull_request_target`, but **it never executes code from the PR**. It sparse-checks out
  only the run folder and reads it as data with scripts from `main`. It loads `.npz` files with
  `allow_pickle=False` and checks sizes, shapes and symlinks.

---

## Data licence and citation

The MSD data is released under **CC-BY-SA 4.0**. If you use it, please cite:

- M. Antonelli et al., *The Medical Segmentation Decathlon*, Nature Communications 13, 4128 (2022).
- A. L. Simpson et al., *A large annotated medical image dataset for the development and evaluation of
  segmentation algorithms*, arXiv:1902.09063 (2019).
