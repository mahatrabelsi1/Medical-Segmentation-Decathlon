# Organizer setup

This is a one-time setup. Participants never need any of it.

## 1. Build the data (already done once in this repo)

```bash
pip install -r requirements.txt
python encryption/generate_keys.py          # writes encryption/public_key.pem + encryption/private_key.pem

mkdir -p data/raw && cd data/raw
curl -LO https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar
tar -xf Task04_Hippocampus.tar && cd ../..

python organizer/build_dataset.py --raw-dir data/raw/Task04_Hippocampus
```

This writes `data/public/` and `data/hidden/test_labels.npz.enc` (commit both). It also writes `data/private/`
(gitignored) with the plaintext test labels, the `test_XXX -> hippocampus_XXX` mapping and `noise_log.csv`.
`noise_log.csv` records which training labels were corrupted and the noise level of each case. Keep it secret.

The noise comes from OS randomness, so every rebuild produces different noise. Settings are under
`"noise"` in `competition.json`, and the noise models are in `organizer/noise.py`.

> The keys and the encrypted labels belong together. If you regenerate the keys, re-run `build_dataset.py --force`.
> That also re-shuffles the test ids, which invalidates every existing submission.

## 2. Add the secret (the only one the workflows need)

The workflows need exactly **one** secret, and it is not split into parts.

1. Open `encryption/private_key.pem` in a text editor and copy **everything**, including the first line
   `-----BEGIN PRIVATE KEY-----` and the last line `-----END PRIVATE KEY-----`.
   On the command line: `cat encryption/private_key.pem | clip` (Windows) or `pbcopy < encryption/private_key.pem` (macOS).
2. On GitHub, open the repo and go to **Settings → Secrets and variables → Actions → New repository secret**.
3. Enter:

   | Field | Value |
   |---|---|
   | Name | `SUBMISSION_PRIVATE_KEY` |
   | Secret | the copied PEM text |

4. Click **Add secret**.

Both workflows read it as `${{ secrets.SUBMISSION_PRIVATE_KEY }}`. The key decrypts participants' submissions and
`data/hidden/test_labels.npz.enc`.

Do **not** commit `encryption/private_key.pem` (it is gitignored). Keep an offline backup of it, for example in a
password manager. If you lose it, nothing can be scored, and you must regenerate the keys and rebuild the data.

## 3. Allow the bot to push

- **Settings → Actions → General → Workflow permissions**: select *Read and write permissions*.
- If `main` is protected, allow `github-actions[bot]` to bypass the rule, or the leaderboard commit fails.
- `pull_request_target` always runs the workflow file from `main`, so a PR cannot change the scorer.
  Fork PRs get the secret, which is safe here because the workflow only reads the PR's two files as data.

## 4. Turn on GitHub Pages

**Settings → Pages → Build and deployment**: *Deploy from a branch*, branch `main`, folder `/docs`.
The site is then at `https://<owner>.github.io/<repo>/` and the leaderboard tab is at `#leaderboard`.

## 5. Optional: seed the leaderboard with the baseline

```bash
pip install torch
python starter/baseline_unet.py --epochs 40 --out runs/baseline
python starter/make_submission.py --pred-dir runs/baseline/test --team organizers \
    --run-id baseline_unet --author-type human --model "3D U-Net baseline (starter)"
# open a PR with submissions/inbox/organizers/baseline_unet/ like any participant
```

## Maintenance

- **Rebuild the leaderboard** (for example after a metric fix): Actions → *Rebuild leaderboard (manual)* → Run.
  It re-scores everything in `submissions/scored/`.
- **Remove a run** (for example a rule violation): delete its folder in `submissions/scored/` and its row in
  `leaderboard/leaderboard.csv`, then run the rebuild workflow.
- **Change limits**: edit `competition.json` (`max_submissions_per_team`, `max_submission_bytes`,
  `metric.nsd_tolerance_mm`).
- **Use another MSD task** (for example `Task02_Heart`, `Task09_Spleen`): update `task`, `source_url` and
  `labels` in `competition.json`, then re-run `build_dataset.py --force`. The scoring code is task-agnostic.
  Larger tasks have images too big to commit comfortably to git. Host `data/public/` elsewhere (for example a
  GitHub Release) and keep only the CSVs plus `data/hidden/` in the repo.

## Known limitation

The test set comes from the public MSD *training* labels, because the official MSD test labels were never
released. A participant who downloads the original MSD release could match test images to their originals.
The rules forbid this. Large score jumps with sparse notes are worth checking with the mapping in
`data/private/test_mapping.csv`.
