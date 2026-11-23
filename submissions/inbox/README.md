# Submission inbox

A submission PR adds exactly two files:

```text
submissions/inbox/<team>/<run_id>/predictions.npz.enc
submissions/inbox/<team>/<run_id>/metadata.json
```

Generate both with:

```bash
python starter/make_submission.py --pred-dir <folder of test_XXX.nii.gz> \
    --team <team> --run-id <run_id> --author-type human --model "<model>"
```

- `team` and `run_id`: 2-40 characters from letters, digits, `_`, `-` and `.`. They must match the folder names.
- `author_type`: `human`, `llm` or `hybrid`.
- Each team gets 3 scored runs, and each run needs a new `run_id`.

The PR is scored automatically and closed, so you don't need to merge it. Scored runs are archived
(still encrypted) in `submissions/scored/`.
