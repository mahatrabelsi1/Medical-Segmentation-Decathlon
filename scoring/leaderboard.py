"""Read / rank / write the leaderboard (CSV for the site, Markdown for the repo)."""
from __future__ import annotations

import csv
from pathlib import Path

from common import ROOT, foreground_labels, load_config

LB_CSV = ROOT / "leaderboard" / "leaderboard.csv"
LB_MD = ROOT / "leaderboard" / "leaderboard.md"
DOCS_CSV = ROOT / "docs" / "leaderboard.csv"


def columns(cfg: dict) -> list[str]:
    names = list(foreground_labels(cfg).values())
    return (["team", "run_id", "author_type", "model", "notes", "dice_mean", "nsd_mean"]
            + [f"dice_{n}" for n in names] + [f"nsd_{n}" for n in names] + ["submitted_at"])


def read_rows() -> list[dict]:
    if not LB_CSV.exists():
        return []
    with open(LB_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _key(r: dict):
    return (-float(r["dice_mean"]), -float(r["nsd_mean"]), r["submitted_at"])


def best_per_team(rows: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for r in sorted(rows, key=_key):
        best.setdefault(r["team"].lower(), r)
    ranked = sorted(best.values(), key=_key)
    out, prev, rank = [], None, 0
    for i, r in enumerate(ranked, start=1):
        score = (round(float(r["dice_mean"]), 6), round(float(r["nsd_mean"]), 6))
        if score != prev:
            rank, prev = i, score
        out.append({**r, "rank": rank})
    return out


def write_all(rows: list[dict]) -> None:
    cfg = load_config()
    cols = columns(cfg)
    LB_CSV.parent.mkdir(parents=True, exist_ok=True)
    for path in (LB_CSV, DOCS_CSV):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["submitted_at"]))

    names = list(foreground_labels(cfg).values())
    lines = [f"# {cfg['name']} - Leaderboard\n\n",
             "Best submission per team. Ranked by mean Dice, ties broken by mean NSD.\n\n",
             "| Rank | Team | Run | Author | Model | Dice | NSD | "
             + " | ".join(f"Dice {n}" for n in names) + " | Submitted |\n",
             "|---:|---|---|---|---|---:|---:|" + "---:|" * len(names) + "---|\n"]
    for r in best_per_team(rows):
        cells = " | ".join(f"{float(r[f'dice_{n}']):.4f}" for n in names)
        model = r["model"].replace("|", "/")
        lines.append(f"| {r['rank']} | {r['team']} | {r['run_id']} | {r['author_type']} | {model} | "
                     f"{float(r['dice_mean']):.4f} | {float(r['nsd_mean']):.4f} | {cells} | {r['submitted_at']} |\n")
    if not rows:
        lines.append("\n_No submissions yet._\n")
    LB_MD.write_text("".join(lines), encoding="utf-8")
