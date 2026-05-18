#!/usr/bin/env python3
"""A4: Bootstrap 95% CIs for Macro-F1 across every config and language.

For each (config, language, split), reads the existing prediction JSONL,
takes the canonical aggregated prediction per row, and bootstraps the
Macro-F1 by resampling text_ids with replacement N times. Reports mean,
2.5th, and 97.5th percentiles per language; computes a per-config average.

Configs covered (those with JSONLs present in --predictions-dir):
  width:  b_l1, b_l2, b_l3
  depth:  c_l1, c_l2, c_l3

The aggregated prediction comes from the row's "aggregated" field. For width,
that's the canonical 2-of-3 majority vote; for depth, that's the final layer
output (layer_3 for C-L3, etc.).

Usage:
    python ablation/bootstrap_ci/scripts/bootstrap_ci.py --split test --n-bootstrap 1000
    python ablation/bootstrap_ci/scripts/bootstrap_ci.py --split dev  --n-bootstrap 1000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_f1_scores
from src.data.loader import EMOTIONS

LANGUAGES = ["pcm", "chn", "mar", "ary", "tat", "vmw", "ptmz", "zul", "ind"]
CONFIGS = ["b_l1", "b_l2", "b_l3", "c_l1", "c_l2", "c_l3"]


def gold_to_dict(gold) -> dict[str, int]:
    if isinstance(gold, dict):
        return {e: int(gold.get(e, 0)) for e in EMOTIONS}
    if isinstance(gold, list):
        present = set(gold)
        return {e: 1 if e in present else 0 for e in EMOTIONS}
    raise ValueError(f"Unknown gold format: {type(gold).__name__}")


def aggregated_to_dict(agg) -> dict[str, int]:
    """Pull the binary prediction vector out of the aggregated field.

    Width JSONLs: aggregated = {"emotions": {...}, ...}
    Depth JSONLs: aggregated may be the same shape, OR may be the final layer dict.
    Both cases have an "emotions" sub-dict.
    """
    if isinstance(agg, dict):
        emo = agg.get("emotions", agg)
        return {e: int(emo.get(e, 0)) for e in EMOTIONS}
    raise ValueError(f"Unknown aggregated shape: {type(agg).__name__}")


def bootstrap_macro_f1(gold: list[dict], preds: list[dict],
                       n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Returns (mean, p2.5, p97.5) of macro_f1 over n_bootstrap resamples."""
    n = len(gold)
    indices = np.arange(n)
    scores = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sample = rng.choice(indices, size=n, replace=True)
        g_sample = [gold[i] for i in sample]
        p_sample = [preds[i] for i in sample]
        scores[b] = compute_f1_scores(g_sample, p_sample)["macro_f1"]
    return float(scores.mean()), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate_language(jsonl_path: Path, n_bootstrap: int, seed: int) -> dict:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not rows:
        return {}
    gold = [gold_to_dict(r["gold"]) for r in rows]
    preds = [aggregated_to_dict(r["aggregated"]) for r in rows]
    point = compute_f1_scores(gold, preds)["macro_f1"]
    rng = np.random.default_rng(seed)
    mean, lo, hi = bootstrap_macro_f1(gold, preds, n_bootstrap, rng)
    return {
        "n_samples": len(rows),
        "macro_f1_point": point,
        "macro_f1_boot_mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_half_width": (hi - lo) / 2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--configs", nargs="+", default=CONFIGS)
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    out_root = PROJECT_ROOT / "ablation" / "bootstrap_ci" / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    for cfg in args.configs:
        print(f"\n=== {cfg} ({args.split}, n_bootstrap={args.n_bootstrap}) ===")
        print(f"{'lang':6s} {'n':>5s} {'point':>7s} {'boot_mean':>10s} "
              f"{'CI_lo':>7s} {'CI_hi':>7s} {'±halfwidth':>11s}")
        print("-" * 60)

        rows = []
        for lang in args.languages:
            jsonl = pred_dir / f"{cfg}_{lang}_{args.split}.jsonl"
            if not jsonl.exists():
                print(f"  SKIP {lang}: {jsonl} not found")
                continue
            r = evaluate_language(jsonl, args.n_bootstrap, args.seed)
            if not r:
                continue
            rows.append({
                "language": lang,
                "n_samples": r["n_samples"],
                "macro_f1_point": round(r["macro_f1_point"], 2),
                "macro_f1_boot_mean": round(r["macro_f1_boot_mean"], 2),
                "ci_lo": round(r["ci_lo"], 2),
                "ci_hi": round(r["ci_hi"], 2),
                "ci_half_width": round(r["ci_half_width"], 2),
            })
            print(f"  {lang:5s} {r['n_samples']:>5d} {r['macro_f1_point']:>7.2f} "
                  f"{r['macro_f1_boot_mean']:>10.2f} {r['ci_lo']:>7.2f} {r['ci_hi']:>7.2f} "
                  f"{r['ci_half_width']:>11.2f}")

        if not rows:
            continue

        df = pd.DataFrame(rows)
        avg = {"language": "AVERAGE", "n_samples": int(df["n_samples"].mean())}
        for col in ["macro_f1_point", "macro_f1_boot_mean", "ci_lo", "ci_hi", "ci_half_width"]:
            avg[col] = round(df[col].mean(), 2)
        df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
        out_path = out_root / f"{cfg}_bootstrap_{args.split}.csv"
        df.to_csv(out_path, index=False)
        print(f"  AVG point={avg['macro_f1_point']:.2f}  "
              f"boot_mean={avg['macro_f1_boot_mean']:.2f}  "
              f"CI=[{avg['ci_lo']:.2f}, {avg['ci_hi']:.2f}]  "
              f"±{avg['ci_half_width']:.2f}")
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
