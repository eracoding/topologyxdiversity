#!/usr/bin/env python3
"""A13: Depth truncation re-aggregation for ALL THREE depth configs.

Generalizes A3 (c_l3_truncation) to also cover C-L1 and C-L2. For each
(config, language, split), computes Macro-F1 of the predictions you'd get
by truncating the depth chain at each layer:

  L1 only:    final = layer_1.emotions
  L1+L2:      final = layer_2.emotions  (already conditioned on L1)
  L1+L2+L3:   final = layer_3.emotions  (canonical baseline)

For C-L3 the numbers reproduce A3 exactly. The new question this script
answers is whether the "L1 alone is the best operating point" finding
generalizes from C-L3 (specialist adapters) to C-L1 (stochastic prompt
diversity at each layer) and C-L2 (Analyst → Critic → Calibrator role
decomposition). If it does, the depth-pipeline structure itself — not the
specialist adapters — is what's harming performance.

All values are read from the existing JSONLs in outputs/predictions.
JSONL keys per row: text_id, text, gold, layer_1, layer_2, layer_3,
aggregated.

Usage:
    python ablation/c_l1_l2_truncation/scripts/score_truncations_all.py --split dev
    python ablation/c_l1_l2_truncation/scripts/score_truncations_all.py --split test
    # or restrict:
    python ablation/c_l1_l2_truncation/scripts/score_truncations_all.py \\
        --split dev --configs c_l1 c_l2
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_f1_scores  # noqa: E402
from src.data.loader import EMOTIONS  # noqa: E402

LANGUAGES = ["pcm", "chn", "mar", "ary", "tat", "vmw", "ptmz", "zul", "ind"]
CONFIGS = ["c_l1", "c_l2", "c_l3"]


def gold_to_dict(gold) -> dict[str, int]:
    if isinstance(gold, dict):
        return {e: int(gold.get(e, 0)) for e in EMOTIONS}
    if isinstance(gold, list):
        present = set(gold)
        return {e: 1 if e in present else 0 for e in EMOTIONS}
    raise ValueError(f"Unknown gold format: {type(gold).__name__}")


def layer_to_pred(layer_dict: dict) -> dict[str, int]:
    return {e: int(layer_dict["emotions"].get(e, 0)) for e in EMOTIONS}


def evaluate_language(jsonl_path: Path) -> dict:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not rows:
        return {}
    gold = [gold_to_dict(r["gold"]) for r in rows]
    l1 = [layer_to_pred(r["layer_1"]) for r in rows]
    l2 = [layer_to_pred(r["layer_2"]) for r in rows]
    l3 = [layer_to_pred(r["layer_3"]) for r in rows]
    return {
        "n_samples": len(rows),
        "l1_only": compute_f1_scores(gold, l1)["macro_f1"],
        "l1_l2": compute_f1_scores(gold, l2)["macro_f1"],
        "l1_l2_l3": compute_f1_scores(gold, l3)["macro_f1"],
    }


def run_config(config: str, split: str, pred_dir: Path, out_dir: Path,
               languages: list[str]) -> dict | None:
    """Process one config (c_l1 / c_l2 / c_l3). Returns avg dict or None."""
    print("\n" + "=" * 80)
    print(f"CONFIG: {config.upper()}  —  split={split}")
    print("=" * 80)
    print(f"{'lang':6s} {'n':>5s} {'L1 only':>9s} {'L1+L2':>9s} {'L1+L2+L3':>10s}  | "
          f"{'Δ L1-canon':>11s} {'Δ L1+L2-canon':>14s}")
    print("-" * 80)

    rows = []
    for lang in languages:
        jsonl = pred_dir / f"{config}_{lang}_{split}.jsonl"
        if not jsonl.exists():
            print(f"  SKIP {lang}: {jsonl} not found")
            continue
        r = evaluate_language(jsonl)
        if not r:
            continue
        rows.append({
            "language": lang,
            "n_samples": r["n_samples"],
            "l1_only": round(r["l1_only"], 2),
            "l1_l2": round(r["l1_l2"], 2),
            "l1_l2_l3": round(r["l1_l2_l3"], 2),
            "delta_l1_minus_canon": round(r["l1_only"] - r["l1_l2_l3"], 2),
            "delta_l1l2_minus_canon": round(r["l1_l2"] - r["l1_l2_l3"], 2),
        })
        print(f"  {lang:5s} {r['n_samples']:>5d} {r['l1_only']:>9.2f} {r['l1_l2']:>9.2f} "
              f"{r['l1_l2_l3']:>10.2f}  | {r['l1_only']-r['l1_l2_l3']:>+11.2f} "
              f"{r['l1_l2']-r['l1_l2_l3']:>+14.2f}")

    if not rows:
        print("  (no data)")
        return None

    df = pd.DataFrame(rows)
    avg = {"language": "AVERAGE", "n_samples": int(df["n_samples"].mean())}
    for col in df.columns:
        if col in ("language", "n_samples"):
            continue
        avg[col] = round(df[col].mean(), 2)
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

    out_path = out_dir / f"{config}_truncation_{split}_results.csv"
    df.to_csv(out_path, index=False)

    print("-" * 80)
    print(f"  {'AVG':5s} {avg['n_samples']:>5d} {avg['l1_only']:>9.2f} {avg['l1_l2']:>9.2f} "
          f"{avg['l1_l2_l3']:>10.2f}  | {avg['delta_l1_minus_canon']:>+11.2f} "
          f"{avg['delta_l1l2_minus_canon']:>+14.2f}")
    print(f"  saved → {out_path}")
    return {"config": config, **avg}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--output-dir", default=None,
                        help="Default: ablation/c_l1_l2_truncation/results")
    parser.add_argument("--configs", nargs="+", default=CONFIGS,
                        choices=CONFIGS,
                        help="Subset of depth configs to score (default: all 3)")
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "ablation" / "c_l1_l2_truncation" / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDepth-truncation re-aggregation — split={args.split}")
    print(f"Predictions dir: {pred_dir}")
    print(f"Output dir:      {out_dir}")
    print(f"Configs:         {' '.join(args.configs)}")
    print(f"Languages:       {' '.join(args.languages)}")

    summary_rows = []
    for cfg in args.configs:
        avg = run_config(cfg, args.split, pred_dir, out_dir, args.languages)
        if avg is not None:
            summary_rows.append(avg)

    if not summary_rows:
        print("\nNo data — exiting.")
        return

    # Cross-config summary
    print("\n" + "#" * 80)
    print(f"CROSS-CONFIG SUMMARY — split={args.split} (averages across languages)")
    print("#" * 80)
    print(f"{'config':6s} {'L1 only':>9s} {'L1+L2':>9s} {'L1+L2+L3':>10s}  | "
          f"{'Δ L1-canon':>11s} {'Δ L1+L2-canon':>14s}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['config']:6s} {r['l1_only']:>9.2f} {r['l1_l2']:>9.2f} "
              f"{r['l1_l2_l3']:>10.2f}  | {r['delta_l1_minus_canon']:>+11.2f} "
              f"{r['delta_l1l2_minus_canon']:>+14.2f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"summary_truncation_{args.split}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCross-config summary saved → {summary_path}")


if __name__ == "__main__":
    main()
