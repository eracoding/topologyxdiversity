#!/usr/bin/env python3
"""A3: C-L3 depth truncation re-aggregation (no new inference).

For each language, computes Macro-F1 of the predictions you'd get by
truncating the depth chain at each layer:
  L1 only:    final = layer_1.emotions
  L1+L2:      final = layer_2.emotions  (already conditioned on L1)
  L1+L2+L3:   final = layer_3.emotions  (canonical baseline)

All values are read from the existing C-L3 JSONLs in outputs/predictions.
JSONL keys per row: text_id, text, gold, layer_1, layer_2, layer_3, aggregated.

Usage:
    python ablation/c_l3_truncation/scripts/score_truncations.py --split dev
    python ablation/c_l3_truncation/scripts/score_truncations.py --split test
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_f1_scores
from src.data.loader import EMOTIONS

LANGUAGES = ["pcm", "chn", "mar", "ary", "tat", "vmw", "ptmz", "zul", "ind"]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--output", default=None)
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "ablation" / "c_l3_truncation" / "results"
        / f"c_l3_truncation_{args.split}_results.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nC-L3 truncation — split={args.split}")
    print(f"Predictions: {pred_dir}")
    print(f"Output:      {output_path}\n")
    print(f"{'lang':6s} {'n':>5s} {'L1 only':>9s} {'L1+L2':>9s} {'L1+L2+L3':>10s}  | "
          f"{'Δ L1-canon':>11s} {'Δ L1+L2-canon':>14s}")
    print("-" * 80)

    rows = []
    for lang in args.languages:
        jsonl = pred_dir / f"c_l3_{lang}_{args.split}.jsonl"
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
        print("\nNo data — exiting.")
        return

    df = pd.DataFrame(rows)
    avg = {"language": "AVERAGE", "n_samples": int(df["n_samples"].mean())}
    for col in df.columns:
        if col in ("language", "n_samples"):
            continue
        avg[col] = round(df[col].mean(), 2)
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print(f"AVERAGE: L1={avg['l1_only']:.2f}  L1+L2={avg['l1_l2']:.2f}  "
          f"L1+L2+L3={avg['l1_l2_l3']:.2f}")
    print(f"  Δ(L1 - canon)    = {avg['delta_l1_minus_canon']:+.2f}")
    print(f"  Δ(L1+L2 - canon) = {avg['delta_l1l2_minus_canon']:+.2f}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
