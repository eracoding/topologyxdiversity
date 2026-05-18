#!/usr/bin/env python3
"""A5: Aggregation-strategy sweep across all width topologies (no new inference).

For each topology in {b_l1, b_l2, b_l3} and each language, re-aggregates the
3 agents under several decision rules:
  any_of_3:           predict 1 iff >=1 agent predicts 1     (high recall)
  majority_2_of_3:    predict 1 iff >=2 agents predict 1     (canonical)
  unanimous_3_of_3:   predict 1 iff all 3 predict 1          (high precision)
  confidence_avg:     predict 1 iff mean confidence >= 0.5
  confidence_max:     predict 1 iff any confidence >= 0.5

Question: does the canonical 2-of-3 rule actually outperform alternatives, or
is it accidental? Particularly relevant given A2's LOO finding that low-baseline
languages (vmw/zul/ind) gain 5-12pp under the permissive (any-of-2) rule.

Reads existing JSONLs from outputs/predictions; writes one CSV per (topology, split).

Usage:
    python ablation/aggregation_strategy/scripts/score_strategies.py --split dev
    python ablation/aggregation_strategy/scripts/score_strategies.py --split test
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
TOPOLOGIES = ["b_l1", "b_l2", "b_l3"]
STRATEGIES = [
    "any_of_3",
    "majority_2_of_3",
    "unanimous_3_of_3",
    "confidence_avg",
    "confidence_max",
]


def gold_to_dict(gold) -> dict[str, int]:
    if isinstance(gold, dict):
        return {e: int(gold.get(e, 0)) for e in EMOTIONS}
    if isinstance(gold, list):
        present = set(gold)
        return {e: 1 if e in present else 0 for e in EMOTIONS}
    raise ValueError(f"Unknown gold format: {type(gold).__name__}")


def aggregate(triple: list[dict], strategy: str) -> dict[str, int]:
    """Apply one aggregation strategy to one row's 3 agent dicts."""
    out = {}
    for e in EMOTIONS:
        votes = [int(a["emotions"].get(e, 0)) for a in triple]
        confs = [float(a["confidence"].get(e, 0.5)) for a in triple]
        s = sum(votes)
        if strategy == "any_of_3":
            out[e] = 1 if s >= 1 else 0
        elif strategy == "majority_2_of_3":
            out[e] = 1 if s >= 2 else 0
        elif strategy == "unanimous_3_of_3":
            out[e] = 1 if s == 3 else 0
        elif strategy == "confidence_avg":
            out[e] = 1 if (sum(confs) / len(confs)) >= 0.5 else 0
        elif strategy == "confidence_max":
            out[e] = 1 if max(confs) >= 0.5 else 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return out


def evaluate_language(jsonl_path: Path) -> dict:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not rows:
        return {}
    gold = [gold_to_dict(r["gold"]) for r in rows]
    triples = [[r["agent_1"], r["agent_2"], r["agent_3"]] for r in rows]
    out = {"n_samples": len(rows)}
    for s in STRATEGIES:
        preds = [aggregate(t, s) for t in triples]
        out[s] = compute_f1_scores(gold, preds)["macro_f1"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--topologies", nargs="+", default=TOPOLOGIES)
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    out_root = PROJECT_ROOT / "ablation" / "aggregation_strategy" / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    for topo in args.topologies:
        print(f"\n=== {topo} ({args.split}) ===")
        print(f"{'lang':6s} {'n':>5s} " + " ".join(f"{s:>16s}" for s in STRATEGIES))
        print("-" * (12 + 17 * len(STRATEGIES)))

        rows = []
        for lang in args.languages:
            jsonl = pred_dir / f"{topo}_{lang}_{args.split}.jsonl"
            if not jsonl.exists():
                print(f"  SKIP {lang}: {jsonl} not found")
                continue
            r = evaluate_language(jsonl)
            if not r:
                continue
            row = {"language": lang, "n_samples": r["n_samples"]}
            for s in STRATEGIES:
                row[s] = round(r[s], 2)
            rows.append(row)
            print(f"  {lang:5s} {r['n_samples']:>5d} " +
                  " ".join(f"{r[s]:>16.2f}" for s in STRATEGIES))

        if not rows:
            print("  no data, skipping")
            continue

        df = pd.DataFrame(rows)
        avg = {"language": "AVERAGE", "n_samples": int(df["n_samples"].mean())}
        for s in STRATEGIES:
            avg[s] = round(df[s].mean(), 2)
        df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
        out_path = out_root / f"{topo}_aggregation_{args.split}.csv"
        df.to_csv(out_path, index=False)
        print(f"  AVG: " + " ".join(f"{avg[s]:>16.2f}" for s in STRATEGIES))
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
