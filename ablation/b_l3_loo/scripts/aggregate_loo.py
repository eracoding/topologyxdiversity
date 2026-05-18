#!/usr/bin/env python3
"""Width Leave-One-Out re-aggregation (A2 — primary; A11 — generalised).

For each language and each LOO mask (drop one of the 3 agents), re-aggregates
the surviving 2 agents under TWO rules and writes a per-language CSV.

Rules (computed in parallel):
  * majority_threshold_2 — predict 1 iff >=2 agents predict 1.
      - For 3 agents (baseline): canonical B-L3 majority vote.
      - For 2 agents (LOO):      unanimity (both must predict 1) — conservative.
  * confidence_weighted    — predict 1 iff mean confidence across surviving agents >= 0.5.
      - With the stubbed 0.1/0.9 confidence values stored in the JSONLs:
          * 3 agents: equivalent to majority vote (>=2 of 3).
          * 2 agents: equivalent to "predict 1 iff EITHER agent predicts 1" (permissive).

Reporting both lets the reader see the full LOO sensitivity range:
  * permissive rule overestimates LOO performance (high recall floor).
  * conservative rule underestimates it (high precision floor).
The truth lies between the two.

Usage:
    python ablation/b_l3_loo/scripts/aggregate_loo.py --split dev
    python ablation/b_l3_loo/scripts/aggregate_loo.py --split test
    python ablation/b_l3_loo/scripts/aggregate_loo.py --split test --topology b_l1 \\
        --output ablation/width_loo_l1_l2/results/b_l1_loo_test.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.base_agent import AgentOutput
from src.aggregation.majority_vote import (
    majority_vote,
    confidence_weighted_average,
)
from src.evaluation.metrics import compute_f1_scores
from src.data.loader import EMOTIONS

LANGUAGES = ["pcm", "chn", "mar", "ary", "tat", "vmw", "ptmz", "zul", "ind"]

# Map agent positional index -> human-readable specialist name per topology.
# Order follows the canonical config (general/ambiguity/contrastive for B-L3).
AGENT_LABELS = {
    "b_l3": ["general", "ambiguity", "contrastive"],
    "b_l2": ["surface", "context", "structure"],
    "b_l1": ["agent_1", "agent_2", "agent_3"],
}


def gold_to_dict(gold) -> dict[str, int]:
    """Normalise gold to {emotion: 0/1}. Accepts dict (canonical) or list."""
    if isinstance(gold, dict):
        return {e: int(gold.get(e, 0)) for e in EMOTIONS}
    if isinstance(gold, list):
        present = set(gold)
        return {e: 1 if e in present else 0 for e in EMOTIONS}
    raise ValueError(f"Unknown gold format: {type(gold).__name__}")


def agent_dict_to_output(agent_dict: dict) -> AgentOutput:
    return AgentOutput(
        emotions={e: int(agent_dict["emotions"].get(e, 0)) for e in EMOTIONS},
        confidence={e: float(agent_dict["confidence"].get(e, 0.5)) for e in EMOTIONS},
        reasoning=agent_dict.get("reasoning", ""),
        raw_response="",
        parse_success=agent_dict.get("parse_success", True),
    )


def aggregate_majority(outputs: list[AgentOutput]) -> dict[str, int]:
    return majority_vote(outputs, threshold=2)


def aggregate_confidence(outputs: list[AgentOutput]) -> dict[str, int]:
    preds, _ = confidence_weighted_average(outputs, threshold=0.5)
    return preds


def evaluate_language(jsonl_path: Path) -> dict:
    """Compute baseline + 3 LOO Macro-F1 scores for one language under both rules."""
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not rows:
        return {}

    gold = [gold_to_dict(r["gold"]) for r in rows]

    # Build the 3 AgentOutputs per row
    triples = [
        [
            agent_dict_to_output(r["agent_1"]),
            agent_dict_to_output(r["agent_2"]),
            agent_dict_to_output(r["agent_3"]),
        ]
        for r in rows
    ]

    def score(preds):
        return compute_f1_scores(gold, preds)["macro_f1"]

    out = {"n_samples": len(rows)}

    # Baseline (all 3 agents) under both rules
    out["baseline_majority"] = score([aggregate_majority(t) for t in triples])
    out["baseline_confidence"] = score([aggregate_confidence(t) for t in triples])

    # LOO: drop each agent under both rules
    for drop in (0, 1, 2):
        keep = [i for i in range(3) if i != drop]
        kept = [[t[i] for i in keep] for t in triples]
        out[f"loo_drop{drop}_majority"] = score([aggregate_majority(k) for k in kept])
        out[f"loo_drop{drop}_confidence"] = score([aggregate_confidence(k) for k in kept])

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument(
        "--topology", default="b_l3", choices=["b_l1", "b_l2", "b_l3"]
    )
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument(
        "--output",
        default=None,
        help="CSV path. Default: ablation/<topology>_loo/results/<topology>_loo_<split>.csv",
    )
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    if args.output:
        output_path = Path(args.output)
    else:
        # default goes into the right ablation folder per topology
        sub = "b_l3_loo" if args.topology == "b_l3" else "width_loo_l1_l2"
        output_path = (
            PROJECT_ROOT / "ablation" / sub / "results"
            / f"{args.topology}_loo_{args.split}.csv"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = AGENT_LABELS.get(args.topology, ["agent_1", "agent_2", "agent_3"])

    rows = []
    print(f"\nWidth LOO — topology={args.topology}, split={args.split}")
    print(f"Predictions dir: {pred_dir}")
    print(f"Output:          {output_path}\n")
    print(
        f"{'lang':6s} {'n':>5s} {'base_maj':>9s} "
        f"{'-' + labels[0][:6]:>10s} {'-' + labels[1][:6]:>10s} {'-' + labels[2][:6]:>10s}  | "
        f"{'base_cw':>9s} {'-' + labels[0][:6]:>10s} {'-' + labels[1][:6]:>10s} {'-' + labels[2][:6]:>10s}"
    )
    print("-" * 130)

    for lang in args.languages:
        jsonl = pred_dir / f"{args.topology}_{lang}_{args.split}.jsonl"
        if not jsonl.exists():
            print(f"  SKIP {lang}: {jsonl} not found")
            continue
        r = evaluate_language(jsonl)
        if not r:
            print(f"  SKIP {lang}: empty file")
            continue

        row = {
            "language": lang,
            "n_samples": r["n_samples"],
            "baseline_majority": r["baseline_majority"],
            f"loo_drop_{labels[0]}_majority": r["loo_drop0_majority"],
            f"loo_drop_{labels[1]}_majority": r["loo_drop1_majority"],
            f"loo_drop_{labels[2]}_majority": r["loo_drop2_majority"],
            f"delta_drop_{labels[0]}_majority": round(
                r["loo_drop0_majority"] - r["baseline_majority"], 2
            ),
            f"delta_drop_{labels[1]}_majority": round(
                r["loo_drop1_majority"] - r["baseline_majority"], 2
            ),
            f"delta_drop_{labels[2]}_majority": round(
                r["loo_drop2_majority"] - r["baseline_majority"], 2
            ),
            "baseline_confidence": r["baseline_confidence"],
            f"loo_drop_{labels[0]}_confidence": r["loo_drop0_confidence"],
            f"loo_drop_{labels[1]}_confidence": r["loo_drop1_confidence"],
            f"loo_drop_{labels[2]}_confidence": r["loo_drop2_confidence"],
            f"delta_drop_{labels[0]}_confidence": round(
                r["loo_drop0_confidence"] - r["baseline_confidence"], 2
            ),
            f"delta_drop_{labels[1]}_confidence": round(
                r["loo_drop1_confidence"] - r["baseline_confidence"], 2
            ),
            f"delta_drop_{labels[2]}_confidence": round(
                r["loo_drop2_confidence"] - r["baseline_confidence"], 2
            ),
        }
        rows.append(row)
        print(
            f"  {lang:5s} {r['n_samples']:>5d} {r['baseline_majority']:>9.2f} "
            f"{r['loo_drop0_majority']:>10.2f} {r['loo_drop1_majority']:>10.2f} {r['loo_drop2_majority']:>10.2f}  | "
            f"{r['baseline_confidence']:>9.2f} "
            f"{r['loo_drop0_confidence']:>10.2f} {r['loo_drop1_confidence']:>10.2f} {r['loo_drop2_confidence']:>10.2f}"
        )

    if not rows:
        print("\nNo data — exiting.")
        return

    df = pd.DataFrame(rows)
    avg_row = {"language": "AVERAGE"}
    for col in df.columns:
        if col == "language":
            continue
        if col == "n_samples":
            avg_row[col] = int(df[col].mean())
        else:
            avg_row[col] = round(df[col].mean(), 2)
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 130)
    print("AVERAGES across languages:")
    print(f"  Majority-vote rule:")
    print(f"    baseline (3 agents)         = {avg_row['baseline_majority']:.2f}")
    print(f"    drop {labels[0]:12s} = {avg_row[f'loo_drop_{labels[0]}_majority']:.2f}  (Δ = {avg_row[f'delta_drop_{labels[0]}_majority']:+.2f})")
    print(f"    drop {labels[1]:12s} = {avg_row[f'loo_drop_{labels[1]}_majority']:.2f}  (Δ = {avg_row[f'delta_drop_{labels[1]}_majority']:+.2f})")
    print(f"    drop {labels[2]:12s} = {avg_row[f'loo_drop_{labels[2]}_majority']:.2f}  (Δ = {avg_row[f'delta_drop_{labels[2]}_majority']:+.2f})")
    print(f"  Confidence-weighted rule:")
    print(f"    baseline (3 agents)         = {avg_row['baseline_confidence']:.2f}")
    print(f"    drop {labels[0]:12s} = {avg_row[f'loo_drop_{labels[0]}_confidence']:.2f}  (Δ = {avg_row[f'delta_drop_{labels[0]}_confidence']:+.2f})")
    print(f"    drop {labels[1]:12s} = {avg_row[f'loo_drop_{labels[1]}_confidence']:.2f}  (Δ = {avg_row[f'delta_drop_{labels[1]}_confidence']:+.2f})")
    print(f"    drop {labels[2]:12s} = {avg_row[f'loo_drop_{labels[2]}_confidence']:.2f}  (Δ = {avg_row[f'delta_drop_{labels[2]}_confidence']:+.2f})")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
