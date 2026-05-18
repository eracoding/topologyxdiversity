"""End-to-end experiment runner for Config B (Width Topology).

Loads data for all 9 languages, runs width topology on dev set (or test),
evaluates, and saves predictions as JSONL and results as CSV.
"""

import argparse
import json
import sys
from pathlib import Path

import jsonlines
import pandas as pd
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_split, get_texts_and_labels, EMOTIONS
from src.topologies.width import build_width_topology
from src.evaluation.metrics import compute_f1_scores, compute_per_language_results
from src.utils.cost_tracker import CostTracker


def load_configs(base_path: str, width_path: str) -> tuple[dict, dict]:
    with open(base_path) as f:
        base = yaml.safe_load(f)
    with open(width_path) as f:
        width = yaml.safe_load(f)
    return base, width


def run_language(
    topology,
    texts: list[str],
    labels: list[dict[str, int]],
    language: str,
) -> tuple[list[dict], dict]:
    """Run topology on all texts for a language and evaluate.

    Returns (predictions_list, metrics_dict)
    """
    predictions = []
    pred_labels = []

    for i, text in enumerate(tqdm(texts, desc=f"  {language}", leave=False)):
        result = topology.predict_single(text)
        pred_labels.append(result["emotions"])

        # Build prediction record for JSONL output
        record = {
            "language": language,
            "index": i,
            "text": text,
            "true_labels": labels[i],
            "pred_labels": result["emotions"],
            "confidence": result["confidence"],
            "parse_failures": result["parse_failures"],
            "agent_predictions": [
                {
                    "agent": topology.agents[j].name,
                    "emotions": o.emotions,
                    "confidence": o.confidence,
                    "reasoning": o.reasoning,
                    "parse_success": o.parse_success,
                }
                for j, o in enumerate(result["agent_outputs"])
            ],
        }
        predictions.append(record)

    metrics = compute_f1_scores(labels, pred_labels)
    return predictions, metrics


def main():
    parser = argparse.ArgumentParser(description="Run Config B (Width Topology)")
    parser.add_argument(
        "--split", default="dev", choices=["dev", "test"],
        help="Which split to evaluate on (default: dev)",
    )
    parser.add_argument(
        "--languages", nargs="+", default=None,
        help="Languages to evaluate (default: all 9)",
    )
    parser.add_argument(
        "--base-config", default="config/base.yaml",
        help="Path to base config",
    )
    parser.add_argument(
        "--width-config", default="config/config_b_width.yaml",
        help="Path to width config",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Output directory",
    )
    args = parser.parse_args()

    # Load configs
    base_cfg, width_cfg = load_configs(args.base_config, args.width_config)

    languages = args.languages or base_cfg["data"]["languages"]
    data_path = base_cfg["data"]["base_path"]
    pred_dir = Path(args.output_dir) / "predictions"
    results_dir = Path(args.output_dir) / "results"
    pred_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set up cost tracker and topology
    cost_tracker = CostTracker()
    cost_tracker.start_timer()
    topology = build_width_topology(width_cfg, base_cfg, cost_tracker)

    print(f"Config B (Width) — split={args.split}, languages={languages}")
    print(f"Agents: {[a.name for a in topology.agents]}")
    print(f"Aggregation: {topology.aggregation}")
    print()

    # Run per language
    all_results = {}
    all_predictions = []

    for lang in languages:
        print(f"Processing {lang}...")
        try:
            df = load_split(data_path, lang, args.split)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        texts, labels = get_texts_and_labels(df)
        predictions, metrics = run_language(topology, texts, labels, lang)
        all_results[lang] = metrics
        all_predictions.extend(predictions)

        print(f"  Macro-F1: {metrics['macro_f1']:.2f}%  Micro-F1: {metrics['micro_f1']:.2f}%  (n={metrics['n_samples']})")

    # Summary
    if all_results:
        summary = compute_per_language_results(all_results)
        print(f"\nAverage Macro-F1: {summary['avg_macro_f1']:.2f}%")
        print(f"Average Micro-F1: {summary['avg_micro_f1']:.2f}%")

        # Save predictions as JSONL
        pred_file = pred_dir / f"config_b_{args.split}.jsonl"
        with jsonlines.open(pred_file, mode="w") as writer:
            for record in all_predictions:
                writer.write(record)
        print(f"\nPredictions saved to {pred_file}")

        # Save results as CSV
        rows = []
        for lang, metrics in all_results.items():
            row = {"language": lang, "macro_f1": metrics["macro_f1"], "micro_f1": metrics["micro_f1"], "n_samples": metrics["n_samples"]}
            for emo, stats in metrics["per_emotion"].items():
                row[f"{emo}_f1"] = round(stats["f1"] * 100, 2)
            rows.append(row)

        results_df = pd.DataFrame(rows)
        # Add average row
        avg_row = {"language": "AVERAGE"}
        for col in results_df.columns:
            if col != "language":
                avg_row[col] = round(results_df[col].mean(), 2)
        results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)

        results_file = results_dir / f"config_b_{args.split}_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

        # Save cost summary
        cost = cost_tracker.summary()
        print(f"\nCost summary:")
        print(f"  Requests: {cost['num_requests']}")
        print(f"  Total tokens: {cost['total_tokens']}")
        print(f"  Wall clock: {cost['wall_clock_seconds']}s")

        cost_file = results_dir / f"config_b_{args.split}_cost.json"
        with open(cost_file, "w") as f:
            json.dump(cost, f, indent=2)

    else:
        print("No results — check that data files exist in the expected paths.")


if __name__ == "__main__":
    main()
