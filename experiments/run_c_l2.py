"""C-L2: Depth topology with designed diversity.

3 layers with designed roles:
  - Layer 1 (Analyst):    Standard CoT classification
  - Layer 2 (Critic):     Finds and corrects errors in Layer 1's output
  - Layer 3 (Calibrator): Adjudicates, removes weak predictions, confirms strong ones

Final output is Layer 3's prediction.

See EXPERIMENT_SPEC.md Section 4.2.

Usage:
    python experiments/run_c_l2.py [--split dev|test] [--languages pcm ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_split, get_texts_and_labels
from src.topologies.depth import build_depth_topology
from src.utils.cost_tracker import CostTracker
from src.evaluation.layer_analysis import analyse_layer_transitions, format_layer_analysis_report
from experiments.runner_utils import (
    run_language_with_depth_topology,
    run_language_with_depth_topology_batch,
    load_depth_results_from_jsonl,
    save_predictions_jsonl,
    build_results_csv,
)


def main():
    parser = argparse.ArgumentParser(description="Run C-L2 (Depth, Designed Diversity)")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--config", default="config/config_c_l2.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch", action="store_true", help="Use vLLM native batched chat (faster; cannot coexist with HTTP server)")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.config) as f:
        exp_cfg = yaml.safe_load(f)

    languages = args.languages or base_cfg["data"]["languages"]
    data_path = base_cfg["data"]["base_path"]
    pred_dir = Path(args.output_dir) / "predictions"
    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cost_tracker = CostTracker()
    cost_tracker.start_timer()
    topology = build_depth_topology(exp_cfg, base_cfg, cost_tracker)

    layer_names = [layer.name for layer in topology.layers]
    layer_roles = [layer.role for layer in topology.layers]
    print(f"C-L2 (Depth, Designed) — split={args.split}, languages={languages}")
    print(f"Layers: {list(zip(layer_names, layer_roles))}")
    print(f"Temperatures: {[layer.sampling_params.temperature for layer in topology.layers]}")
    print()

    all_results = {}
    all_layer_results = {}

    for lang in languages:
        pred_file = pred_dir / f"c_l2_{lang}_{args.split}.jsonl"

        # Resume: reuse existing predictions instead of re-running inference
        cached = load_depth_results_from_jsonl(pred_file, layer_names)
        if cached is not None:
            metrics, layer_metrics = cached
            all_results[lang] = metrics
            all_layer_results[lang] = layer_metrics
            print(f"  {lang}: loaded from checkpoint — Macro-F1: {metrics['macro_f1']:.2f}%")
            continue

        print(f"Processing {lang}...")
        try:
            df = load_split(data_path, lang, args.split)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        texts, labels, ids = get_texts_and_labels(df, include_ids=True)
        run_fn = run_language_with_depth_topology_batch if args.batch else run_language_with_depth_topology
        records, metrics, layer_metrics, all_layer_outputs = run_fn(
            topology, texts, labels, ids, lang
        )
        all_results[lang] = metrics
        all_layer_results[lang] = layer_metrics

        save_predictions_jsonl(records, pred_file)

        print(f"  Macro-F1: {metrics['macro_f1']:.2f}%  Micro-F1: {metrics['micro_f1']:.2f}%  (n={metrics['n_samples']})")
        for layer_name, lm in layer_metrics.items():
            print(f"    [{layer_name}] Macro-F1: {lm['macro_f1']:.2f}%")

        # Layer transition analysis
        analysis = analyse_layer_transitions(all_layer_outputs, labels)
        print(f"  {format_layer_analysis_report(analysis)}")

        # Save layer analysis JSON
        analysis_file = results_dir / f"c_l2_{lang}_{args.split}_layer_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)

        # Incremental checkpoint: write CSV after every language
        results_df = build_results_csv(all_results, all_layer_results)
        results_df.to_csv(results_dir / f"c_l2_{args.split}_results.csv", index=False)

    if not all_results:
        print("No results — check data files.")
        return

    macro_f1s = [m["macro_f1"] for m in all_results.values()]
    print(f"\nAverage Macro-F1: {np.mean(macro_f1s):.2f}%")

    # Final CSV save
    results_df = build_results_csv(all_results, all_layer_results)
    results_file = results_dir / f"c_l2_{args.split}_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    cost = cost_tracker.summary()
    cost_file = results_dir / f"c_l2_{args.split}_cost.json"
    with open(cost_file, "w") as f:
        json.dump(cost, f, indent=2)
    print(f"Cost summary: {cost['num_requests']} requests, {cost['total_tokens']} tokens, {cost['wall_clock_seconds']:.0f}s")


if __name__ == "__main__":
    main()
