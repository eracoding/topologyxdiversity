"""C-L3: Depth topology with learned diversity (sequential QLoRA adapters).

3 layers, each using a QLoRA adapter trained for its position in the pipeline:
  - Layer 1 (initial_classifier):  text only → standard classification
  - Layer 2 (error_corrector):     text + Layer 1 output → corrects errors
  - Layer 3 (calibrator):          text + Layer 2 output → adjudicates and calibrates

Unlike B-L3 (parallel), adapters here are sequentially conditioned: Layer N was
trained on inputs that include Layer N-1's predictions, so it has learned to
read and improve on prior analysis rather than simply classify from scratch.

Adapters must be trained in order before running this script:
    python training/train_c_adapters.py --phase train1
    python training/train_c_adapters.py --phase collect1
    python training/train_c_adapters.py --phase train2
    python training/train_c_adapters.py --phase collect2
    python training/train_c_adapters.py --phase train3

See EXPERIMENT_SPEC.md Section 4.3.

Usage:
    python experiments/run_c_l3.py [--split dev|test] [--languages pcm ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_split, get_texts_and_labels
from src.topologies.depth import build_lora_depth_topology
from src.utils.cost_tracker import CostTracker
from src.evaluation.layer_analysis import analyse_layer_transitions, format_layer_analysis_report
from experiments.runner_utils import (
    run_language_with_depth_topology,
    run_language_with_depth_topology_batch,
    load_depth_results_from_jsonl,
    save_predictions_jsonl,
    build_results_csv,
)


def check_adapters_exist(config: dict) -> list[str]:
    """Return list of adapters that have not been trained yet."""
    missing = []
    for layer_def in config["layers"]:
        path = Path(layer_def["lora_path"])
        if not (path / "adapter_config.json").exists():
            missing.append(f"{layer_def['name']} → {layer_def['lora_path']}")
    return missing


def main():
    parser = argparse.ArgumentParser(description="Run C-L3 (Depth, Learned Diversity)")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--config", default="config/config_c_l3.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch", action="store_true", help="Use vLLM native batched chat (faster; cannot coexist with HTTP server)")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.config) as f:
        exp_cfg = yaml.safe_load(f)

    # Guard: verify all adapters are trained
    missing = check_adapters_exist(exp_cfg)
    if missing:
        print("ERROR: The following adapters have not been trained yet:")
        for m in missing:
            print(f"  {m}")
        print("\nTrain them first (in order):")
        print("  python training/train_c_adapters.py --phase train1")
        print("  python training/train_c_adapters.py --phase collect1")
        print("  python training/train_c_adapters.py --phase train2")
        print("  python training/train_c_adapters.py --phase collect2")
        print("  python training/train_c_adapters.py --phase train3")
        sys.exit(1)

    languages = args.languages or base_cfg["data"]["languages"]
    data_path = base_cfg["data"]["base_path"]
    pred_dir = Path(args.output_dir) / "predictions"
    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cost_tracker = CostTracker()
    cost_tracker.start_timer()
    topology = build_lora_depth_topology(exp_cfg, base_cfg, cost_tracker)

    layer_names = [layer.name for layer in topology.layers]
    layer_roles = [layer.role for layer in topology.layers]
    print(f"C-L3 (Depth, Learned) — split={args.split}, languages={languages}")
    print(f"Layers: {list(zip(layer_names, layer_roles))}")
    print(f"Adapters: {[layer.lora_request.lora_name for layer in topology.layers]}")
    print(f"Temperatures: {[layer.sampling_params.temperature for layer in topology.layers]}")
    print()

    all_results = {}
    all_layer_results = {}

    for lang in languages:
        pred_file = pred_dir / f"c_l3_{lang}_{args.split}.jsonl"

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

        analysis_file = results_dir / f"c_l3_{lang}_{args.split}_layer_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)

        # Incremental checkpoint: write CSV after every language
        results_df = build_results_csv(all_results, all_layer_results)
        results_df.to_csv(results_dir / f"c_l3_{args.split}_results.csv", index=False)

    if not all_results:
        print("No results — check data files.")
        return

    macro_f1s = [m["macro_f1"] for m in all_results.values()]
    print(f"\nAverage Macro-F1: {np.mean(macro_f1s):.2f}%")

    results_df = build_results_csv(all_results, all_layer_results)
    results_file = results_dir / f"c_l3_{args.split}_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    cost = cost_tracker.summary()
    cost_file = results_dir / f"c_l3_{args.split}_cost.json"
    with open(cost_file, "w") as f:
        json.dump(cost, f, indent=2)
    print(f"Cost summary: {cost['num_requests']} requests, {cost['total_tokens']} tokens, {cost['wall_clock_seconds']:.0f}s")


if __name__ == "__main__":
    main()
