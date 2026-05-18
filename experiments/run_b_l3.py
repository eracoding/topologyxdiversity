"""B-L3: Width topology with learned diversity (QLoRA adapters).

3 agents each using a different QLoRA adapter on frozen Qwen2.5-14B:
  - general_specialist:     trained on all data, standard BCE loss
  - ambiguity_specialist:   trained on multi-label + hard instances, 2x weight
  - contrastive_specialist: trained on all data, BCE + contrastive loss

Adapters must be trained before running this script:
    python training/train_b_adapters.py --adapter all

See EXPERIMENT_SPEC.md Section 3.3.

Usage:
    python experiments/run_b_l3.py [--split dev|test] [--languages pcm ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_split, get_texts_and_labels
from src.topologies.width import build_lora_width_topology
from src.utils.cost_tracker import CostTracker
from experiments.runner_utils import (
    run_language_with_topology,
    run_language_with_topology_batch,
    load_results_from_jsonl,
    save_predictions_jsonl,
    build_results_csv,
)


def check_adapters_exist(config: dict) -> list[str]:
    """Return list of adapters that have not been trained yet."""
    missing = []
    for agent_def in config["agents"]:
        path = Path(agent_def["lora_path"])
        # Check for adapter_config.json which PEFT saves on completion
        if not (path / "adapter_config.json").exists():
            missing.append(f"{agent_def['name']} → {agent_def['lora_path']}")
    return missing


def main():
    parser = argparse.ArgumentParser(description="Run B-L3 (Width, Learned Diversity)")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--config", default="config/config_b_l3.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch", action="store_true", help="Use vLLM native batched chat (faster; cannot coexist with HTTP server)")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.config) as f:
        exp_cfg = yaml.safe_load(f)

    # Guard: verify adapters are trained
    missing = check_adapters_exist(exp_cfg)
    if missing:
        print("ERROR: The following adapters have not been trained yet:")
        for m in missing:
            print(f"  {m}")
        print("\nTrain them first:")
        print("  python training/train_b_adapters.py --adapter all")
        sys.exit(1)

    languages = args.languages or base_cfg["data"]["languages"]
    data_path = base_cfg["data"]["base_path"]
    pred_dir = Path(args.output_dir) / "predictions"
    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cost_tracker = CostTracker()
    cost_tracker.start_timer()
    topology = build_lora_width_topology(exp_cfg, base_cfg, cost_tracker)

    print(f"B-L3 (Width, Learned Diversity) — split={args.split}, languages={languages}")
    print(f"Agents: {[a.name for a in topology.agents]}")
    print(f"Adapters: {[a.lora_request.lora_name for a in topology.agents]}")
    print(f"Aggregation: {topology.aggregation}")
    print()

    all_results = {}
    all_agent_results = {}

    agent_names = [a.name for a in topology.agents]

    for lang in languages:
        pred_file = pred_dir / f"b_l3_{lang}_{args.split}.jsonl"

        # Resume: reuse existing predictions instead of re-running inference
        cached = load_results_from_jsonl(pred_file, agent_names)
        if cached is not None:
            metrics, agent_metrics = cached
            all_results[lang] = metrics
            all_agent_results[lang] = agent_metrics
            print(f"  {lang}: loaded from checkpoint — Macro-F1: {metrics['macro_f1']:.2f}%")
            continue

        print(f"Processing {lang}...")
        try:
            df = load_split(data_path, lang, args.split)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        texts, labels, ids = get_texts_and_labels(df, include_ids=True)
        run_fn = run_language_with_topology_batch if args.batch else run_language_with_topology
        records, metrics, agent_metrics = run_fn(topology, texts, labels, ids, lang)
        all_results[lang] = metrics
        all_agent_results[lang] = agent_metrics

        save_predictions_jsonl(records, pred_file)

        print(f"  Macro-F1: {metrics['macro_f1']:.2f}%  Micro-F1: {metrics['micro_f1']:.2f}%  (n={metrics['n_samples']})")
        for agent_name, am in agent_metrics.items():
            print(f"    [{agent_name}] Macro-F1: {am['macro_f1']:.2f}%")

        # Incremental checkpoint: write CSV after every language so a crash loses at most one language
        results_df = build_results_csv(all_results, all_agent_results)
        results_df.to_csv(results_dir / f"b_l3_{args.split}_results.csv", index=False)

    if not all_results:
        print("No results — check data files.")
        return

    macro_f1s = [m["macro_f1"] for m in all_results.values()]
    print(f"\nAverage Macro-F1: {np.mean(macro_f1s):.2f}%")

    results_df = build_results_csv(all_results, all_agent_results)
    results_file = results_dir / f"b_l3_{args.split}_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    cost = cost_tracker.summary()
    cost_file = results_dir / f"b_l3_{args.split}_cost.json"
    with open(cost_file, "w") as f:
        json.dump(cost, f, indent=2)
    print(f"Cost summary: {cost['num_requests']} requests, {cost['total_tokens']} tokens, {cost['wall_clock_seconds']:.0f}s")


if __name__ == "__main__":
    main()
