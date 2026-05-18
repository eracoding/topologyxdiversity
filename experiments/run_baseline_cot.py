"""Single-pass CoT baseline for pipeline verification.

MUST be run first on pcm before any multi-agent experiments.
Expected result: 49.48% Macro-F1 ± 1 pp (thesis CoT baseline for pcm).

If this does not match, debug before proceeding.
See EXPERIMENT_SPEC.md Section 5, Step 1.

Usage:
    python experiments/run_baseline_cot.py [--split dev|test] [--languages pcm]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm import LLM, SamplingParams

from src.data.loader import load_split, get_texts_and_labels, EMOTIONS
from src.prompts.task_template import format_user_prompt
from src.prompts.cot_enhancement import build_cot_block
from src.utils.json_parser import parse_agent_output
from src.utils.cost_tracker import CostTracker
from src.evaluation.metrics import compute_f1_scores
from experiments.runner_utils import save_predictions_jsonl


def run_single_pass_cot(
    llm: LLM,
    sampling_params: SamplingParams,
    texts: list[str],
    labels: list[dict],
    ids: list[str],
    language: str,
    reasoning_banks_path: str | None,
    cost_tracker: CostTracker,
) -> tuple[list[dict], dict]:
    """Single-pass CoT on all texts for one language."""
    from tqdm import tqdm

    cot_block = build_cot_block(language, reasoning_banks_path)
    user_template = format_user_prompt

    records = []
    preds = []

    for text, gold, text_id in tqdm(zip(texts, labels, ids), total=len(texts), desc=f"  {language}", leave=False):
        import time
        user_message = user_template(text)
        start = time.time()
        outputs = llm.chat(
            messages=[
                {"role": "system", "content": cot_block},
                {"role": "user", "content": user_message},
            ],
            sampling_params=sampling_params,
        )
        latency = time.time() - start

        output = outputs[0]
        raw = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        cost_tracker.record_request(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency=latency,
        )

        parsed = parse_agent_output(raw)
        if parsed is None:
            emotions = {e: 0 for e in EMOTIONS}
            confidence = {e: 0.5 for e in EMOTIONS}
            reasoning = "[PARSE FAILURE]"
            parse_success = False
        else:
            emotions = parsed["emotions"]
            confidence = parsed["confidence"]
            reasoning = parsed["reasoning"]
            parse_success = True

        preds.append(emotions)
        records.append({
            "text_id": text_id,
            "text": text,
            "gold": gold,
            "agent_1": {
                "emotions": emotions,
                "confidence": confidence,
                "reasoning": reasoning,
                "parse_success": parse_success,
            },
            "aggregated": {
                "emotions": emotions,
                "method": "single_pass",
            },
        })

    metrics = compute_f1_scores(labels, preds)
    return records, metrics


def main():
    parser = argparse.ArgumentParser(description="Run single-pass CoT baseline")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--languages", nargs="+", default=["pcm"],
                        help="Languages to evaluate (default: pcm for pipeline check)")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)

    model_cfg = base_cfg["model"]
    reasoning_banks_path = base_cfg.get("data", {}).get("reasoning_banks_path")
    data_path = base_cfg["data"]["base_path"]
    results_dir = Path(args.output_dir) / "results"
    pred_dir = Path(args.output_dir) / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=model_cfg["name"],
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.90),
        dtype=model_cfg.get("dtype", "bfloat16"),
        seed=model_cfg.get("seed", 42),
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=model_cfg.get("max_tokens", 512),
        seed=42,
    )

    cost_tracker = CostTracker()
    cost_tracker.start_timer()

    print(f"Single-pass CoT baseline — split={args.split}, languages={args.languages}")
    print("Expected pcm Macro-F1: ~49.48% (thesis baseline). ±1pp is acceptable.")
    print()

    all_results = {}
    for lang in args.languages:
        print(f"Processing {lang}...")
        try:
            df = load_split(data_path, lang, args.split)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        texts, labels, ids = get_texts_and_labels(df, include_ids=True)
        records, metrics = run_single_pass_cot(
            llm, sampling_params, texts, labels, ids, lang,
            reasoning_banks_path, cost_tracker,
        )
        all_results[lang] = metrics

        pred_file = pred_dir / f"baseline_cot_{lang}_{args.split}.jsonl"
        save_predictions_jsonl(records, pred_file)
        print(f"  Macro-F1: {metrics['macro_f1']:.2f}%  Micro-F1: {metrics['micro_f1']:.2f}%  (n={metrics['n_samples']})")

        if lang == "pcm":
            diff = abs(metrics["macro_f1"] - 49.48)
            if diff <= 1.0:
                print(f"  PASS: within ±1pp of thesis baseline (49.48%)")
            else:
                print(f"  WARNING: {diff:.2f}pp away from thesis baseline (49.48%). Debug before running multi-agent experiments.")

    if all_results:
        macro_f1s = [m["macro_f1"] for m in all_results.values()]
        print(f"\nAverage Macro-F1: {np.mean(macro_f1s):.2f}%")

        cost = cost_tracker.summary()
        cost_file = results_dir / f"baseline_cot_{args.split}_cost.json"
        with open(cost_file, "w") as f:
            json.dump(cost, f, indent=2)


if __name__ == "__main__":
    main()
