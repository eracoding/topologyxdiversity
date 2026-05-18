"""Pairwise disagreement metric for Config B (Width Topology)"""

import json
from itertools import combinations
from pathlib import Path

from src.data.loader import EMOTIONS, LANGUAGES


def pairwise_disagreement_rate(agent_outputs: list[dict]) -> float:
    """Fraction of (agent_i, agent_j) pairs that disagree on >=1 emotion label"""
    if len(agent_outputs) < 2:
        return 0.0

    pairs = list(combinations(agent_outputs, 2))
    disagreements = 0
    for a, b in pairs:
        emotions_a = a.get("emotions", {})
        emotions_b = b.get("emotions", {})
        all_keys = set(emotions_a) | set(emotions_b)
        if any(emotions_a.get(k, 0) != emotions_b.get(k, 0) for k in all_keys):
            disagreements += 1

    return disagreements / len(pairs)


def per_emotion_disagreement(agent_outputs: list[dict]) -> dict[str, float]:
    """Per-emotion disagreement rate across all agent pairs"""
    if len(agent_outputs) < 2:
        return {emo: 0.0 for emo in EMOTIONS}

    pairs = list(combinations(agent_outputs, 2))
    counts = {emo: 0 for emo in EMOTIONS}
    for a, b in pairs:
        emotions_a = a.get("emotions", {})
        emotions_b = b.get("emotions", {})
        for emo in EMOTIONS:
            if emotions_a.get(emo, 0) != emotions_b.get(emo, 0):
                counts[emo] += 1

    return {emo: counts[emo] / len(pairs) for emo in EMOTIONS}


def compute_diversity_report(predictions_jsonl_path: str) -> dict:
    """Compute pairwise disagreement statistics from a run_config_b JSONL output"""
    records = []
    with open(predictions_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"error": "No records found in JSONL file."}

    # Per-instance pairwise disagreement
    instance_disagree = []
    for rec in records:
        agent_outputs = rec.get("agent_predictions", [])
        instance_disagree.append(pairwise_disagreement_rate(agent_outputs))

    # Group records by language
    by_language: dict[str, list[dict]] = {}
    for rec in records:
        lang = rec.get("language", "unknown")
        by_language.setdefault(lang, []).append(rec)

    per_language: dict[str, float] = {}
    per_language_per_emotion: dict[str, dict[str, float]] = {}
    for lang, lang_records in by_language.items():
        lang_disagree = [
            pairwise_disagreement_rate(r.get("agent_predictions", []))
            for r in lang_records
        ]
        per_language[lang] = round(sum(lang_disagree) / len(lang_disagree), 4)

        # Per-emotion for this language
        emo_counts = {emo: 0.0 for emo in EMOTIONS}
        for rec in lang_records:
            emo_d = per_emotion_disagreement(rec.get("agent_predictions", []))
            for emo in EMOTIONS:
                emo_counts[emo] += emo_d[emo]
        per_language_per_emotion[lang] = {
            emo: round(emo_counts[emo] / len(lang_records), 4)
            for emo in EMOTIONS
        }

    # Global per-emotion: average across all instances
    global_emo_counts = {emo: 0.0 for emo in EMOTIONS}
    for rec in records:
        emo_d = per_emotion_disagreement(rec.get("agent_predictions", []))
        for emo in EMOTIONS:
            global_emo_counts[emo] += emo_d[emo]
    per_emotion_global = {
        emo: round(global_emo_counts[emo] / len(records), 4)
        for emo in EMOTIONS
    }

    return {
        "overall_disagreement_rate": round(sum(instance_disagree) / len(instance_disagree), 4),
        "per_language": per_language,
        "per_emotion": per_emotion_global,
        "per_language_per_emotion": per_language_per_emotion,
        "n_instances": len(records),
    }


def print_diversity_report(report: dict) -> None:
    """Pretty-print a diversity report to stdout."""
    if "error" in report:
        print(f"Error: {report['error']}")
        return

    print(f"Pairwise Disagreement Report  (n={report['n_instances']})")
    print("=" * 52)
    print(f"Overall disagreement rate: {report['overall_disagreement_rate']:.4f}")

    print("\nPer-language disagreement rate:")
    for lang in sorted(report["per_language"]):
        print(f"  {lang:6s}  {report['per_language'][lang]:.4f}")

    print("\nPer-emotion disagreement rate (global):")
    for emo in EMOTIONS:
        print(f"  {emo:10s}  {report['per_emotion'][emo]:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute diversity metrics for Config B predictions.")
    parser.add_argument("jsonl", help="Path to config_b_{split}.jsonl")
    parser.add_argument("--save", default=None, help="Optional path to save JSON report")
    args = parser.parse_args()

    report = compute_diversity_report(args.jsonl)
    print_diversity_report(report)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.save}")
