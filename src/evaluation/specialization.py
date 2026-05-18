"""Emergent specialization analysis for Config B (Width Topology)"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.data.loader import EMOTIONS, LANGUAGES


def _f1_scores_for_agent(
    true_labels: list[dict[str, int]],
    agent_preds: list[dict[str, int]],
    emotions: list[str] | None = None,
) -> dict:
    """Compute Macro-F1, per-emotion F1 for one agent against gold labels."""
    if emotions is None:
        emotions = EMOTIONS

    true_arr = np.array([[d.get(e, 0) for e in emotions] for d in true_labels])
    pred_arr = np.array([[d.get(e, 0) for e in emotions] for d in agent_preds])

    macro_f1 = float(f1_score(true_arr, pred_arr, average="macro", zero_division=0))
    per_emotion = {
        emo: round(float(f1_score(true_arr[:, i], pred_arr[:, i], zero_division=0)), 4)
        for i, emo in enumerate(emotions)
    }
    return {
        "macro_f1": round(macro_f1 * 100, 2),
        "per_emotion_f1": per_emotion,
        "n_samples": len(true_labels),
    }


def per_agent_f1(predictions_jsonl_path: str) -> dict:
    """Compute per-agent F1 scores from a run_config_b JSONL output

    Returns:
        Nested dict keyed by agent name:
        {
            "<agent_name>": {
                "macro_f1": float,           # overall, in %
                "per_emotion_f1": {...},      # emotion -> F1 in [0,1]
                "per_language_macro_f1": {lang: float},  # in %
                "n_samples": int,
            },
            ...
        }
    """
    records = []
    with open(predictions_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {}

    # agent names from the first record that has agent_predictions
    agent_names: list[str] = []
    for rec in records:
        aps = rec.get("agent_predictions", [])
        if aps:
            agent_names = [ap["agent"] for ap in aps]
            break

    if not agent_names:
        return {}

    # per-agent, per-language collections
    # agent -> language -> (true_labels, agent_preds)
    data: dict[str, dict[str, tuple[list, list]]] = {
        name: {} for name in agent_names
    }

    for rec in records:
        lang = rec.get("language", "unknown")
        true = rec.get("true_labels", {})
        for ap in rec.get("agent_predictions", []):
            name = ap["agent"]
            if name not in data:
                continue
            if lang not in data[name]:
                data[name][lang] = ([], [])
            data[name][lang][0].append(true)
            data[name][lang][1].append(ap.get("emotions", {}))

    results: dict[str, dict] = {}
    for name in agent_names:
        # all languages for overall scores
        all_true: list[dict] = []
        all_pred: list[dict] = []
        per_lang: dict[str, float] = {}

        for lang, (trues, preds) in data[name].items():
            all_true.extend(trues)
            all_pred.extend(preds)
            lang_metrics = _f1_scores_for_agent(trues, preds)
            per_lang[lang] = lang_metrics["macro_f1"]

        overall = _f1_scores_for_agent(all_true, all_pred)
        results[name] = {
            "macro_f1": overall["macro_f1"],
            "per_emotion_f1": overall["per_emotion_f1"],
            "per_language_macro_f1": per_lang,
            "n_samples": overall["n_samples"],
        }

    return results


def specialization_summary(per_agent_f1_dict: dict) -> str:
    """Build a text table showing which agent leads on which emotion/language """
    if not per_agent_f1_dict:
        return "No data."

    agent_names = list(per_agent_f1_dict.keys())
    lines: list[str] = []

    # header
    lines.append("Per-Agent F1 Summary — Emergent Specialization")
    lines.append("=" * 60)

    #  Macro-F1
    lines.append("\nOverall Macro-F1 (%):")
    for name in agent_names:
        lines.append(f"  {name:35s}  {per_agent_f1_dict[name]['macro_f1']:6.2f}")

    # per-emotion: show each agent's F1 and flag the leader with *
    lines.append("\nPer-emotion F1 (in [0, 1]) — leader marked with *:")
    header_parts = [f"{'Emotion':12s}"]
    for name in agent_names:
        short = name.split("_")[0][:12]
        header_parts.append(f"{short:>12s}")
    lines.append("  " + "  ".join(header_parts))
    lines.append("  " + "-" * (14 + 14 * len(agent_names)))

    for emo in EMOTIONS:
        vals = {
            name: per_agent_f1_dict[name]["per_emotion_f1"].get(emo, 0.0)
            for name in agent_names
        }
        best_val = max(vals.values())
        row = [f"{emo:12s}"]
        for name in agent_names:
            v = vals[name]
            marker = "*" if v == best_val and best_val > 0 else " "
            row.append(f"{v:>10.4f}{marker} ")
        lines.append("  " + "  ".join(row))

    # per-language Macro-F1
    all_langs = sorted(
        {lang for name in agent_names for lang in per_agent_f1_dict[name]["per_language_macro_f1"]}
    )
    if all_langs:
        lines.append("\nPer-language Macro-F1 (%) — leader marked with *:")
        header_parts = [f"{'Language':8s}"]
        for name in agent_names:
            short = name.split("_")[0][:12]
            header_parts.append(f"{short:>12s}")
        lines.append("  " + "  ".join(header_parts))
        lines.append("  " + "-" * (10 + 14 * len(agent_names)))

        for lang in all_langs:
            vals = {
                name: per_agent_f1_dict[name]["per_language_macro_f1"].get(lang, 0.0)
                for name in agent_names
            }
            best_val = max(vals.values())
            row = [f"{lang:8s}"]
            for name in agent_names:
                v = vals[name]
                marker = "*" if v == best_val and best_val > 0 else " "
                row.append(f"{v:>10.2f}{marker} ")
            lines.append("  " + "  ".join(row))

    return "\n".join(lines)


def print_specialization_report(predictions_jsonl_path: str) -> dict:
    """Compute and print the full specialization report"""
    results = per_agent_f1(predictions_jsonl_path)
    summary = specialization_summary(results)
    print(summary)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute emergent specialization analysis for Config B predictions."
    )
    parser.add_argument("jsonl", help="Path to config_b_{split}.jsonl")
    parser.add_argument("--save", default=None, help="Optional path to save JSON results")
    args = parser.parse_args()

    results = print_specialization_report(args.jsonl)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save}")
