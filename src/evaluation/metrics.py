"""Evaluation metrics: Macro-F1, Micro-F1, per-language, per-emotion breakdowns."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.loader import EMOTIONS


def compute_f1_scores(
    y_true: list[dict[str, int]],
    y_pred: list[dict[str, int]],
    emotions: list[str] | None = None,
) -> dict:
    """Compute Macro-F1 and Micro-F1 for multi-label emotion predictions"""
    if emotions is None:
        emotions = EMOTIONS

    # Convert to numpy arrays: (n_samples, n_labels)
    true_arr = np.array([[d.get(e, 0) for e in emotions] for d in y_true])
    pred_arr = np.array([[d.get(e, 0) for e in emotions] for d in y_pred])

    macro_f1 = f1_score(true_arr, pred_arr, average="macro", zero_division=0)
    micro_f1 = f1_score(true_arr, pred_arr, average="micro", zero_division=0)

    # Per-emotion F1
    per_emotion = {}
    for i, emo in enumerate(emotions):
        per_emotion[emo] = {
            "f1": f1_score(true_arr[:, i], pred_arr[:, i], zero_division=0),
            "precision": precision_score(true_arr[:, i], pred_arr[:, i], zero_division=0),
            "recall": recall_score(true_arr[:, i], pred_arr[:, i], zero_division=0),
            "support": int(true_arr[:, i].sum()),
        }

    return {
        "macro_f1": round(float(macro_f1) * 100, 2),
        "micro_f1": round(float(micro_f1) * 100, 2),
        "per_emotion": {
            emo: {k: round(float(v), 4) if isinstance(v, float) else v
                  for k, v in stats.items()}
            for emo, stats in per_emotion.items()
        },
        "n_samples": len(y_true),
    }


def compute_per_language_results(
    results_by_lang: dict[str, dict],
) -> dict:
    """Aggregate per-language results into a summary table"""
    languages = list(results_by_lang.keys())
    macro_f1s = [results_by_lang[l]["macro_f1"] for l in languages]
    micro_f1s = [results_by_lang[l]["micro_f1"] for l in languages]

    return {
        "per_language": {
            lang: {
                "macro_f1": results_by_lang[lang]["macro_f1"],
                "micro_f1": results_by_lang[lang]["micro_f1"],
                "n_samples": results_by_lang[lang]["n_samples"],
            }
            for lang in languages
        },
        "avg_macro_f1": round(float(np.mean(macro_f1s)), 2),
        "avg_micro_f1": round(float(np.mean(micro_f1s)), 2),
    }
