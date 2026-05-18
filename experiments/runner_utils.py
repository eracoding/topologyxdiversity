"""Shared utilities for experiment runners.

Output schema follows EXPERIMENT_SPEC.md Section 2.7:
  outputs/predictions/{config}_{level}_{language}.jsonl
  Each line: text_id, text, gold, agent_1/2/3, aggregated

Depth topology (Config C) uses a parallel schema:
  Each line: text_id, text, gold, layer_1, layer_2, layer_3, final
  "final" mirrors the "aggregated" key so downstream tooling is compatible.
"""

import json
from pathlib import Path

import jsonlines
import pandas as pd

from src.data.loader import EMOTIONS
from src.evaluation.metrics import compute_f1_scores


def run_language_with_topology(
    topology,
    texts: list[str],
    labels: list[dict[str, int]],
    ids: list[str],
    language: str,
) -> tuple[list[dict], dict, dict]:
    """Run topology on all texts for one language.

    Returns:
        (records, aggregated_metrics, per_agent_metrics)
        records: list of JSONL records (spec schema)
        aggregated_metrics: output of compute_f1_scores on aggregated preds
        per_agent_metrics: {agent_name: compute_f1_scores result}
    """
    from tqdm import tqdm

    records = []
    agg_preds = []
    # per-agent predictions: {agent_name: [pred_dict, ...]}
    agent_pred_lists: dict[str, list[dict]] = {}

    for i, (text, gold, text_id) in enumerate(
        tqdm(zip(texts, labels, ids), total=len(texts), desc=f"  {language}", leave=False)
    ):
        result = topology.predict_single(text, language=language)
        agg_preds.append(result["emotions"])

        # Build JSONL record per spec
        record = {
            "text_id": text_id,
            "text": text,
            "gold": gold,
        }
        for j, agent_out in enumerate(result["agent_outputs"]):
            key = f"agent_{j + 1}"
            record[key] = {
                "emotions": agent_out.emotions,
                "confidence": agent_out.confidence,
                "reasoning": agent_out.reasoning,
                "parse_success": agent_out.parse_success,
            }
            agent_name = topology.agents[j].name
            agent_pred_lists.setdefault(agent_name, []).append(agent_out.emotions)

        record["aggregated"] = {
            "emotions": result["emotions"],
            "method": result["aggregation_method"],
        }
        records.append(record)

    aggregated_metrics = compute_f1_scores(labels, agg_preds)

    per_agent_metrics = {}
    for agent_name, preds in agent_pred_lists.items():
        per_agent_metrics[agent_name] = compute_f1_scores(labels, preds)

    return records, aggregated_metrics, per_agent_metrics


def run_language_with_topology_batch(
    topology,
    texts: list[str],
    labels: list[dict[str, int]],
    ids: list[str],
    language: str,
) -> tuple[list[dict], dict, dict]:
    """Run width topology on all texts for one language using vLLM batched chat.

    Calls topology.predict_batch(texts) once (3 batched GPU calls internally)
    instead of N×3 individual calls. Same return schema as run_language_with_topology.
    """
    print(f"  {language}: batching {len(texts)} samples...")
    batch_results = topology.predict_batch(texts, language=language)

    records = []
    agg_preds = []
    agent_pred_lists: dict[str, list[dict]] = {}

    for i, (result, gold, text_id, text) in enumerate(zip(batch_results, labels, ids, texts)):
        agg_preds.append(result["emotions"])
        record = {"text_id": text_id, "text": text, "gold": gold}

        for j, agent_out in enumerate(result["agent_outputs"]):
            key = f"agent_{j + 1}"
            record[key] = {
                "emotions": agent_out.emotions,
                "confidence": agent_out.confidence,
                "reasoning": agent_out.reasoning,
                "parse_success": agent_out.parse_success,
            }
            agent_name = topology.agents[j].name
            agent_pred_lists.setdefault(agent_name, []).append(agent_out.emotions)

        record["aggregated"] = {
            "emotions": result["emotions"],
            "method": result["aggregation_method"],
        }
        records.append(record)

    aggregated_metrics = compute_f1_scores(labels, agg_preds)
    per_agent_metrics = {
        name: compute_f1_scores(labels, preds)
        for name, preds in agent_pred_lists.items()
    }
    return records, aggregated_metrics, per_agent_metrics


def run_language_with_depth_topology_batch(
    topology,
    texts: list[str],
    labels: list[dict[str, int]],
    ids: list[str],
    language: str,
) -> tuple[list[dict], dict, dict, list[list]]:
    """Run depth topology on all texts for one language using vLLM batched chat.

    Calls topology.predict_many(texts) once (3 batched GPU calls, one per layer)
    instead of N×3 individual calls. Same return schema as run_language_with_depth_topology.
    """
    print(f"  {language}: batching {len(texts)} samples (3 layers)...")
    batch_results = topology.predict_many(texts, language=language)

    layer_names = [layer.name for layer in topology.layers]
    records = []
    final_preds = []
    layer_pred_lists: dict[str, list[dict]] = {}
    all_layer_outputs = []

    for result, gold, text_id, text in zip(batch_results, labels, ids, texts):
        final_preds.append(result["emotions"])
        all_layer_outputs.append(result["layer_outputs"])

        record = {"text_id": text_id, "text": text, "gold": gold}
        for j, layer_out in enumerate(result["layer_outputs"]):
            key = f"layer_{j + 1}"
            record[key] = {
                "emotions": layer_out.emotions,
                "confidence": layer_out.confidence,
                "reasoning": layer_out.reasoning,
                "parse_success": layer_out.parse_success,
            }
            name = layer_names[j]
            layer_pred_lists.setdefault(name, []).append(layer_out.emotions)

        record["aggregated"] = {"emotions": result["emotions"], "method": "final_layer"}
        records.append(record)

    final_metrics = compute_f1_scores(labels, final_preds)
    per_layer_metrics = {
        name: compute_f1_scores(labels, preds)
        for name, preds in layer_pred_lists.items()
    }
    return records, final_metrics, per_layer_metrics, all_layer_outputs


def load_results_from_jsonl(path: Path, agent_names: list[str]) -> tuple[dict, dict] | None:
    """Recompute metrics from an existing per-language JSONL file.

    Returns (aggregated_metrics, per_agent_metrics) matching the shape returned
    by run_language_with_topology, or None if the file does not exist.
    """
    if not path.exists():
        return None

    labels = []
    agg_preds = []
    agent_pred_lists: dict[str, list[dict]] = {name: [] for name in agent_names}

    with jsonlines.open(path) as reader:
        for record in reader:
            labels.append(record["gold"])
            agg_preds.append(record["aggregated"]["emotions"])
            for j, name in enumerate(agent_names):
                key = f"agent_{j + 1}"
                if key in record:
                    agent_pred_lists[name].append(record[key]["emotions"])

    aggregated_metrics = compute_f1_scores(labels, agg_preds)
    per_agent_metrics = {
        name: compute_f1_scores(labels, preds)
        for name, preds in agent_pred_lists.items()
        if preds
    }
    return aggregated_metrics, per_agent_metrics


def save_predictions_jsonl(records: list[dict], path: Path) -> None:
    """Save prediction records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for record in records:
            writer.write(record)


def run_language_with_depth_topology(
    topology,
    texts: list[str],
    labels: list[dict[str, int]],
    ids: list[str],
    language: str,
) -> tuple[list[dict], dict, dict, list[list]]:
    """Run a DepthTopology on all texts for one language.

    Returns:
        (records, final_metrics, per_layer_metrics, all_layer_outputs)
        records: list of JSONL records
        final_metrics: compute_f1_scores on Layer 3 predictions
        per_layer_metrics: {layer_name: compute_f1_scores result}
        all_layer_outputs: list[list[AgentOutput]] — for layer_analysis
    """
    from tqdm import tqdm

    records = []
    final_preds = []
    layer_pred_lists: dict[str, list[dict]] = {}
    all_layer_outputs = []

    layer_names = [layer.name for layer in topology.layers]

    for text, gold, text_id in tqdm(
        zip(texts, labels, ids), total=len(texts), desc=f"  {language}", leave=False
    ):
        result = topology.predict_single(text, language=language)
        final_preds.append(result["emotions"])
        all_layer_outputs.append(result["layer_outputs"])

        record = {
            "text_id": text_id,
            "text": text,
            "gold": gold,
        }
        for j, layer_out in enumerate(result["layer_outputs"]):
            key = f"layer_{j + 1}"
            record[key] = {
                "emotions": layer_out.emotions,
                "confidence": layer_out.confidence,
                "reasoning": layer_out.reasoning,
                "parse_success": layer_out.parse_success,
            }
            name = layer_names[j]
            layer_pred_lists.setdefault(name, []).append(layer_out.emotions)

        # "aggregated" key mirrors final layer for cross-config compatibility
        record["aggregated"] = {
            "emotions": result["emotions"],
            "method": "final_layer",
        }
        records.append(record)

    final_metrics = compute_f1_scores(labels, final_preds)
    per_layer_metrics = {
        name: compute_f1_scores(labels, preds)
        for name, preds in layer_pred_lists.items()
    }

    return records, final_metrics, per_layer_metrics, all_layer_outputs


def load_depth_results_from_jsonl(
    path: Path,
    layer_names: list[str],
) -> tuple[dict, dict] | None:
    """Recompute metrics from an existing depth-topology JSONL file.

    Returns (final_metrics, per_layer_metrics) or None if file does not exist.
    Note: layer_outputs for layer_analysis are not reconstructable from JSONL
    (AgentOutput objects lost); layer_analysis must be re-run if needed.
    """
    if not path.exists():
        return None

    labels = []
    final_preds = []
    layer_pred_lists: dict[str, list[dict]] = {name: [] for name in layer_names}

    with jsonlines.open(path) as reader:
        for record in reader:
            labels.append(record["gold"])
            final_preds.append(record["aggregated"]["emotions"])
            for j, name in enumerate(layer_names):
                key = f"layer_{j + 1}"
                if key in record:
                    layer_pred_lists[name].append(record[key]["emotions"])

    final_metrics = compute_f1_scores(labels, final_preds)
    per_layer_metrics = {
        name: compute_f1_scores(labels, preds)
        for name, preds in layer_pred_lists.items()
        if preds
    }
    return final_metrics, per_layer_metrics


def build_results_csv(
    all_results: dict[str, dict],
    all_agent_results: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    """Build results DataFrame from per-language metrics.

    Args:
        all_results: {language: aggregated metrics dict}
        all_agent_results: {language: {agent_name: metrics dict}}

    Returns:
        DataFrame with one row per language + AVERAGE row.
    """
    rows = []
    for lang, metrics in all_results.items():
        row = {
            "language": lang,
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "n_samples": metrics["n_samples"],
        }
        for emo, stats in metrics["per_emotion"].items():
            row[f"{emo}_f1"] = round(stats["f1"] * 100, 2)

        # Per-agent aggregated macro-F1
        agent_metrics = all_agent_results.get(lang, {})
        for agent_name, a_metrics in agent_metrics.items():
            row[f"{agent_name}_macro_f1"] = a_metrics["macro_f1"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add AVERAGE row (numeric columns only)
    avg_row = {"language": "AVERAGE"}
    for col in df.columns:
        if col != "language":
            avg_row[col] = round(df[col].mean(), 2)
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    return df
