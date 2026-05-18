"""Training data preparation for B-L3 QLoRA adapters.

Converts BRIGHTER CSV training data into chat-format examples for SFT.
Each example:
  [SYSTEM]    CoT enhancement block for the language
  [USER]      Task instruction + text
  [ASSISTANT] Ground-truth JSON output

Also handles adapter-specific filtering:
  - Adapter A (general):     all training data
  - Adapter B (ambiguity):   multi-label instances + mDeBERTa error instances
  - Adapter C (contrastive): all training data (same as A, different loss)
"""

import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.loader import EMOTIONS, LANGUAGES
from src.prompts.cot_enhancement import build_cot_block
from src.prompts.task_template import format_user_prompt


def build_assistant_response(label_dict: dict[str, int]) -> str:
    """Build the ground-truth JSON assistant response from a label dict.

    Confidence: 0.9 for present emotions, 0.1 for absent (avoids hard 1.0/0.0
    to reduce overconfidence in training targets).
    Reasoning: brief templated explanation derived from present emotions.
    """
    present = [e for e, v in label_dict.items() if v == 1]
    absent = [e for e, v in label_dict.items() if v == 0]

    confidence = {}
    for e in EMOTIONS:
        confidence[e] = 0.9 if label_dict.get(e, 0) == 1 else 0.1

    if present:
        if len(present) == 1:
            reasoning = f"The text expresses {present[0]}."
        else:
            reasoning = f"The text expresses multiple emotions: {', '.join(present)}."
    else:
        reasoning = "No clear emotion is expressed in this text."

    output = {
        "emotions": {e: label_dict.get(e, 0) for e in EMOTIONS},
        "confidence": {e: round(confidence[e], 1) for e in EMOTIONS},
        "reasoning": reasoning,
    }
    return json.dumps(output, ensure_ascii=False)


def load_training_examples(
    data_path: str,
    reasoning_banks_path: str,
    languages: Optional[list[str]] = None,
    adapter_type: str = "general",
    mdeberta_errors_path: Optional[str] = None,
) -> list[dict]:
    """Load and format training examples for a given adapter type.

    Args:
        data_path: path to cleaned_csv directory
        reasoning_banks_path: path to reasoning_banks directory
        languages: language codes to include (default: all 9)
        adapter_type: one of "general", "ambiguity", "contrastive"
        mdeberta_errors_path: optional path to JSON file mapping text_id → bool
                              (True if mDeBERTa made an error). Used for adapter B.

    Returns:
        List of chat-format dicts: {"messages": [...], "language": lang,
                                    "text_id": id, "labels": {...}, "weight": float}
    """
    if languages is None:
        languages = LANGUAGES

    # Load mDeBERTa errors if provided (for adapter B)
    mdeberta_errors: set[str] = set()
    if mdeberta_errors_path and Path(mdeberta_errors_path).exists():
        with open(mdeberta_errors_path) as f:
            err_data = json.load(f)
        mdeberta_errors = {k for k, v in err_data.items() if v}

    examples = []

    for lang in languages:
        path = Path(data_path) / lang / "train.csv"
        if not path.exists():
            print(f"  [data_prep] No train split for {lang}, skipping.")
            continue

        df = pd.read_csv(path)
        required = ["text"] + EMOTIONS
        if not all(c in df.columns for c in required):
            print(f"  [data_prep] Missing columns in {lang} train, skipping.")
            continue

        for _, row in df.iterrows():
            text = str(row["text"])
            labels = {e: int(row[e]) for e in EMOTIONS}
            text_id = str(row.get("id", "")) if "id" in df.columns else ""
            n_present = sum(labels.values())

            # Adapter-specific filtering
            if adapter_type == "ambiguity":
                is_multilabel = n_present >= 2
                is_mdeberta_error = text_id in mdeberta_errors
                if not (is_multilabel or is_mdeberta_error):
                    continue
                # 2x weight for multi-label instances
                weight = 2.0 if is_multilabel else 1.0
            else:
                weight = 1.0

            # Build chat messages
            cot_block = build_cot_block(lang, reasoning_banks_path)
            user_message = format_user_prompt(text)
            assistant_response = build_assistant_response(labels)

            example = {
                "messages": [
                    {"role": "system", "content": cot_block},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_response},
                ],
                "language": lang,
                "text_id": text_id,
                "labels": labels,
                "weight": weight,
            }
            examples.append(example)

    return examples


def apply_chat_template(examples: list[dict], tokenizer) -> list[dict]:
    """Apply the model's chat template to format messages for training.

    Returns list of dicts with "input_ids", "attention_mask", "labels", "weight".
    For SFT: loss is computed only on the assistant turn (labels for system+user = -100).
    """
    formatted = []
    for ex in examples:
        messages = ex["messages"]

        # Tokenize full conversation
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Find where assistant turn starts to mask system+user from loss
        assistant_prefix = tokenizer.apply_chat_template(
            messages[:-1],  # system + user only
            tokenize=False,
            add_generation_prompt=True,
        )

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=1024,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Compute prefix length (tokens to mask from loss)
        prefix_ids = tokenizer(
            assistant_prefix,
            truncation=True,
            max_length=1024,
            return_tensors=None,
        )["input_ids"]
        prefix_len = len(prefix_ids)

        # Labels: -100 for prefix, real ids for assistant turn
        label_ids = [-100] * prefix_len + input_ids[prefix_len:]

        entry = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "weight": ex["weight"],
        }
        # Carry emotion labels for contrastive loss (list of 6 ints in EMOTIONS order)
        if "labels" in ex and isinstance(ex["labels"], dict):
            entry["emotion_labels"] = [ex["labels"].get(e, 0) for e in EMOTIONS]
        formatted.append(entry)

    return formatted


def get_data_stats(examples: list[dict]) -> dict:
    """Return statistics about a prepared training set."""
    langs = {}
    multilabel = 0
    for ex in examples:
        lang = ex["language"]
        langs[lang] = langs.get(lang, 0) + 1
        if sum(ex["labels"].values()) >= 2:
            multilabel += 1

    return {
        "total": len(examples),
        "by_language": langs,
        "multilabel_count": multilabel,
        "multilabel_pct": round(100 * multilabel / len(examples), 1) if examples else 0,
        "weighted_total": round(sum(ex["weight"] for ex in examples), 1),
    }
