"""Train QLoRA adapters for B-L3 Width topology.

Three adapters, each with different training data and loss strategy:
  - general:     all training data, standard BCE
  - ambiguity:   multi-label + mDeBERTa-error instances, 2x weight on multi-label
  - contrastive: all training data, BCE + contrastive loss on hidden states

All adapters share the same QLoRA hyperparameters from EXPERIMENT_SPEC.md Section 3.3.

Usage:
    python training/train_b_adapters.py --adapter all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class EmotionChatDataset(Dataset):
    """PyTorch dataset wrapping formatted training examples."""

    def __init__(self, formatted_examples: list[dict]):
        self.examples = formatted_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        item = {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
            "weight": torch.tensor(ex["weight"], dtype=torch.float),
        }
        # emotion_labels: used by contrastive adapter compute_loss
        if "emotion_labels" in ex:
            item["emotion_labels"] = torch.tensor(ex["emotion_labels"], dtype=torch.float)
        return item


def collate_fn(batch: list[dict]) -> dict:
    """Pad batch to max length."""
    max_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    weights = torch.zeros(len(batch), dtype=torch.float)
    has_emotion_labels = "emotion_labels" in batch[0]
    if has_emotion_labels:
        n_emotions = batch[0]["emotion_labels"].size(0)
        emotion_labels = torch.zeros(len(batch), n_emotions, dtype=torch.float)

    for i, ex in enumerate(batch):
        n = len(ex["input_ids"])
        input_ids[i, :n] = ex["input_ids"]
        attention_mask[i, :n] = ex["attention_mask"]
        labels[i, :n] = ex["labels"]
        weights[i] = ex["weight"]
        if has_emotion_labels:
            emotion_labels[i] = ex["emotion_labels"]

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "weights": weights,
    }
    if has_emotion_labels:
        result["emotion_labels"] = emotion_labels
    return result


# ─── Contrastive Loss ─────────────────────────────────────────────────────────

def contrastive_loss(
    hidden_states: torch.Tensor,
    label_matrix: torch.Tensor,
    margin: float = 0.5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss over per-emotion embeddings.

    For each emotion dimension, pushes representations of positive (label=1)
    examples away from negative (label=0) examples via cosine distance.

    Args:
        hidden_states: (batch, hidden_dim) — mean pooled last hidden state
        label_matrix:  (batch, n_emotions) — binary emotion labels for batch
        margin:        minimum cosine distance between pos and neg pairs
        temperature:   softmax temperature

    Returns scalar contrastive loss, or 0 if batch has no valid pairs.
    """
    from torch.nn.functional import normalize, cosine_similarity

    hidden_states = normalize(hidden_states, dim=-1)
    n_emotions = label_matrix.size(1)
    total_loss = torch.tensor(0.0, device=hidden_states.device)
    n_terms = 0

    for e in range(n_emotions):
        pos_mask = label_matrix[:, e] == 1   # (batch,)
        neg_mask = label_matrix[:, e] == 0

        pos_idx = pos_mask.nonzero(as_tuple=True)[0]
        neg_idx = neg_mask.nonzero(as_tuple=True)[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        pos_emb = hidden_states[pos_idx]  # (n_pos, hidden)
        neg_emb = hidden_states[neg_idx]  # (n_neg, hidden)

        # Pairwise cosine similarity: (n_pos, n_neg)
        sim = torch.mm(pos_emb, neg_emb.T)

        # Hinge loss: penalize when similarity > -margin (i.e., not far enough apart)
        loss_e = torch.clamp(sim + margin, min=0).mean()
        total_loss = total_loss + loss_e
        n_terms += 1

    return total_loss / n_terms if n_terms > 0 else total_loss


# ─── Custom Trainer ───────────────────────────────────────────────────────────

class WeightedSFTTrainer:
    """Minimal training loop supporting per-sample weights and optional contrastive loss.

    Uses HuggingFace Trainer internally but overrides the loss computation.
    """

    def __init__(
        self,
        model,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer,
        output_dir: str,
        training_args,
        adapter_type: str = "general",
        contrastive_alpha: float = 0.1,
        emotions: list[str] = None,
    ):
        from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

        self.adapter_type = adapter_type
        self.contrastive_alpha = contrastive_alpha
        self.emotions = emotions or ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

        # For contrastive adapter, wrap compute_loss
        if adapter_type == "contrastive":
            import types

            def compute_loss(trainer_self, model, inputs, return_outputs=False, **kwargs):
                weights = inputs.pop("weights", None)
                # emotion_labels: (batch, 6) float tensor — passed from dataset
                emotion_label_matrix = inputs.pop("emotion_labels", None)
                seq_labels = inputs.get("labels")

                # Request hidden states per-call (not globally) to avoid OOM
                outputs = model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                hidden = outputs.hidden_states[-1]  # last layer: (batch, seq, hidden)

                # Standard CLM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = seq_labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_labels.size())

                seq_loss = token_loss.sum(-1) / (shift_labels != -100).sum(-1).clamp(min=1)
                if weights is not None:
                    seq_loss = seq_loss * weights.to(seq_loss.device)
                base_loss = seq_loss.mean()

                # Contrastive term: mean-pool hidden state, push emotion boundaries apart
                if emotion_label_matrix is not None:
                    attn_mask = inputs["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * attn_mask).sum(1) / attn_mask.sum(1).clamp(min=1)
                    # Detach to avoid double-backprop through hidden states twice
                    c_loss = contrastive_loss(
                        pooled.float(),
                        emotion_label_matrix.to(pooled.device),
                    )
                    total_loss = base_loss + self.contrastive_alpha * c_loss
                else:
                    total_loss = base_loss

                # Free hidden states immediately to reclaim GPU memory
                del hidden
                return (total_loss, outputs) if return_outputs else total_loss

            self._trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=collate_fn,
            )
            self._trainer.compute_loss = types.MethodType(compute_loss, self._trainer)

        else:
            # Weighted BCE trainer (adapter A and B)
            import types

            def compute_loss(trainer_self, model, inputs, return_outputs=False, **kwargs):
                weights = inputs.pop("weights", None)
                inputs.pop("emotion_labels", None)  # not used for general/ambiguity
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs["labels"]

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_labels.size())

                seq_loss = token_loss.sum(-1) / (shift_labels != -100).sum(-1).clamp(min=1)
                if weights is not None:
                    seq_loss = seq_loss * weights.to(seq_loss.device)
                return (seq_loss.mean(), outputs) if return_outputs else seq_loss.mean()

            self._trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=collate_fn,
            )
            self._trainer.compute_loss = types.MethodType(compute_loss, self._trainer)

    def train(self):
        return self._trainer.train()

    def save_model(self, path: str):
        self._trainer.save_model(path)


# ─── Dev Macro-F1 evaluation callback ─────────────────────────────────────────

class MacroF1Callback:
    """Early-stopping callback based on dev Macro-F1.

    HuggingFace's EarlyStoppingCallback requires compute_metrics; this wraps it.
    """

    def __init__(self, patience: int = 2):
        self.patience = patience
        self.best_f1 = 0.0
        self.rounds_no_improve = 0

    def should_stop(self, current_f1: float) -> bool:
        if current_f1 > self.best_f1 + 0.001:
            self.best_f1 = current_f1
            self.rounds_no_improve = 0
        else:
            self.rounds_no_improve += 1
        return self.rounds_no_improve >= self.patience


# ─── Main training function ───────────────────────────────────────────────────

def train_adapter(
    adapter_type: str,
    base_cfg: dict,
    l3_cfg: dict,
    verbose: bool = True,
):
    """Train one QLoRA adapter"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from transformers import EarlyStoppingCallback
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from sklearn.metrics import f1_score

    from training.data_prep import (
        load_training_examples,
        apply_chat_template,
        get_data_stats,
    )
    from src.data.loader import EMOTIONS

    train_cfg = l3_cfg["training"]
    data_cfg = base_cfg["data"]
    output_dir = Path(train_cfg["output_dir"]) / f"b_l3_{adapter_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training adapter: b_l3_{adapter_type}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

    # ── Load tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg["base_model"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Prepare training data ───────────────────────────────────────────────
    mdeberta_errors_path = data_cfg.get("mdeberta_errors_path")

    train_examples = load_training_examples(
        data_path=data_cfg["base_path"],
        reasoning_banks_path=data_cfg["reasoning_banks_path"],
        languages=data_cfg["languages"],
        adapter_type=adapter_type,
        mdeberta_errors_path=mdeberta_errors_path,
    )

    # Dev data for early stopping (use dev split, all languages, general adapter type)
    dev_examples = load_training_examples(
        data_path=data_cfg["base_path"],
        reasoning_banks_path=data_cfg["reasoning_banks_path"],
        languages=data_cfg["languages"],
        adapter_type="general",   # no filtering for dev eval
    )
    # Replace "train.csv" with "dev.csv" — reload with dev split
    from training.data_prep import load_training_examples as _load
    # Dev split has no filtering; re-use a small subset for eval loss
    dev_examples_raw = []
    for lang in data_cfg["languages"]:
        import pandas as pd
        path = Path(data_cfg["base_path"]) / lang / "dev.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not all(c in df.columns for c in ["text"] + EMOTIONS):
            continue
        from src.prompts.cot_enhancement import build_cot_block
        from src.prompts.task_template import format_user_prompt
        from training.data_prep import build_assistant_response
        cot_block = build_cot_block(lang, data_cfg["reasoning_banks_path"])
        for _, row in df.iterrows():
            labels = {e: int(row[e]) for e in EMOTIONS}
            dev_examples_raw.append({
                "messages": [
                    {"role": "system", "content": cot_block},
                    {"role": "user", "content": format_user_prompt(str(row["text"]))},
                    {"role": "assistant", "content": build_assistant_response(labels)},
                ],
                "language": lang,
                "text_id": str(row.get("id", "")),
                "labels": labels,
                "weight": 1.0,
            })

    stats = get_data_stats(train_examples)
    if verbose:
        print(f"Training examples: {stats['total']} (weighted: {stats['weighted_total']})")
        print(f"Multi-label: {stats['multilabel_count']} ({stats['multilabel_pct']}%)")
        print(f"By language: {stats['by_language']}")
        print(f"Dev examples: {len(dev_examples_raw)}")

    # Apply chat template
    train_formatted = apply_chat_template(train_examples, tokenizer)
    dev_formatted = apply_chat_template(dev_examples_raw, tokenizer)

    train_dataset = EmotionChatDataset(train_formatted)
    dev_dataset = EmotionChatDataset(dev_formatted)

    # ── Load model in 4-bit ─────────────────────────────────────────────────
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        train_cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        # Do NOT set output_hidden_states=True here — storing all hidden layers
        # globally causes OOM on every forward pass. For the contrastive adapter
        # we request hidden states only inside compute_loss via model(**inputs, ...).
    )
    model = prepare_model_for_kbit_training(model)

    # ── Apply LoRA ──────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=train_cfg["lora_rank"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments ──────────────────────────────────────────────────
    # Total steps for warmup
    steps_per_epoch = max(1, len(train_dataset) // (train_cfg["batch_size"] * train_cfg["gradient_accumulation_steps"]))
    total_steps = steps_per_epoch * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["lr"],
        lr_scheduler_type=train_cfg["scheduler"],
        warmup_steps=warmup_steps,
        bf16=train_cfg["bf16"],
        fp16=False,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=False,  # keep 'weight' column
    )

    # ── Train ───────────────────────────────────────────────────────────────
    trainer = WeightedSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        training_args=training_args,
        adapter_type=adapter_type,
    )

    trainer._trainer.add_callback(
        EarlyStoppingCallback(early_stopping_patience=2)
    )

    if verbose:
        print("\nStarting training...")
    trainer.train()

    # ── Save adapter ────────────────────────────────────────────────────────
    adapter_path = output_dir
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # Save training metadata
    meta = {
        "adapter_type": adapter_type,
        "base_model": train_cfg["base_model"],
        "n_train_examples": stats["total"],
        "n_dev_examples": len(dev_examples_raw),
        "lora_rank": train_cfg["lora_rank"],
        "lora_alpha": train_cfg["lora_alpha"],
    }
    with open(output_dir / "adapter_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\nAdapter saved to {adapter_path}")

    return str(adapter_path)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train B-L3 QLoRA adapters")
    parser.add_argument(
        "--adapter",
        choices=["general", "ambiguity", "contrastive", "all"],
        required=True,
        help="Which adapter to train. 'all' trains all three sequentially.",
    )
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--l3-config", default="config/config_b_l3.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.l3_config) as f:
        l3_cfg = yaml.safe_load(f)

    adapters_to_train = (
        ["general", "ambiguity", "contrastive"]
        if args.adapter == "all"
        else [args.adapter]
    )

    for adapter_type in adapters_to_train:
        train_adapter(adapter_type, base_cfg, l3_cfg)

    print("\nAll done.")


if __name__ == "__main__":
    main()
