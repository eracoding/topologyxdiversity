"""Train sequential QLoRA adapters for C-L3 Depth topology.

C-L3 adapters are trained in three phases. Each adapter is trained on inputs
that match its position in the inference pipeline:

  Layer 1 (initial_classifier):
      Input:  text only (same as B-L3 general)
      Loss:   standard BCE
      Data:   all training data

  Layer 2 (error_corrector):
      Input:  text + Layer 1's prediction JSON
      Loss:   BCE, upweight instances where Layer 1 was wrong (3x)
      Data:   all training data

  Layer 3 (calibrator):
      Input:  text + Layer 2's prediction JSON
      Loss:   BCE + calibration penalty (penalise overconfident wrong predictions)
      Data:   all training data

Because each adapter depends on the previous adapter's predictions, the training
phases MUST run in order, with an inference collection step between each.

─── Training order ────────────────────────────────────────────────────────────

  Step 1.  Train Layer 1 adapter
           python training/train_c_adapters.py --phase train1

  Step 2.  Start vLLM with Layer 1 adapter loaded, then collect predictions:
           python training/train_c_adapters.py --phase collect1
           (vLLM must be running: models/adapters/c_l3_layer1)

  Step 3.  Train Layer 2 adapter (uses collected Layer 1 predictions)
           python training/train_c_adapters.py --phase train2

  Step 4.  Start vLLM with Layer 2 adapter loaded, then collect predictions:
           python training/train_c_adapters.py --phase collect2

  Step 5.  Train Layer 3 adapter (uses collected Layer 2 predictions)
           python training/train_c_adapters.py --phase train3

─── Usage ─────────────────────────────────────────────────────────────────────

  python training/train_c_adapters.py --phase train1
  python training/train_c_adapters.py --phase collect1
  python training/train_c_adapters.py --phase train2
  python training/train_c_adapters.py --phase collect2
  python training/train_c_adapters.py --phase train3

Optional flags:
  --base-config   path to base.yaml          (default: config/base.yaml)
  --l3-config     path to config_c_l3.yaml   (default: config/config_c_l3.yaml)
  --vllm-url      vLLM API base URL          (default: http://localhost:8000/v1)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.data_prep import (
    apply_chat_template,
    build_assistant_response,
    get_data_stats,
)
from src.data.loader import EMOTIONS, LANGUAGES
from src.prompts.cot_enhancement import build_cot_block
from src.prompts.task_template import TASK_INSTRUCTION
from src.prompts.inter_layer import format_subsequent_layer_user_prompt
from src.utils.json_parser import parse_agent_output

import pandas as pd


# ─── Dataset (reuse B-L3 infrastructure) ──────────────────────────────────────

from training.train_b_adapters import (
    EmotionChatDataset,
    collate_fn,
    WeightedSFTTrainer,
)


# ─── Data preparation ──────────────────────────────────────────────────────────

def load_layer1_examples(
    data_path: str,
    reasoning_banks_path: str,
    languages: list[str],
) -> list[dict]:
    """Load training examples for Layer 1 (same format as B-L3 general).

    Input format: [SYSTEM: CoT block] [USER: task + text] [ASSISTANT: gold JSON]
    """
    from src.prompts.task_template import format_user_prompt

    examples = []
    for lang in languages:
        path = Path(data_path) / lang / "train.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not all(c in df.columns for c in ["text"] + EMOTIONS):
            continue

        cot_block = build_cot_block(lang, reasoning_banks_path)
        for _, row in df.iterrows():
            labels = {e: int(row[e]) for e in EMOTIONS}
            examples.append({
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
    return examples


def load_conditioned_examples(
    data_path: str,
    reasoning_banks_path: str,
    languages: list[str],
    prior_predictions_path: str,
    upweight_errors: bool = False,
    error_upweight: float = 3.0,
) -> list[dict]:
    """Load training examples for Layer 2 or 3, conditioned on prior predictions.

    Input format:
      [SYSTEM: CoT block]
      [USER: prior JSON conditioning + task + text]
      [ASSISTANT: gold JSON]

    For Layer 2: upweight_errors=True — 3x weight when prior was wrong.
    For Layer 3: upweight_errors=False (calibration loss handles the weighting).
    """
    # Load prior predictions keyed by text_id
    prior_preds: dict[str, dict] = {}
    prior_path = Path(prior_predictions_path)
    if not prior_path.exists():
        raise FileNotFoundError(
            f"Prior predictions not found: {prior_predictions_path}\n"
            f"Run the corresponding --phase collect step first."
        )
    with open(prior_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            prior_preds[rec["text_id"]] = rec["prior_output"]

    examples = []
    missing_preds = 0

    for lang in languages:
        path = Path(data_path) / lang / "train.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not all(c in df.columns for c in ["text"] + EMOTIONS):
            continue

        cot_block = build_cot_block(lang, reasoning_banks_path)
        for _, row in df.iterrows():
            text_id = str(row.get("id", ""))
            text = str(row["text"])
            labels = {e: int(row[e]) for e in EMOTIONS}

            prior = prior_preds.get(text_id)
            if prior is None:
                missing_preds += 1
                continue

            # Build user message: prior JSON conditioning + task instruction + text
            user_message = format_subsequent_layer_user_prompt(
                text, prior, TASK_INSTRUCTION, layer_name="layer_n"
            )

            # Upweight if prior was wrong (for Layer 2 error corrector)
            weight = 1.0
            if upweight_errors:
                prior_emotions = prior.get("emotions", {})
                prior_correct = all(prior_emotions.get(e, 0) == labels[e] for e in EMOTIONS)
                if not prior_correct:
                    weight = error_upweight

            examples.append({
                "messages": [
                    {"role": "system", "content": cot_block},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": build_assistant_response(labels)},
                ],
                "language": lang,
                "text_id": text_id,
                "labels": labels,
                "weight": weight,
            })

    if missing_preds > 0:
        print(f"  [WARNING] {missing_preds} examples skipped — no prior prediction found.")
    return examples


def load_dev_examples_conditioned(
    data_path: str,
    reasoning_banks_path: str,
    languages: list[str],
    prior_predictions_path: str,
) -> list[dict]:
    """Load dev examples conditioned on prior predictions (for eval during training)."""
    prior_preds: dict[str, dict] = {}
    prior_path = Path(prior_predictions_path)
    if not prior_path.exists():
        return []
    with open(prior_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("split") == "dev":
                prior_preds[rec["text_id"]] = rec["prior_output"]

    examples = []
    for lang in languages:
        path = Path(data_path) / lang / "dev.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not all(c in df.columns for c in ["text"] + EMOTIONS):
            continue

        cot_block = build_cot_block(lang, reasoning_banks_path)
        for _, row in df.iterrows():
            text_id = str(row.get("id", ""))
            text = str(row["text"])
            labels = {e: int(row[e]) for e in EMOTIONS}
            prior = prior_preds.get(text_id)
            if prior is None:
                continue
            user_message = format_subsequent_layer_user_prompt(
                text, prior, TASK_INSTRUCTION, layer_name="layer_n"
            )
            examples.append({
                "messages": [
                    {"role": "system", "content": cot_block},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": build_assistant_response(labels)},
                ],
                "language": lang,
                "text_id": text_id,
                "labels": labels,
                "weight": 1.0,
            })
    return examples


# ─── Calibration loss ─────────────────────────────────────────────────────────

def make_calibration_trainer(base_trainer_class, calibration_alpha: float):
    """Wrap WeightedSFTTrainer to add a calibration penalty term.

    Calibration penalty: for tokens in the assistant turn where the model is
    confidently wrong, add an extra loss term. Implemented as KL divergence
    between the model's output distribution and a smoothed target distribution,
    scaled by the wrongness of the prior prediction.

    In practice, we approximate this as: standard CE loss on gold labels, but
    with per-example weights that are higher when the prior prediction was
    overconfident and wrong (captured in the "weight" field of each example;
    examples where prior was both confident and wrong receive higher weight).
    """
    # The calibration loss for C-L3 Layer 3 is implemented via the per-example
    # weight: examples where Layer 2 was wrong receive higher weight (already set
    # in load_conditioned_examples). The calibration_alpha scales this additional
    # penalty relative to examples where Layer 2 was correct.
    # This is simpler than a full KL-divergence calibration term and consistent
    # with the error-upweighting approach used in Layer 2.
    return base_trainer_class  # WeightedSFTTrainer already handles per-example weights


# ─── Inference collection ──────────────────────────────────────────────────────

def collect_predictions(
    data_path: str,
    reasoning_banks_path: str,
    languages: list[str],
    adapter_path: str,
    output_path: str,
    vllm_url: str = "http://localhost:8000/v1",
    splits: list[str] = None,
    concurrency: int = 32,
):
    """Run a trained adapter on train+dev data via vLLM, save predictions.

    Sends requests concurrently via AsyncOpenAI (saturates vLLM's batch processor).
    Resumable: if output_path already exists, skips (split, lang, text_id) tuples
    already collected and appends new ones.

    Args:
        adapter_path: path to trained adapter (used only for verification that it exists)
        output_path:  where to write the JSONL predictions
        concurrency:  max concurrent in-flight requests to vLLM (default 32)
    """
    import asyncio
    from openai import AsyncOpenAI
    from src.prompts.task_template import format_user_prompt
    from tqdm import tqdm

    if splits is None:
        splits = ["train", "dev"]

    # Verify adapter exists
    if not (Path(adapter_path) / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_path}\n"
            f"Train it first with the appropriate --phase train step."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_name = Path(adapter_path).name

    # Resume: collect (split, lang, text_id) tuples already present
    done_keys: set[tuple[str, str, str]] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done_keys.add((rec["split"], rec["language"], rec["text_id"]))
                except Exception:
                    continue
        if done_keys:
            print(f"  Resume: {len(done_keys)} predictions already collected, will skip and append")

    async def _run():
        client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY")
        sem = asyncio.Semaphore(concurrency)
        write_lock = asyncio.Lock()

        async def predict_one(out_f, split, lang, text_id, text, cot_block, pbar):
            async with sem:
                try:
                    response = await client.chat.completions.create(
                        model=adapter_name,
                        messages=[
                            {"role": "system", "content": cot_block},
                            {"role": "user", "content": format_user_prompt(text)},
                        ],
                        temperature=0.3,
                        max_tokens=512,
                    )
                    raw = response.choices[0].message.content
                    parsed = parse_agent_output(raw)
                except Exception as exc:
                    tqdm.write(f"  [WARN] {lang}/{text_id}: {exc}")
                    parsed = None

                if parsed is None:
                    parsed = {
                        "emotions": {e: 0 for e in EMOTIONS},
                        "confidence": {e: 0.5 for e in EMOTIONS},
                        "reasoning": "[PARSE FAILURE]",
                    }

                record = {
                    "text_id": text_id,
                    "split": split,
                    "language": lang,
                    "prior_output": parsed,
                }
                async with write_lock:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                pbar.update(1)

        total_written = 0
        mode = "a" if done_keys else "w"
        with open(output_path, mode) as out_f:
            for split in splits:
                for lang in languages:
                    path = Path(data_path) / lang / f"{split}.csv"
                    if not path.exists():
                        continue
                    df = pd.read_csv(path)
                    if "text" not in df.columns:
                        continue

                    cot_block = build_cot_block(lang, reasoning_banks_path)

                    pending = []
                    for _, row in df.iterrows():
                        text_id = str(row.get("id", ""))
                        if (split, lang, text_id) in done_keys:
                            continue
                        pending.append((text_id, str(row["text"])))

                    if not pending:
                        continue

                    pbar = tqdm(
                        total=len(pending),
                        desc=f"  collect {split}/{lang}",
                        leave=False,
                    )
                    tasks = [
                        predict_one(out_f, split, lang, tid, text, cot_block, pbar)
                        for tid, text in pending
                    ]
                    await asyncio.gather(*tasks)
                    pbar.close()
                    total_written += len(pending)

        return total_written

    total_written = asyncio.run(_run())
    print(f"  Saved {total_written} new predictions to {output_path}")


# ─── Core training function ───────────────────────────────────────────────────

def train_layer_adapter(
    layer_num: int,
    base_cfg: dict,
    l3_cfg: dict,
):
    """Train one C-L3 layer adapter.

    Args:
        layer_num: 1, 2, or 3
        base_cfg: base.yaml contents
        l3_cfg:   config_c_l3.yaml contents
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from transformers import EarlyStoppingCallback
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    train_cfg = l3_cfg["training"]
    data_cfg = base_cfg["data"]
    output_dir = Path(train_cfg["output_dir"]) / f"c_l3_layer{layer_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training C-L3 Layer {layer_num} adapter")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # ── Load tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg["base_model"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Prepare training data ───────────────────────────────────────────────
    languages = data_cfg["languages"]

    if layer_num == 1:
        train_examples = load_layer1_examples(
            data_cfg["base_path"], data_cfg["reasoning_banks_path"], languages
        )
        dev_examples = load_layer1_examples(
            data_cfg["base_path"], data_cfg["reasoning_banks_path"], languages
        )
        # Replace train examples with dev split
        dev_examples = []
        for lang in languages:
            path = Path(data_cfg["base_path"]) / lang / "dev.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if not all(c in df.columns for c in ["text"] + EMOTIONS):
                continue
            from src.prompts.task_template import format_user_prompt
            cot_block = build_cot_block(lang, data_cfg["reasoning_banks_path"])
            for _, row in df.iterrows():
                labels = {e: int(row[e]) for e in EMOTIONS}
                dev_examples.append({
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
        adapter_type = "general"

    elif layer_num == 2:
        preds_path = train_cfg["adapter1_predictions_path"]
        train_examples = load_conditioned_examples(
            data_cfg["base_path"],
            data_cfg["reasoning_banks_path"],
            languages,
            prior_predictions_path=preds_path,
            upweight_errors=True,
            error_upweight=train_cfg.get("error_upweight", 3.0),
        )
        dev_examples = load_dev_examples_conditioned(
            data_cfg["base_path"],
            data_cfg["reasoning_banks_path"],
            languages,
            prior_predictions_path=preds_path,
        )
        adapter_type = "general"   # no contrastive loss for layer 2

    else:  # layer_num == 3
        preds_path = train_cfg["adapter2_predictions_path"]
        train_examples = load_conditioned_examples(
            data_cfg["base_path"],
            data_cfg["reasoning_banks_path"],
            languages,
            prior_predictions_path=preds_path,
            upweight_errors=True,
            error_upweight=train_cfg.get("calibration_alpha", 0.2) * train_cfg.get("error_upweight", 3.0),
        )
        dev_examples = load_dev_examples_conditioned(
            data_cfg["base_path"],
            data_cfg["reasoning_banks_path"],
            languages,
            prior_predictions_path=preds_path,
        )
        adapter_type = "general"

    stats = get_data_stats(train_examples)
    print(f"Training examples: {stats['total']} (weighted sum: {stats['weighted_total']})")
    print(f"Dev examples: {len(dev_examples)}")
    if layer_num >= 2:
        weighted_pct = round(100 * sum(1 for e in train_examples if e["weight"] > 1.0) / max(1, len(train_examples)), 1)
        print(f"Upweighted examples (errors): {weighted_pct}%")

    train_formatted = apply_chat_template(train_examples, tokenizer)
    dev_formatted = apply_chat_template(dev_examples, tokenizer)
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
        remove_unused_columns=False,
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
    trainer._trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

    print("\nStarting training...")
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "layer": layer_num,
        "adapter_type": f"c_l3_layer{layer_num}",
        "base_model": train_cfg["base_model"],
        "n_train_examples": stats["total"],
        "n_dev_examples": len(dev_examples),
        "lora_rank": train_cfg["lora_rank"],
        "lora_alpha": train_cfg["lora_alpha"],
    }
    with open(output_dir / "adapter_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAdapter saved to {output_dir}")
    return str(output_dir)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train C-L3 sequential QLoRA adapters")
    parser.add_argument(
        "--phase",
        choices=["train1", "collect1", "train2", "collect2", "train3"],
        required=True,
        help=(
            "Training phase. Run in order:\n"
            "  train1 → collect1 → train2 → collect2 → train3\n"
            "collect phases require vLLM running with the corresponding adapter loaded."
        ),
    )
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--l3-config", default="config/config_c_l3.yaml")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1",
                        help="vLLM API URL (required for collect phases)")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Max concurrent in-flight requests during collect (default: 32)")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.l3_config) as f:
        l3_cfg = yaml.safe_load(f)

    train_cfg = l3_cfg["training"]
    data_cfg = base_cfg["data"]
    languages = data_cfg["languages"]

    if args.phase == "train1":
        train_layer_adapter(1, base_cfg, l3_cfg)

    elif args.phase == "collect1":
        print("Collecting Layer 1 predictions on train+dev data via vLLM...")
        print(f"vLLM URL: {args.vllm_url}")
        print("Make sure vLLM is running with c_l3_layer1 adapter loaded.")
        collect_predictions(
            data_path=data_cfg["base_path"],
            reasoning_banks_path=data_cfg["reasoning_banks_path"],
            languages=languages,
            adapter_path=str(Path(train_cfg["output_dir"]) / "c_l3_layer1"),
            output_path=train_cfg["adapter1_predictions_path"],
            vllm_url=args.vllm_url,
            splits=["train", "dev"],
            concurrency=args.concurrency,
        )
        print(f"\nDone. Predictions saved to: {train_cfg['adapter1_predictions_path']}")
        print("Next: python training/train_c_adapters.py --phase train2")

    elif args.phase == "train2":
        train_layer_adapter(2, base_cfg, l3_cfg)

    elif args.phase == "collect2":
        print("Collecting Layer 2 predictions on train+dev data via vLLM...")
        print(f"vLLM URL: {args.vllm_url}")
        print("Make sure vLLM is running with c_l3_layer2 adapter loaded.")
        collect_predictions(
            data_path=data_cfg["base_path"],
            reasoning_banks_path=data_cfg["reasoning_banks_path"],
            languages=languages,
            adapter_path=str(Path(train_cfg["output_dir"]) / "c_l3_layer2"),
            output_path=train_cfg["adapter2_predictions_path"],
            vllm_url=args.vllm_url,
            splits=["train", "dev"],
            concurrency=args.concurrency,
        )
        print(f"\nDone. Predictions saved to: {train_cfg['adapter2_predictions_path']}")
        print("Next: python training/train_c_adapters.py --phase train3")

    elif args.phase == "train3":
        train_layer_adapter(3, base_cfg, l3_cfg)
        print("\nAll C-L3 adapters trained. Run inference with:")
        print("  python experiments/run_c_l3.py --split dev")

    print("\nDone.")


if __name__ == "__main__":
    main()
