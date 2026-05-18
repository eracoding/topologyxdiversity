# Topology versus Diversity in Multi-Agent LLMs: A Controlled Matrix for Multilingual Low-Resource Emotion Detection

Multi-agent emotion classification on the BRIGHTER benchmark (SemEval-2025 Task 11),
factoring agent design into two orthogonal axes — **topology** (how the three calls
relate to one another) and **diversity source** (what makes the three calls differ) —
and evaluating the full 2 × 3 cross-product on a single open-weight backbone at a
fixed inference budget of **3 LLM calls per sample**.

| Topology \\ Diversity | L1 — Stochastic (seed) | L2 — Designed (lens prompt) | L3 — Learned (QLoRA adapter) |
|---|---|---|---|
| **Width** (parallel + per-label majority vote) | B-L1 | B-L2 | B-L3 |
| **Depth** (sequential + JSON conditioning, T₁ > T₂ > T₃) | C-L1 | C-L2 | C-L3 |

> Codebase naming: directories and config files use `b_*` for Width and `c_*` for Depth.
> Paper notation uses W/D; the mapping is W ↔ B, D ↔ C.

All six cells share the same backbone (Qwen2.5-14B-Instruct served by vLLM on a
single 40 GB GPU), the same per-language reasoning bank, the same family-transfer
prompting protocol, and the same precision-prior decision policy — so any
differences in macro-F1 are attributable to the topology × diversity choice
alone, not to scale, supervision, or prompting infrastructure.

## Method at a glance

- **Backbone.** `Qwen/Qwen2.5-14B-Instruct`, 4-bit quantised, served by vLLM with
  LoRA hot-swapping (`vllm.lora.request.LoRARequest`).
- **Languages (9 target).** `pcm chn mar ary tat vmw ptmz zul ind` — no
  target-language supervision; all adapters and prompts are induced from English.
- **Emotions.** `anger disgust fear joy sadness surprise` (binary, multi-label).
- **Width topology** (`src/topologies/width.py`). Three parallel `PromptAgent` or
  `LoRAAgent` instances see the same input; per-label majority vote (≥ 2 of 3)
  produces the final binary vector.
- **Depth topology** (`src/topologies/depth.py`). Three sequential layers; layer
  ℓ > 1 receives the input plus layer ℓ−1's full JSON output as conditioning,
  with a descending temperature schedule (0.7 → 0.5 → 0.3). The final prediction
  is the layer-3 output — no separate aggregation step.
- **Diversity sources.**
  - **L1 (Stochastic).** Identical prompt across the three calls, sampling seeds
    differ (42 / 123 / 456).
  - **L2 (Designed).** Handcrafted analytical-lens system prompts vary across
    the three calls; for Depth-L2 these become Analyst / Critic / Calibrator.
  - **L3 (Learned).** QLoRA adapters (rank 8, α 16, dropout 0.05, BCE +
    auxiliary losses) — three parallel adapters for Width-L3
    (`general / ambiguity / contrastive`), three sequentially-trained adapters
    for Depth-L3 (`layer1 / layer2 / layer3`, each trained on prior-layer
    mistakes with `error_upweight = 3.0` on Layer 2 and a calibration penalty
    on Layer 3).

## Repository layout

```
agentic_ai_project/
├── src/                          # library code (importable)
│   ├── agents/                   # PromptAgent, LoRAAgent (vLLM wrappers)
│   ├── topologies/               # width.py, depth.py — orchestrators
│   ├── aggregation/              # majority_vote.py, confidence-weighted variants
│   ├── prompts/                  # system prompts, CoT enhancement, inter-layer
│   ├── data/                     # BRIGHTER loader, reasoning-bank loader
│   ├── evaluation/               # metrics, diversity, layer-transition analysis
│   └── utils/                    # cost tracker, JSON parser
├── experiments/                  # one runner script per cell
│   ├── run_b_l1.py / run_b_l2.py / run_b_l3.py
│   ├── run_c_l1.py / run_c_l2.py / run_c_l3.py
│   ├── run_baseline_cot.py       # CoT single-pass reference floor
│   └── runner_utils.py           # shared per-language loop + JSONL I/O
├── training/                     # QLoRA training (L3 cells only)
│   ├── train_b_adapters.py       # parallel training of 3 Width adapters
│   ├── train_c_adapters.py       # 5-phase sequential pipeline for Depth
│   └── data_prep.py              # BRIGHTER CSV → chat-format SFT examples
├── config/                       # one YAML per cell + base.yaml
├── data/
│   ├── cleaned_csv/<lang>/{train,dev,test}.csv   # BRIGHTER cleaned splits
│   ├── reasoning_banks/          # per-language CoT exemplars
│   └── shot_banks/               # per-language few-shot exemplars
├── models/adapters/              # trained QLoRA weights (one dir per adapter)
├── outputs/                      # test-split runs (predictions / results / figures)
├── outputs_dev/                  # dev-split runs (kept separate for tuning)
├── logs/                         # one .md per run — what was launched, key numbers
├── ablation/                     # 13 self-contained ablations (see ablation/README.md)
└── test.ipynb                    # scratch notebook
```

## Setup

Tested on a single NVIDIA GPU with ≥ 40 GB VRAM, CUDA 12.x, Python 3.10+.

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 2. core dependencies (no requirements.txt is shipped — install directly)
pip install "vllm>=0.6" "torch>=2.4" "transformers>=4.45" \
            "peft>=0.13" "bitsandbytes>=0.43" "accelerate>=1.0" \
            pandas numpy scikit-learn pyyaml jsonlines tqdm

# 3. fetch the base model (~28 GB, cached by HF)
python -c "from huggingface_hub import snapshot_download; \
           snapshot_download('Qwen/Qwen2.5-14B-Instruct')"

# 4. place BRIGHTER cleaned splits under data/cleaned_csv/<lang>/{train,dev,test}.csv
#    columns required: text, anger, disgust, fear, joy, sadness, surprise
```

## Running an experiment

Every cell follows the same CLI contract:

```bash
python experiments/run_<cell>.py --split {dev,test} [--languages pcm chn …] \
                                 [--batch]            # use vLLM native batched chat
```

Outputs land under `outputs/` (test split) or `outputs_dev/` (dev split):

```
outputs/
├── predictions/   {b,c}_<level>_<lang>_<split>.jsonl   # per-sample agent outputs + final
├── results/       {b,c}_<level>_<lang>_<split>_results.csv
│                  {b,c}_<level>_<lang>_<split>_cost.json
└── figures/       per-language / per-cell plots
```

### Prompt-only cells (no training required)

```bash
python experiments/run_b_l1.py --split test --batch     # Width, stochastic
python experiments/run_b_l2.py --split test --batch     # Width, designed lens
python experiments/run_c_l1.py --split test --batch     # Depth, stochastic
python experiments/run_c_l2.py --split test --batch     # Depth, designed roles
```

### Width-L3 (3 parallel adapters)

```bash
# train all three in one shot (or pass --adapter {general,ambiguity,contrastive})
python training/train_b_adapters.py --adapter all

# inference
python experiments/run_b_l3.py --split test --batch
```

### Depth-L3 (3 sequentially-trained adapters)

Each adapter is trained on the predictions of the previous one, so the phases
must run **in order**, with an inference-collection step between consecutive
training phases:

```bash
python training/train_c_adapters.py --phase train1     # train Layer 1
python training/train_c_adapters.py --phase collect1   # collect Layer 1 predictions
python training/train_c_adapters.py --phase train2     # train Layer 2 (error_corrector)
python training/train_c_adapters.py --phase collect2   # collect Layer 2 predictions
python training/train_c_adapters.py --phase train3     # train Layer 3 (calibrator)

python experiments/run_c_l3.py --split test --batch
```

### Baseline

```bash
python experiments/run_baseline_cot.py --split test    # CoT single-pass reference
```

## Configuration

`config/base.yaml` holds shared model / data / evaluation defaults (model name,
language list, emotion list, output paths). Each cell has its own
`config/config_<cell>.yaml` that sets topology-specific fields (number of agents
or layers, sampling parameters per slot, aggregation rule for Width, per-layer
adapter path for Depth, training hyperparameters for the L3 cells).

To run only a subset of languages, pass `--languages` on the command line — it
overrides the YAML's language list.

## Headline test-set results (macro-F1, averaged across 9 languages)

| Cell | macro-F1 | Δ vs CoT baseline |
|---|---:|---:|
| CoT single-pass (baseline) | 48.47 | — |
| B-L1 — Width / Stochastic | 40.66 | −7.81 |
| B-L2 — Width / Designed   | 40.71 | −7.76 |
| **B-L3 — Width / Learned**| **52.83** | **+4.36** |
| C-L1 — Depth / Stochastic | 43.88 | −4.59 |
| C-L2 — Depth / Designed   | 45.07 | −3.40 |
| C-L3 — Depth / Learned    | 50.00 | +1.53 |

Dev-set numbers and per-language breakdowns live in `logs/` (one markdown file
per run) and `outputs_dev/results/` / `outputs/results/`.

## Citation

If you use this codebase, please cite:

```bibtex
@misc{shernazarov2026topology,
  author = {Shernazarov, Ulugbek},
  title  = {Topology versus Diversity in Multi-Agent LLMs: 
             A Controlled Matrix for Multilingual Low-Resource Emotion Detection},
  year   = {2026},
  url    = {https://github.com/eracoding/agentic_ai_project}
}
```

## License

MIT