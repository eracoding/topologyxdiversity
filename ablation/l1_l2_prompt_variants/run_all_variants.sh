#!/usr/bin/env bash
# B-L1 prompt-variant sweep — bounds the system-prompt design space for L1.
#
# Three variants of the canonical B-L1 CoT block, run on a 4-language dev
# subset (chn, pcm, vmw, zul). Each variant reuses canonical config_b_l1.yaml
# semantics; only the build_cot_block() output is monkey-patched at runtime.
#
# Variants:
#   1. no_reasoning_bank   — strip language-specific bank, keep decision policy
#   2. no_decision_policy  — strip "default abstain" rule, keep reasoning bank
#   3. permissive_policy   — replace precision-prior with recall-prior
#
# Languages chosen to span the resource spectrum:
#   chn  (high-resource, large dev set)
#   pcm  (mid-resource)
#   vmw  (lowest, ~97% all-zero abstention under canonical)
#   zul  (low, ~86% abstention)
#
# Each variant: ~10-15 min on dev for 4 languages (B-L1 batched is ~25 min for
# all 9 languages). Total: ~40 min wall-clock.
#
# Recommended invocation:
#   nohup bash ablation/l1_l2_prompt_variants/run_all_variants.sh \
#       > ablation/l1_l2_prompt_variants/run_all_variants.log 2>&1 &
#
# Then monitor:
#   tail -f ablation/l1_l2_prompt_variants/run_all_variants.log
#
# Pre-flight:
#   - vLLM HTTP server NOT running (in-process LLM cannot share GPU)
#   - GPU free (`nvidia-smi`)

set -u
cd "$(dirname "$0")/../.."   # project root

LANGS="chn pcm vmw zul"

PHASE_BANNER() {
    echo ""
    echo "###################################################################"
    echo "# $1"
    echo "# Started: $(date)"
    echo "###################################################################"
    echo ""
}

PHASE_BANNER "Variant 1: no_reasoning_bank — DEV"
python ablation/l1_l2_prompt_variants/no_reasoning_bank/scripts/run.py \
    --split dev \
    --config ablation/l1_l2_prompt_variants/no_reasoning_bank/config.yaml \
    --output-dir ablation/l1_l2_prompt_variants/no_reasoning_bank \
    --languages $LANGS \
    --batch
echo "Variant 1 exit code: $?"

PHASE_BANNER "Variant 2: no_decision_policy — DEV"
python ablation/l1_l2_prompt_variants/no_decision_policy/scripts/run.py \
    --split dev \
    --config ablation/l1_l2_prompt_variants/no_decision_policy/config.yaml \
    --output-dir ablation/l1_l2_prompt_variants/no_decision_policy \
    --languages $LANGS \
    --batch
echo "Variant 2 exit code: $?"

PHASE_BANNER "Variant 3: permissive_policy — DEV"
python ablation/l1_l2_prompt_variants/permissive_policy/scripts/run.py \
    --split dev \
    --config ablation/l1_l2_prompt_variants/permissive_policy/config.yaml \
    --output-dir ablation/l1_l2_prompt_variants/permissive_policy \
    --languages $LANGS \
    --batch
echo "Variant 3 exit code: $?"

PHASE_BANNER "ALL DONE"
echo "Result CSVs:"
ls ablation/l1_l2_prompt_variants/no_reasoning_bank/results/   2>/dev/null | sed 's/^/  no_reasoning_bank:   /'
ls ablation/l1_l2_prompt_variants/no_decision_policy/results/  2>/dev/null | sed 's/^/  no_decision_policy:  /'
ls ablation/l1_l2_prompt_variants/permissive_policy/results/   2>/dev/null | sed 's/^/  permissive_policy:   /'

echo ""
echo "Compare against canonical B-L1 dev with:"
echo "  cat outputs/results/b_l1_dev_results.csv | grep -E 'language|chn|pcm|vmw|zul|AVERAGE'"
