#!/usr/bin/env python3
"""Variant 2: B-L1 with NO precision-prior decision policy.

Monkey-patches build_cot_block() in src.topologies.width to return:
  - the bare CoT instruction
  - the language-specific reasoning bank (or fallback note)
but WITHOUT the "Default to single dominant emotion / Assign 1 only when
compelling / when uncertain assign 0" decision policy block.

Tests whether the precision-prior is responsible for the all-zero
abstention rates on vmw/zul under canonical B-L1.

Usage:
    python ablation/l1_l2_prompt_variants/no_decision_policy/scripts/run.py \\
        --split dev \\
        --config ablation/l1_l2_prompt_variants/no_decision_policy/config.yaml \\
        --output-dir ablation/l1_l2_prompt_variants/no_decision_policy \\
        --languages chn pcm vmw zul \\
        --batch
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import src.topologies.width as width_module  # noqa: E402
from src.prompts.cot_enhancement import load_reasoning_bank  # noqa: E402


def _no_policy_cot_block(language: str, reasoning_banks_path=None) -> str:
    """Reconstruct build_cot_block() but skip the decision policy section."""
    language_guidance = None
    if reasoning_banks_path:
        language_guidance = load_reasoning_bank(reasoning_banks_path, language)

    lines = ["Before classifying, reason step by step about the emotions in this text."]

    if language_guidance:
        lines.append(f"\nLanguage-specific guidance for {language}:")
        lines.append(language_guidance)
    else:
        lines.append(f"\nNote: Text is in {language}. Apply appropriate cultural and linguistic knowledge.")

    # NOTE: decision policy intentionally omitted.
    return "\n".join(lines)


width_module.build_cot_block = _no_policy_cot_block


def _verify_patch():
    out = width_module.build_cot_block("pcm", reasoning_banks_path="data/reasoning_banks")
    assert "Decision policy" not in out, "decision policy still present"
    assert "Default to a single dominant emotion" not in out, "policy text still present"
    assert "Assign 1 only when evidence is compelling" not in out, "policy text still present"
    print("[no_decision_policy] build_cot_block patched.")
    print("  - language_guidance: kept")
    print("  - decision policy: STRIPPED")
    print(f"  sample CoT length for pcm: {len(out)} chars")


_verify_patch()


import experiments.run_b_l1 as run_b_l1_module  # noqa: E402

run_b_l1_module.main()
