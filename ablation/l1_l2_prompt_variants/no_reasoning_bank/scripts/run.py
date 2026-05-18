#!/usr/bin/env python3
"""Variant 1: B-L1 with NO language-specific reasoning bank.

Monkey-patches build_cot_block() in src.topologies.width to return:
  - the bare CoT instruction
  - the language note (so the model still knows the input language)
  - the decision policy
but WITHOUT the language-specific reasoning bank content.

Pairs with A7 (which did the analogous test on B-L3).

Usage:
    python ablation/l1_l2_prompt_variants/no_reasoning_bank/scripts/run.py \\
        --split dev \\
        --config ablation/l1_l2_prompt_variants/no_reasoning_bank/config.yaml \\
        --output-dir ablation/l1_l2_prompt_variants/no_reasoning_bank \\
        --languages chn pcm vmw zul \\
        --batch
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import src.topologies.width as width_module  # noqa: E402


def _no_bank_cot_block(language: str, reasoning_banks_path=None) -> str:  # noqa: ARG001
    """Reconstruct build_cot_block() but skip the language_guidance section."""
    lines = ["Before classifying, reason step by step about the emotions in this text."]
    lines.append(f"\nNote: Text is in {language}. Apply appropriate cultural and linguistic knowledge.")
    lines.append(
        "\nDecision policy:"
        "\n- Identify emotional cues first (words, phrases, emoji, tone)"
        f"\n- Consider cultural and linguistic context for {language}"
        "\n- Default to a single dominant emotion unless two distinct cue clusters"
        "\n  clearly justify multiple labels"
        "\n- Assign 1 only when evidence is compelling; when uncertain, assign 0"
    )
    return "\n".join(lines)


width_module.build_cot_block = _no_bank_cot_block


def _verify_patch():
    out = width_module.build_cot_block("pcm", reasoning_banks_path="data/reasoning_banks")
    assert "Language-specific guidance" not in out, "language guidance still present"
    assert "Decision policy" in out, "decision policy was incorrectly stripped"
    assert "Note: Text is in pcm" in out, "language note missing"
    print("[no_reasoning_bank] build_cot_block patched.")
    print("  - language_guidance: STRIPPED")
    print("  - decision policy: kept")
    print("  - language note: kept")
    print(f"  sample CoT length for pcm: {len(out)} chars")


_verify_patch()


import experiments.run_b_l1 as run_b_l1_module  # noqa: E402

run_b_l1_module.main()
