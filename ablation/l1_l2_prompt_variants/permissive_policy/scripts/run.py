#!/usr/bin/env python3
"""Variant 3: B-L1 with a PERMISSIVE (recall-favoring) decision policy.

Monkey-patches build_cot_block() in src.topologies.width to return the same
CoT block as canonical, but with the precision-prior decision policy
("default to single dominant emotion; assign 1 only when compelling")
replaced by a recall-prior:
    "Label any emotion that is plausibly present, even on weak evidence.
     When uncertain between two related emotions, prefer to assign both."

Reasoning bank and bare CoT instruction unchanged.

Tests whether flipping the policy direction recovers vmw/zul (which collapse
to all-zero abstention under canonical B-L1) and at what cost on languages
where the canonical prior was helping.

Usage:
    python ablation/l1_l2_prompt_variants/permissive_policy/scripts/run.py \\
        --split dev \\
        --config ablation/l1_l2_prompt_variants/permissive_policy/config.yaml \\
        --output-dir ablation/l1_l2_prompt_variants/permissive_policy \\
        --languages chn pcm vmw zul \\
        --batch
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import src.topologies.width as width_module  # noqa: E402
from src.prompts.cot_enhancement import load_reasoning_bank  # noqa: E402


def _permissive_cot_block(language: str, reasoning_banks_path=None) -> str:
    """Reconstruct build_cot_block() with a recall-favoring decision policy."""
    language_guidance = None
    if reasoning_banks_path:
        language_guidance = load_reasoning_bank(reasoning_banks_path, language)

    lines = ["Before classifying, reason step by step about the emotions in this text."]

    if language_guidance:
        lines.append(f"\nLanguage-specific guidance for {language}:")
        lines.append(language_guidance)
    else:
        lines.append(f"\nNote: Text is in {language}. Apply appropriate cultural and linguistic knowledge.")

    lines.append(
        "\nDecision policy:"
        "\n- Identify emotional cues first (words, phrases, emoji, tone)"
        f"\n- Consider cultural and linguistic context for {language}"
        "\n- Label any emotion that is plausibly present, even on weak evidence"
        "\n- When uncertain between two related emotions, prefer to assign both"
        "\n- Assigning 0 to all emotions should be rare; only do so when"
        "\n  the text is clearly emotionally neutral"
    )
    return "\n".join(lines)


width_module.build_cot_block = _permissive_cot_block


def _verify_patch():
    out = width_module.build_cot_block("pcm", reasoning_banks_path="data/reasoning_banks")
    assert "Default to a single dominant emotion" not in out, "old precision-prior text still present"
    assert "Assign 1 only when evidence is compelling" not in out, "old precision-prior text still present"
    assert "Label any emotion that is plausibly present" in out, "new permissive policy missing"
    assert "Decision policy" in out, "policy header missing"
    print("[permissive_policy] build_cot_block patched.")
    print("  - language_guidance: kept")
    print("  - decision policy: REPLACED with recall-favoring rule")
    print(f"  sample CoT length for pcm: {len(out)} chars")


_verify_patch()


import experiments.run_b_l1 as run_b_l1_module  # noqa: E402

run_b_l1_module.main()
