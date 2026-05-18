#!/usr/bin/env python3
"""A7 runner: B-L3 with a minimal system prompt.

Monkey-patches the cot_block builder used by src.topologies.width so that the
system prompt becomes a single CoT instruction line, with no language-specific
reasoning bank and no decision policy. Then forwards all CLI arguments to the
canonical run_b_l3 entry point.

The pipeline (src/) is not modified.

Canonical cot_block (per src/prompts/cot_enhancement.py) contains:
  1. "Before classifying, reason step by step about the emotions in this text."
  2. Language-specific reasoning bank (loaded from JSON).
  3. Decision policy ("Identify emotional cues first; ... default to a single
     dominant emotion ...; assign 1 only when evidence is compelling").

This runner replaces all three with line (1) plus a minimal language tag.
The user message (TASK_INSTRUCTION + text) is unchanged — the JSON format
spec still ships in the user role, as it does for canonical B-L3.

Usage:
    python ablation/b_l3_minimal_prompt/scripts/run.py \\
        --split dev \\
        --config ablation/b_l3_minimal_prompt/config.yaml \\
        --output-dir ablation/b_l3_minimal_prompt \\
        --batch
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import the topologies module so we can patch its bound `build_cot_block`.
# Crucial: width.py uses `from src.prompts.cot_enhancement import build_cot_block`,
# which copies the function reference into the width module namespace at import
# time. Patching src.prompts.cot_enhancement after that point would not affect
# width — we have to patch the binding inside `width_module` itself.
import src.topologies.width as width_module  # noqa: E402


def _minimal_cot_block(language: str, reasoning_banks_path=None) -> str:  # noqa: ARG001
    """Replacement cot_block: bare CoT instruction + language tag, nothing else."""
    return (
        f"Before classifying, reason step by step about the emotions in this text. "
        f"The text is in {language}."
    )


width_module.build_cot_block = _minimal_cot_block


def _verify_patch_applied():
    """Sanity-check the patch took effect; abort if not."""
    sample = width_module.build_cot_block("pcm")
    assert "reasoning bank" not in sample.lower(), "patch did not strip reasoning bank!"
    assert "decision policy" not in sample.lower(), "patch did not strip decision policy!"
    assert sample.count("\n") <= 1, f"patch produced >1 line: {sample!r}"
    print(f"[A7 minimal_prompt] cot_block patched. Sample: {sample!r}")


_verify_patch_applied()

# Forward CLI args to the canonical runner.
import experiments.run_b_l3 as run_b_l3_module  # noqa: E402

run_b_l3_module.main()
