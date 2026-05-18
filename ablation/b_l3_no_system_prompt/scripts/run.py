#!/usr/bin/env python3
"""A8 runner: B-L3 with NO system prompt at all.

Pairs with A7 (minimal_prompt). Together they bracket the system-prompt
contribution to B-L3:
  canonical (~2400 chars: CoT + reasoning bank + decision policy)
  → A7      (~100 chars:  bare CoT instruction + language tag)
  → A8      (no system role at all — only the user message)

Two monkey-patches, both confined to this script:

  1. `width_module.build_cot_block` → returns "". Combined with bias=""
     for B-L3, this makes `system_prompt` resolve to "" inside
     WidthTopology.predict_batch.

  2. `vllm.LLM.chat` is wrapped to drop any message whose role=="system"
     and whose content is empty/whitespace, BEFORE the chat template is
     applied. Without this wrap, Qwen2.5's chat template would still
     emit `<|im_start|>system\\n\\n<|im_end|>` (an empty system role tag)
     instead of omitting the system role entirely. We want the cleanest
     possible bracket: A8 = "the model sees only the user turn".

The pipeline (src/) is not modified.

Usage:
    python ablation/b_l3_no_system_prompt/scripts/run.py \\
        --split dev \\
        --config ablation/b_l3_no_system_prompt/config.yaml \\
        --output-dir ablation/b_l3_no_system_prompt \\
        --batch
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import src.topologies.width as width_module  # noqa: E402
import vllm  # noqa: E402


# ---- Patch 1: build_cot_block returns empty ----
def _empty_cot_block(language: str, reasoning_banks_path=None) -> str:  # noqa: ARG001
    return ""


width_module.build_cot_block = _empty_cot_block


# ---- Patch 2: vllm.LLM.chat filters empty system messages ----
_original_chat = vllm.LLM.chat


def _is_empty_system(msg) -> bool:
    if not isinstance(msg, dict):
        return False
    if msg.get("role") != "system":
        return False
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip() == ""
    return not content


def _filter_messages(messages):
    """Strip empty system messages. Handle both batch (list[list[dict]])
    and single (list[dict]) shapes."""
    if not isinstance(messages, list) or not messages:
        return messages
    if isinstance(messages[0], list):
        # batch: list of conversations
        return [[m for m in conv if not _is_empty_system(m)] for conv in messages]
    if isinstance(messages[0], dict):
        # single conversation
        return [m for m in messages if not _is_empty_system(m)]
    return messages


def _patched_chat(self, messages=None, *args, **kwargs):
    if messages is not None:
        messages = _filter_messages(messages)
    return _original_chat(self, messages, *args, **kwargs)


vllm.LLM.chat = _patched_chat


# ---- Verify both patches before launching vLLM ----
def _verify_patches():
    assert width_module.build_cot_block("pcm") == "", "build_cot_block patch did not stick"

    test_batch = [
        [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}],
        [{"role": "system", "content": "  \n  "}, {"role": "user", "content": "hi"}],
        [{"role": "system", "content": "keep me"}, {"role": "user", "content": "hi"}],
    ]
    filtered = _filter_messages(test_batch)
    assert filtered[0] == [{"role": "user", "content": "hi"}], "empty system not stripped"
    assert filtered[1] == [{"role": "user", "content": "hi"}], "whitespace system not stripped"
    assert len(filtered[2]) == 2, "non-empty system was stripped (should be kept)"

    print("[A8 no_system_prompt] both patches verified.")
    print("  build_cot_block('pcm') -> ''")
    print("  vllm.LLM.chat now strips empty system messages")


_verify_patches()


# ---- Forward CLI args to canonical runner ----
import experiments.run_b_l3 as run_b_l3_module  # noqa: E402

run_b_l3_module.main()
