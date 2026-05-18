#!/usr/bin/env python3
"""A9 runner: B-L3 with PER-AGENT system prompts.

Monkey-patches LoRAAgent.get_system_prompt() to return a different bias
string per agent name. The canonical reasoning bank + decision policy
(cot_block) is still appended after, so:

    system_prompt = <per-agent bias> + "\\n\\n" + <canonical CoT block>

This isolates whether prompt-level role differentiation amplifies
adapter-level specialization. B-L2 already showed prompt-only role
differentiation adds ~0pp on test in width — but that was *without*
adapters. The interaction between prompt-role and adapter-role hasn't
been tested.

Per-agent biases below match each adapter's training distribution:
  * general_specialist:   broad classifier, holistic emotion identification
  * ambiguity_specialist: focus on weak/co-occurring/subtle signals
  * contrastive_specialist: focus on disambiguating similar emotions

The pipeline (src/) is not modified.

Usage:
    python ablation/b_l3_per_agent_prompt/scripts/run.py \\
        --split dev \\
        --config ablation/b_l3_per_agent_prompt/config.yaml \\
        --output-dir ablation/b_l3_per_agent_prompt \\
        --batch
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import src.agents.lora_agent as lora_agent_module  # noqa: E402


PER_AGENT_PROMPTS = {
    "general_specialist": (
        "You are a general emotion classifier. Analyze the text holistically "
        "and identify all emotions that are clearly expressed."
    ),
    "ambiguity_specialist": (
        "You specialize in detecting ambiguous and co-occurring emotions. "
        "Pay particular attention to weak, implicit, or mixed emotional signals "
        "that other classifiers might miss."
    ),
    "contrastive_specialist": (
        "You compare candidate emotion labels and resolve borderline cases. "
        "Focus on distinguishing between similar emotions (e.g. anger vs disgust, "
        "fear vs surprise, sadness vs disappointment) and arbitrating close calls."
    ),
}


def _per_agent_get_system_prompt(self) -> str:
    return PER_AGENT_PROMPTS.get(self.name, "")


lora_agent_module.LoRAAgent.get_system_prompt = _per_agent_get_system_prompt


def _verify_patch():
    """Confirm the bound method is now ours."""
    func = lora_agent_module.LoRAAgent.get_system_prompt
    assert func is _per_agent_get_system_prompt, "patch did not bind"
    print("[A9 per_agent_prompt] LoRAAgent.get_system_prompt patched.")
    for name, prompt in PER_AGENT_PROMPTS.items():
        print(f"  {name}: {prompt[:60]}...")


_verify_patch()


# Forward CLI args to the canonical runner.
import experiments.run_b_l3 as run_b_l3_module  # noqa: E402

run_b_l3_module.main()
