"""Inter-layer conditioning templates and depth-topology system prompts"""

import json

# ---------------------------------------------------------------------------
# C-L2 designed system prompts (verbatim from EXPERIMENT_SPEC.md)
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = (
    "You are a critical reviewer of emotion analysis. Your job is to find and "
    "correct errors in a previous analysis. When reviewing, specifically check for:\n"
    "(a) Emotions that may have been missed because they are expressed implicitly, "
    "culturally, or through understatement.\n"
    "(b) Emotions that may have been incorrectly assigned due to surface-level "
    "pattern matching without considering context, sarcasm, or irony.\n"
    "(c) Confidence scores that seem miscalibrated relative to the evidence.\n"
    "(d) Inconsistencies between the reasoning and the predicted labels.\n\n"
    "If the previous analysis is correct, confirm it and explain why. Do not "
    "change predictions without justification."
)

CALIBRATOR_SYSTEM = (
    "You are performing final calibration on an emotion analysis that has "
    "undergone two rounds of review. Your task is to:\n"
    "(a) Confirm emotions where the evidence is strong and both prior analyses agree.\n"
    "(b) Adjudicate where the two analyses disagree, weighing the evidence for each side.\n"
    "(c) Remove emotions assigned on weak or ambiguous evidence — prioritize "
    "precision over recall.\n"
    "(d) Ensure confidence scores accurately reflect the strength of evidence."
)

# Map from config system_prompt value → actual prompt string
DEPTH_PROMPT_MAP = {
    "critic": CRITIC_SYSTEM,
    "calibrator": CALIBRATOR_SYSTEM,
    "": "",
}


# ---------------------------------------------------------------------------
# Inter-layer user message builders
# ---------------------------------------------------------------------------

def format_layer1_user_prompt(text: str, task_instruction: str) -> str:
    """Layer 1: no prior output. Standard task format."""
    return f"{task_instruction}\n\nText: {text}"


def format_subsequent_layer_user_prompt(
    text: str,
    prior_output: dict,
    task_instruction: str,
    layer_name: str,
) -> str:
    """Layers 2+: prepend the prior layer's JSON output before the task.

    Structure (verbatim from EXPERIMENT_SPEC.md Section 4.1):
        A previous analysis of the following text produced this result:
        ---
        {PRIOR JSON}
        ---
        Now analyze the same text yourself. You may agree with, modify, or
        completely revise the previous analysis. Provide your own classification.

        {TASK INSTRUCTION}

        Text: {TEXT}
    """
    prior_json = json.dumps(prior_output, indent=2)

    conditioning = (
        f"A previous analysis of the following text produced this result:\n"
        f"---\n{prior_json}\n---\n\n"
        f"Now analyze the same text yourself. You may agree with, modify, or "
        f"completely revise the previous analysis. Provide your own classification."
    )

    return f"{conditioning}\n\n{task_instruction}\n\nText: {text}"


def format_critic_user_prompt(
    text: str,
    prior_output: dict,
    task_instruction: str,
) -> str:
    """C-L2 Layer 2 (Critical Reviewer): verbatim structure from spec Section 4.2."""
    prior_json = json.dumps(prior_output, indent=2)

    conditioning = (
        f"A previous analysis of the following text produced this result:\n"
        f"---\n{prior_json}\n---\n\n"
        f"Provide your revised classification."
    )

    return f"{conditioning}\n\n{task_instruction}\n\nText: {text}"


def format_calibrator_user_prompt(
    text: str,
    prior_output: dict,
    task_instruction: str,
) -> str:
    """C-L2 Layer 3 (Calibrator): verbatim structure from spec Section 4.2."""
    prior_json = json.dumps(prior_output, indent=2)

    conditioning = (
        f"The most recent analysis of the following text produced this result:\n"
        f"---\n{prior_json}\n---\n\n"
        f"Provide your final classification."
    )

    return f"{conditioning}\n\n{task_instruction}\n\nText: {text}"
