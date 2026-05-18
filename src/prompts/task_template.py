"""Shared task instruction template for all agents.

Verbatim from EXPERIMENT_SPEC.md Section 2.3.
"""

TASK_INSTRUCTION = """Task: Classify the emotions expressed in the following text. The text may express zero, one, or multiple emotions simultaneously.

Emotion categories: anger, disgust, fear, joy, sadness, surprise

For each emotion, decide if it is present (1) or absent (0). Provide a confidence score between 0.0 and 1.0, and brief reasoning.

Respond ONLY with a valid JSON object in exactly this format, with no additional text before or after:
{
  "emotions": {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0, "surprise": 0},
  "confidence": {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.0, "surprise": 0.0},
  "reasoning": "Your analysis here."
}"""


def format_user_prompt(text: str) -> str:
    """Format the user message with task instruction and input text."""
    return f"{TASK_INSTRUCTION}\n\nText: {text}"
