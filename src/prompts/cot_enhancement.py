"""CoT Enhancement Block with language-specific reasoning bank"""

import json
from pathlib import Path


def load_reasoning_bank(reasoning_banks_path: str, language: str) -> str | None:
    """Load the reasoning bank cues for a given language"""
    path = Path(reasoning_banks_path) / f"{language}_reasoning_seed42.json"
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "\n".join(str(item) for item in data)
    if isinstance(data, dict):
        # Flatten dict entries
        lines = []
        for k, v in data.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)
    return str(data)


def build_cot_block(language: str, reasoning_banks_path: str | None = None) -> str:
    """Build the language-specific CoT enhancement block"""
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
        "\n- Default to a single dominant emotion unless two distinct cue clusters"
        "\n  clearly justify multiple labels"
        "\n- Assign 1 only when evidence is compelling; when uncertain, assign 0"
    )

    return "\n".join(lines)
