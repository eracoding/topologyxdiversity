"""Robust JSON parsing for LLM output.

LLMs frequently produce malformed JSON: markdown fences, extra text before/after,
trailing commas, single quotes, etc. This module provides fallback parsing.
"""

import json
import re
from typing import Optional

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def parse_agent_output(raw: str) -> Optional[dict]:
    """Parse LLM output into the expected agent output schema.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Extract JSON from markdown code fences
    3. Find first { ... } block in the text
    4. Regex-based extraction of emotion values

    Returns parsed dict or None if all strategies fail.
    """
    # Strategy 1: direct parse
    result = _try_parse(raw.strip())
    if result and _validate_schema(result):
        return _normalize(result)

    # Strategy 2: extract from markdown fences
    fenced = _extract_fenced_json(raw)
    if fenced:
        result = _try_parse(fenced)
        if result and _validate_schema(result):
            return _normalize(result)

    # Strategy 3: find outermost { ... } block
    braced = _extract_braced_json(raw)
    if braced:
        result = _try_parse(braced)
        if result and _validate_schema(result):
            return _normalize(result)

    # Strategy 4: regex extraction as last resort
    return _regex_extract(raw)


def _try_parse(text: str) -> Optional[dict]:
    """Attempt JSON parse with minor fixups."""
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try fixing trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try replacing single quotes with double quotes
    fixed = text.replace("'", '"')
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _extract_fenced_json(text: str) -> Optional[str]:
    """Extract JSON from ```json ... ``` or ``` ... ``` blocks."""
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_braced_json(text: str) -> Optional[str]:
    """Extract the outermost { ... } block from text."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _validate_schema(data: dict) -> bool:
    """Check that the parsed dict has the expected structure."""
    if not isinstance(data, dict):
        return False
    # Must have emotions dict
    if "emotions" not in data:
        return False
    emotions = data["emotions"]
    if not isinstance(emotions, dict):
        return False
    # Must have at least some emotion keys
    return any(e in emotions for e in EMOTIONS)


def _normalize(data: dict) -> dict:
    """Normalize parsed output to canonical schema."""
    emotions = {}
    confidence = {}
    for emo in EMOTIONS:
        # Normalize emotion values to 0/1
        raw_val = data.get("emotions", {}).get(emo, 0)
        if isinstance(raw_val, (int, float)):
            emotions[emo] = 1 if raw_val >= 0.5 else 0
        elif isinstance(raw_val, bool):
            emotions[emo] = 1 if raw_val else 0
        elif isinstance(raw_val, str):
            emotions[emo] = 1 if raw_val.lower() in ("1", "true", "yes") else 0
        else:
            emotions[emo] = 0

        # Normalize confidence to float
        raw_conf = data.get("confidence", {}).get(emo, 0.5)
        try:
            confidence[emo] = float(raw_conf)
        except (ValueError, TypeError):
            confidence[emo] = 0.5

    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return {
        "emotions": emotions,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def _regex_extract(text: str) -> Optional[dict]:
    """Last-resort regex extraction of emotion predictions."""
    emotions = {}
    confidence = {}

    for emo in EMOTIONS:
        # Look for patterns like "anger": 1 or "anger": 0
        pattern = rf'["\']?{emo}["\']?\s*:\s*([01])'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            emotions[emo] = int(match.group(1))
        else:
            emotions[emo] = 0

        # Look for confidence values
        conf_pattern = rf'["\']?{emo}["\']?\s*:\s*(0\.\d+|1\.0|0|1)'
        # Search in confidence section if present
        conf_section = re.search(r'"confidence"\s*:\s*\{([^}]+)\}', text, re.DOTALL)
        if conf_section:
            conf_match = re.search(conf_pattern, conf_section.group(1), re.IGNORECASE)
            if conf_match:
                confidence[emo] = float(conf_match.group(1))
            else:
                confidence[emo] = 0.5
        else:
            confidence[emo] = 0.5

    if not any(v == 1 for v in emotions.values()):
        return None

    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
    reasoning = reasoning_match.group(1) if reasoning_match else ""

    return {
        "emotions": emotions,
        "confidence": confidence,
        "reasoning": reasoning,
    }
