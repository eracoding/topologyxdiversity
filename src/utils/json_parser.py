import json
import re
from typing import Optional

EMOTIONS_MAP = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

def parse_agent_output(raw: str) -> Optional[dict]:
    # 1
    result = _try_parse(raw.strip())
    if result and _validate_schema(result):
        return _normalize(result)
    # 2
    fenced = _extract_fenced_json(raw)
    if fenced:
        result = _try_parse(fenced)
        if result and _validate_schema(result):
            return _normalize(result)
    # 3
    braced = _extract_braced_json(raw)
    if braced:
        result = _try_parse(braced)
        if result and _validate_schema(result):
            return _normalize(result)
    # 4 
    return _regex_extract(raw)


def _try_parse(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    fixed = text.replace("'", '"')
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    return None

def _validate_schema(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    if "emotions" not in data:
        return False
    emotions = data["emotions"]
    if not isinstance(emotions, dict):
        return False
    return any(e in emotions for e in EMOTIONS_MAP)

def _normalize(data: dict) -> dict:
    emotions = {}
    confidence = {}
    for emo in EMOTIONS_MAP:
        raw_val = data.get("emotions", {}).get(emo, 0)
        if isinstance(raw_val, (int, float)):
            emotions[emo] = 1 if raw_val >= 0.5 else 0
        elif isinstance(raw_val, bool):
            emotions[emo] = 1 if raw_val else 0
        elif isinstance(raw_val, str):
            emotions[emo] = 1 if raw_val.lower() in ("1", "true", "yes") else 0
        else:
            emotions[emo] = 0
        
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


def _extract_fenced_json(text: str) -> Optional[str]:
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def _extract_braced_json(text:str) -> Optional[str]:
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
                return text[start: i + 1]
    return None

def _regex_extract(text: str) -> Optional[dict]:
    emotions = {}
    confidence = {}

    for emo in EMOTIONS_MAP:
        pattern = rf'["\']?{emo}["\']?\s*:\s*([01])'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            emotions[emo] = int(match.group(1))
        else:
            emotions[emo] = 0

        conf_pattern = rf'["\']?{emo}["\']?\s*:\s*(0\.\d+|1\.0|0|1)'
        conf_section = re.search(r'"confidence"\s*:\s*\{([^}]+)\}', text, re.DOTALL)
        if conf_section:
            conf_match = re.search(conf_pattern, conf_section.group(1))
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


if __name__ == __name__:

    test1 = '{"emotions": {"joy": 1, "sadness": 0, "anger": 0}, "confidence": {"joy": 0.9, "sadness": 0.1, "anger": 0.2}, "reasoning": "The text is clearly positive."}'
    test2 = 'Here is the analysis:\n```json\n{"emotions": {"fear": 1, "surprise": 1}, "confidence": {"fear": 0.8, "surprise": 0.6}, "reasoning": "The situation is alarming."}\n```\nLet me know if you need anything else.'
    test3 = 'After careful analysis of the input, I concluded: {"emotions": {"disgust": 1, "anger": 1}, "confidence": {"disgust": 0.75, "anger": 0.85}, "reasoning": "Strong negative reaction detected."} Hope this helps!'
    test4 = 'The agent output was corrupted. Detected emotions: joy: 1, sadness: 0, "reasoning": "Partial output recovered by regex."'

    # 1
    print(parse_agent_output(test1))
    # 2
    print(parse_agent_output(test2))
    # 3
    print(parse_agent_output(test3))
    # 4
    print(parse_agent_output(test4))