"""Per-label majority vote aggregation for width topology."""

from src.agents.base_agent import AgentOutput
from src.data.loader import EMOTIONS


def majority_vote(
    outputs: list[AgentOutput],
    threshold: int = 2,
    emotions: list[str] | None = None,
) -> dict[str, int]:
    """Per-label majority vote across agent outputs.

    For each emotion, if >= threshold agents predict 1, final = 1.
    """
    if emotions is None:
        emotions = EMOTIONS

    result = {}
    for emo in emotions:
        votes = sum(1 for o in outputs if o.emotions.get(emo, 0) == 1)
        result[emo] = 1 if votes >= threshold else 0
    return result


def confidence_weighted_average(
    outputs: list[AgentOutput],
    threshold: float = 0.5,
    emotions: list[str] | None = None,
) -> tuple[dict[str, int], dict[str, float]]:
    """Confidence-weighted averaging across agent outputs.

    For each emotion, compute weighted average of confidence scores,
    then threshold to get binary prediction.
    """
    if emotions is None:
        emotions = EMOTIONS

    n = len(outputs)
    predictions = {}
    avg_confidence = {}

    for emo in emotions:
        total_conf = sum(o.confidence.get(emo, 0.5) for o in outputs)
        avg = total_conf / n if n > 0 else 0.5
        avg_confidence[emo] = round(avg, 4)
        predictions[emo] = 1 if avg >= threshold else 0

    return predictions, avg_confidence
