"""Layer-level analysis for depth topology (Config C)"""

from src.data.loader import EMOTIONS


def _emotions_equal(a: dict[str, int], b: dict[str, int]) -> bool:
    """Return True if two emotion dicts are identical across all 6 emotions."""
    return all(a.get(e, 0) == b.get(e, 0) for e in EMOTIONS)


def analyse_layer_transitions(
    layer_outputs_list: list[list],   # list of per-sample layer_outputs
    gold_labels: list[dict[str, int]],
) -> dict:
    """Compute layer transition statistics over a full language run.

    Args:
        layer_outputs_list: for each sample, a list of AgentOutput objects
                            (one per layer, in order).
        gold_labels: gold emotion dicts, same length as layer_outputs_list.

    Returns dict with:
        per_transition: list of dicts, one per layer transition (L1→L2, L2→L3)
            - name:            "L1→L2" or "L2→L3"
            - correction_rate: fraction of samples where change improved F1
            - error_rate:      fraction where change hurt F1
            - rubber_stamp:    fraction with no change at all
        oscillation_rate: fraction where L3 reverted to L1 after L2 changed it
        n_samples: number of samples analysed
    """
    n = len(layer_outputs_list)
    if n == 0:
        return {}

    num_layers = len(layer_outputs_list[0])
    assert num_layers >= 2, "Need at least 2 layers for transition analysis"

    # Per-transition stats: indexes (0,1) = L1→L2, (1,2) = L2→L3
    transitions = []
    for t in range(num_layers - 1):
        corrections = 0
        errors = 0
        rubber_stamps = 0

        for i, layer_outs in enumerate(layer_outputs_list):
            pred_prev = layer_outs[t].emotions
            pred_curr = layer_outs[t + 1].emotions
            gold = gold_labels[i]

            if _emotions_equal(pred_prev, pred_curr):
                rubber_stamps += 1
                continue

            # Changed — check if it was a correction or an error
            prev_matches = _emotions_equal(pred_prev, gold)
            curr_matches = _emotions_equal(pred_curr, gold)

            if not prev_matches and curr_matches:
                corrections += 1
            elif prev_matches and not curr_matches:
                errors += 1
            # else: both wrong or both right but differently — neutral change, not counted

        transitions.append({
            "name": f"L{t+1}→L{t+2}",
            "correction_rate": round(corrections / n, 4),
            "error_rate": round(errors / n, 4),
            "rubber_stamp_rate": round(rubber_stamps / n, 4),
        })

    # Oscillation: only meaningful when there are 3+ layers
    oscillation_rate = None
    if num_layers >= 3:
        oscillations = 0
        for layer_outs in layer_outputs_list:
            pred_l1 = layer_outs[0].emotions
            pred_l2 = layer_outs[1].emotions
            pred_l3 = layer_outs[2].emotions
            # L2 changed from L1, and L3 reverted back to L1
            if (not _emotions_equal(pred_l1, pred_l2)
                    and _emotions_equal(pred_l1, pred_l3)):
                oscillations += 1
        oscillation_rate = round(oscillations / n, 4)

    return {
        "per_transition": transitions,
        "oscillation_rate": oscillation_rate,
        "n_samples": n,
    }


def format_layer_analysis_report(analysis: dict) -> str:
    """Format layer analysis as a human-readable string for logging."""
    lines = [f"Layer transition analysis (n={analysis['n_samples']}):"]
    for t in analysis.get("per_transition", []):
        lines.append(
            f"  {t['name']}: "
            f"correction={t['correction_rate']:.1%}  "
            f"error={t['error_rate']:.1%}  "
            f"rubber_stamp={t['rubber_stamp_rate']:.1%}"
        )
    if analysis.get("oscillation_rate") is not None:
        lines.append(f"  Oscillation rate (L3 reverts to L1): {analysis['oscillation_rate']:.1%}")
    return "\n".join(lines)
