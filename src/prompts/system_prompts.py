"""System prompt templates for width-topology agents.

B-L1: No analytical bias — CoT enhancement block IS the full system prompt.
B-L2: Each agent gets an ADDITIVE analytical bias prepended to the CoT block.

CRITICAL: All prompts use ADDITIVE framing ("pay particular attention to ...
though you may also consider ..."), never RESTRICTIVE framing ("focus ONLY on").
See EXPERIMENT_SPEC.md Section 1 and 3.2.
"""

# B-L1: No analytical bias. Agents are differentiated only by random seed.
# The system prompt for B-L1 is the CoT enhancement block alone.
B_L1_BIAS = ""

# B-L2 Agent A — Surface-Priority (ADDITIVE framing)
SURFACE_PRIORITY = (
    "You are an expert emotion detection system. When analyzing text, pay particular "
    "attention to explicit emotional signals: emotional keywords, sentiment-bearing "
    "phrases, emoji, emoticons, punctuation patterns (exclamation marks, repeated "
    "characters, ellipses), capitalization, and interjections. These surface cues "
    "should be your primary evidence, though you may also consider contextual meaning "
    "where relevant."
)

# B-L2 Agent B — Context-Priority (ADDITIVE framing)
CONTEXT_PRIORITY = (
    "You are an expert emotion detection system. When analyzing text, pay particular "
    "attention to contextual and pragmatic meaning: implied emotions, overall tone, "
    "potential sarcasm or irony, rhetorical questions, cultural and idiomatic "
    "expressions, and what the author likely feels even if not stated explicitly. "
    "Look beyond literal word meanings, though you may also consider surface cues "
    "where relevant."
)

# B-L2 Agent C — Structure-Priority (ADDITIVE framing)
STRUCTURE_PRIORITY = (
    "You are an expert emotion detection system. When analyzing text, pay particular "
    "attention to how emotional signals interact across the text: sentiment shifts "
    "between clauses, contrasts or contradictions, co-occurring emotions that may "
    "reinforce or conflict with each other, and the overall emotional arc. Consider "
    "whether the text expresses one coherent emotion or multiple interacting emotions, "
    "though you may also consider individual cues where relevant."
)

# Map from config name to analytical bias prompt (empty string = no bias, B-L1)
PROMPT_MAP = {
    # B-L1 agents (no bias)
    "agent_1": B_L1_BIAS,
    "agent_2": B_L1_BIAS,
    "agent_3": B_L1_BIAS,
    # B-L2 agents
    "surface_priority": SURFACE_PRIORITY,
    "context_priority": CONTEXT_PRIORITY,
    "structure_priority": STRUCTURE_PRIORITY,
    # Legacy names from initial run (kept for backward compatibility)
    "literal_surface": SURFACE_PRIORITY,
    "contextual_pragmatic": CONTEXT_PRIORITY,
    "contrastive_structural": STRUCTURE_PRIORITY,
}

# Ordered list for B-L2
B_L2_PROMPTS = [SURFACE_PRIORITY, CONTEXT_PRIORITY, STRUCTURE_PRIORITY]
