# Ablations

Each subdirectory is a self-contained ablation — config (if applicable), predictions, results, run logs, and a `README.md` with motivation, setup, commands, results, and verdict. Each directory is designed to lift directly into the paper appendix.

## Index

| #  | Ablation | Question | Type | Cost (wall) | Headline |
|----|----------|----------|------|-------------|----------|
| A1 | [b_l3_temp07](b_l3_temp07/) | Of the +12.12pp B-L3 gain, how much is adapters vs temperature? | new inference | dev ~25m + test ~110m | **Adapters explain ~90% test, ~100% dev**; temperature drop adds ~1.3pp test |
| A2 | [b_l3_loo](b_l3_loo/) | Are all 3 B-L3 specialists pulling weight? | re-aggregation | seconds | **All 3 load-bearing**: general −5.29, contrastive −4.78, ambiguity −2.01 (test, conservative) |
| A3 | [c_l3_truncation](c_l3_truncation/) | Does each depth layer add value? | re-aggregation | seconds | **Layer 1 alone beats canonical L1+L2+L3 by +1.6 dev / +2.1 test on C-L3**; L2 net-zero, L3 net-harmful (see A13 for cross-config picture) |
| A4 | [bootstrap_ci](bootstrap_ci/) | What are the 95% CIs on per-language Macro-F1? | analysis | ~80 min | Per-language ±2-4pp test, ±5-8pp dev. **B-L3 vs B-L2 (+12.12pp) is statistically real**; B-L3 vs C-L3 (+2.83pp) is borderline |
| A5 | [aggregation_strategy](aggregation_strategy/) | Does the aggregation rule matter? | re-aggregation | seconds | **`confidence_max` lifts B-L1 +3.73 / B-L2 +4.41 over canonical 2-of-3 majority** (free post-hoc gain); B-L3 prefers `any_of_3` (+1.13) |
| A6 | [c_l3_skip_l2](c_l3_skip_l2/) | Is C-L3 Layer 2 actively harmful? | new inference | ~2.5h | Skip-L2 ≈ canonical (+0.04pp test). With A3, **L3 itself is the harmful component** — destroys ~2pp of L1's gain |
| A7 | [b_l3_minimal_prompt](b_l3_minimal_prompt/) | How much of B-L3 is the engineered prompt vs adapters? | new inference | ~2.5h | Minimal prompt costs **−2.71pp test**; adapters retain ~78% of gap over B-L2 |
| A8 | [b_l3_no_system_prompt](b_l3_no_system_prompt/) | Are adapters robust to dropping the system prompt entirely? | new inference | ~2.5h | No-system costs **−2.84pp test ≈ A7**. Adapters need no system role; entire prompt contribution = reasoning bank content |
| A9 | [b_l3_per_agent_prompt](b_l3_per_agent_prompt/) | Do per-agent prompts amplify adapter specialization? | new inference | ~2.5h | **Flat (+0.08 dev / −0.12 test)**. Adapters already encode role differentiation; explicit prompts redundant |
| A10 | [b_l3_seed_stability](b_l3_seed_stability/) | Is B-L3 seed-stable independently of A1? | new inference | ~7.5h (3 seeds) | **Deferred** — optional robustness check |
| A11 | [width_loo_l1_l2](width_loo_l1_l2/) | Do B-L1/B-L2 show the same redundancy pattern as B-L3? | re-aggregation | seconds | All 3 width levels show non-trivial agent specialization; B-L3 has the largest spread (3.28pp range), B-L1 the smallest (1.98pp) |
| A12 | [l1_l2_prompt_variants](l1_l2_prompt_variants/) | Is B-L1 prompt-design sensitive (no bank / no policy / permissive)? | new inference | ~40 min (3 variants × 4 langs, dev only) | **Ready to launch** — script written, awaiting GPU availability |
| A13 | [c_l1_l2_truncation](c_l1_l2_truncation/) | Does the "L1 alone wins" finding generalize from C-L3 to C-L1/C-L2? | re-aggregation | seconds | **No — C-L3-specific.** For C-L1 (Δ −2.01pp test) and C-L2 (Δ −3.21pp), the full chain strictly helps. The harmful behaviour is the interaction of trained-aggression specialists with inter-layer conditioning, not the depth structure itself |

## Headline Findings

1. **B-L3 wins for the right reason - adapters, not co-varying factors.** A1 (temperature) and A7+A8 (prompt) both bracket the contribution of the two co-varying factors and pin ≥77% of the gap over B-L2 on the LoRA adapters under every framing.
2. **The depth pipeline is suboptimal *only* at L3 - and not because of the chain itself.** A3 shows C-L3 L1-alone (52.11 test) > canonical L1→L2→L3 (50.00). A6 confirms removing L2 changes nothing (50.04). **A13 shows this pattern is C-L3-specific**: for C-L1 the full chain helps (Δ +2.01pp test over L1-alone), for C-L2 it helps more (+3.21pp). The pathology is the interaction of trained-aggression specialist adapters with inter-layer conditioning, not the depth structure. The optimal C-L3 operating point is Layer 1; the optimal C-L1/C-L2 point is the full chain.
3. **Aggregation choice is a free lever for prompt-only configs.** A5 shows `confidence_max` adds +3.7-4.4pp test on B-L1 and B-L2 with no inference cost. B-L3 is rule-insensitive at the average but rule-sensitive per-language.
4. **Adapters and prompts are not stackable.** A9 shows per-agent prompts add zero on top of trained adapter specialization. The two diversity sources do not compose.
5. **Per-language asymmetry is a real phenomenon, not noise.** A7+A8+A5 all show pcm/tat/chn behave opposite to mar/ptmz/ind under prompt and aggregation perturbations. Per-language adaptive prompting/aggregation is the natural follow-up.
6. **All 3 agents in B-L3 are load-bearing** (A2: ≥2pp drop from removing any one) and the same is true for B-L1/B-L2 (A11: ≥1pp drop from removing any one). Diversity is real at every level.
7. **Per-language CIs are tight enough to validate the headline rankings** (A4: ±2-4pp test). B-L3 vs B-L2 (+12.12pp) is statistically significant; C-L2 vs C-L1 (+1.19pp) is not.

## Directory layout convention

```
ablation/<name>/
├── README.md          # motivation / setup / commands / results / verdict
├── config.yaml        # YAML used (if a new config is needed)
├── scripts/           # any ablation-specific scripts (re-aggregation, analysis)
├── predictions/       # b_<topology>_<level>_<lang>_<split>.jsonl
├── results/           # *_results.csv and *_cost.json
└── run_{dev,test}.logs
```
