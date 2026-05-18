# Experiment Run Logs

One file per experimental condition. Records what was run, key results, known issues, and decisions made.

## Files

| File | Condition | Status |
|------|-----------|--------|
| [b_l1_dev.md](b_l1_dev.md) | B-L1 Width Stochastic — dev | Done ✓ (Run 2 with banks, 2026-04-03) |
| [b_l2_dev.md](b_l2_dev.md) | B-L2 Width Designed — dev | Done |
| [b_l3_dev.md](b_l3_dev.md) | B-L3 Width Learned (LoRA) — dev | Done ✓ best dev result |
| [b_l1_test.md](b_l1_test.md) | B-L1 Width Stochastic — test | Done ✓ (2026-04-06) |
| [b_l2_test.md](b_l2_test.md) | B-L2 Width Designed — test | Done ✓ (2026-04-17) |
| [b_l3_test.md](b_l3_test.md) | B-L3 Width Learned (LoRA) — test | Done ✓ (2026-04-19) **best test result** |
| [c_l1_dev.md](c_l1_dev.md) | C-L1 Depth Stochastic — dev | Done ✓ (2026-04-20) |
| [c_l2_dev.md](c_l2_dev.md) | C-L2 Depth Designed — dev | Done ✓ (2026-04-22) |
| [c_l3_dev.md](c_l3_dev.md) | C-L3 Depth Learned (LoRA) — dev | Done ✓ (2026-04-25) |
| [c_l1_test.md](c_l1_test.md) | C-L1 Depth Stochastic — test | Done ✓ (2026-04-26, --batch) |
| [c_l2_test.md](c_l2_test.md) | C-L2 Depth Designed — test | Done ✓ (2026-04-27, --batch) |
| [c_l3_test.md](c_l3_test.md) | C-L3 Depth Learned (LoRA) — test | Done ✓ (2026-04-28, --batch) |


**Ablations** live under [`../ablation/`](../ablation/) (separate directory so they can be lifted directly into the paper appendix):
- [`ablation/b_l3_temp07/`](../ablation/b_l3_temp07/) — B-L3 at temp=0.7 vs canonical 0.3. Done ✓ (2026-04-29). Adapters explain ~90-100% of the gain over B-L2; temperature is a small contributor.

---

## Dev Set Results Summary

| Condition | Avg Macro-F1 | vs CoT baseline | Notes |
|-----------|-------------|----------------|-------|
| CoT single-pass (thesis) | 48.47% | — | Reference floor |
| CoT+SC k=7 (thesis) | 48.81% | +0.34pp | Reference ceiling |
| B-L1 (Width/Stochastic) Run 2 | 42.00% | -6.47pp | reasoning banks loaded (canonical) |
| B-L2 (Width/Designed) | 42.40% | -6.07pp | designed adds only +0.40pp over stochastic |
| **B-L3 (Width/Learned)** | **53.61%** | **+5.14pp** | **best dev result** |
| C-L1 (Depth/Stochastic) | 44.61% | -3.86pp | depth > width at stochastic level (+2.61pp over B-L1) |
| C-L2 (Depth/Designed) | 45.69% | -2.78pp | +1.08pp over C-L1; critic doubles revision rate |
| C-L3 (Depth/Learned) | 50.74% | +2.27pp | beats CoT; -2.87pp behind B-L3 (depth-LoRA suboptimal) |

---

## Test Set Results Summary

| Condition | Avg Macro-F1 | vs CoT baseline | dev→test gap | Notes |
|-----------|-------------|----------------|--------------|-------|
| B-L1 (Width/Stochastic) | 40.66% | -7.81pp | -1.34pp | 2026-04-06; -1.34pp vs dev |
| B-L2 (Width/Designed)   | 40.71% | -7.76pp | -1.69pp | 2026-04-17; designed adds only +0.05pp over stochastic on test |
| **B-L3 (Width/Learned)** | **52.83%** | **+4.36pp** | **-0.78pp** | **2026-04-19; best test result, beats both thesis baselines** |
| C-L1 (Depth/Stochastic) | 43.88% | -4.59pp | -0.73pp | 2026-04-26 (--batch); +3.22pp over B-L1 (depth wins at stochastic) |
| C-L2 (Depth/Designed)   | 45.07% | -3.40pp | -0.62pp | 2026-04-27 (--batch); +1.19pp over C-L1; smallest dev→test gap |
| C-L3 (Depth/Learned)    | 50.00% | +1.53pp | -0.74pp | 2026-04-28 (--batch); beats CoT; -2.83pp behind B-L3 |

**Test ranking**: B-L3 (52.83) > C-L3 (50.00) > C-L2 (45.07) > C-L1 (43.88) > B-L2 (40.71) ≈ B-L1 (40.66)

**Same ranking on dev and test** — no reordering. Dev→test gap is small (≤2pp) for every config, confirming all configurations generalise reliably.

---

## Headline findings (test set)

1. **Learned diversity is the dominant factor** — adding LoRA adapters lifts macro-F1 by +12pp (width) and +6pp (depth) over the same topology with prompt-only diversity. **Confirmed by temperature ablation (2026-04-29):** at matched temperature 0.7 the LoRA-adapter B-L3 still scores 51.57% test / 53.59% dev (vs B-L2 40.71% / 42.40%), so adapters explain ~90% of the test gain and ~100% of the dev gain — temperature is a small contributor at best.
2. **Width topology wins when adapters are used** — B-L3 (52.83%) beats C-L3 (50.00%) by 2.83pp. The depth Layer 2 (error_corrector) adapter introduces 4.2× more errors than corrections (9.36% vs 2.22%); Layer 1 alone is stronger than the full C-L3 pipeline.
3. **Depth wins when no adapters are used** — C-L1 (+3.22pp over B-L1) and C-L2 (+4.36pp over B-L2) confirm sequential conditioning > parallel sampling at the prompt-only diversity levels.
4. **Designed prompt diversity adds essentially nothing in width** (B-L2 vs B-L1: +0.05pp on test) but adds +1.19pp in depth (C-L2 vs C-L1) — role differentiation has more leverage in a sequential pipeline than 3 parallel analytical lenses.
5. **Two configs beat both thesis baselines** — B-L3 (+4.36pp over CoT, +4.02pp over CoT+SC k=7) and C-L3 (+1.53pp over CoT).
6. **Low-resource languages benefit most from learned adapters** — vmw (B-L1: 4.39% → B-L3: 19.16%), zul (13.59% → 27.05%), ptmz (38.60% → 56.40%). Adapters recover signal that prompt diversity cannot.
7. **vmw and zul remain the hardest languages** — even with B-L3 they top out at 19% and 27% respectively. The base model has near-zero competence for Makua and Zulu.

---

## Quality issues found in test results

- **B-L3 ptmz contrastive_specialist parse-failure spike** — 85/418 (20%) test predictions failed to parse JSON for the contrastive adapter on Mozambican Portuguese only. Same pattern on dev (18/138 = 13%). Other adapters and other languages are fine. Majority vote with the other 2 agents still produces a valid prediction, so the per-language F1 (56.40%) is not catastrophically affected — but the contrastive_specialist's per-agent score on ptmz (49.77%) is depressed by the all-zero fallbacks.
- All other configs/languages: parse-failure rate < 1.5%; most languages 0%.
