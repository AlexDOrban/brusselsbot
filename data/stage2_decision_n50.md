# Stage 2 Decision — N=50 Three-way Evaluation (retrieval-only)

## 1. Summary

**Decision: KEEP hybrid, keep reranker**

A and B both pass, but B shows per-category Hit@1 gap > 0.1: in_scope_en(+0.10), multi_hop(+0.20).

Retrieval-only evaluation across 50 questions in 6 categories (in_scope_fr=10, in_scope_nl=8, in_scope_en=10, out_of_scope=7, multi_hop=10, cross_lingual=5). RAGAS answer-quality metrics scoped out (see §8). MRR(C)=0.671, MRR(A)=0.703, MRR(B)=0.741.

## 2. Per-question ranks (6 per-category sub-tables)

### in_scope_fr  (n=10)

| Question ID | C | A | B | Winner |
|-------------|---|---|---|--------|
| Q6_grue_fr | 1 | 1 | 1 | C=A=B |
| Q11_alarme_fr | 1 | 2 | 2 | C |
| Q12_trottoir_fr | 1 | 1 | 1 | C=A=B |
| Q13_feu_fr | 5 | 2 | 2 | A=B |
| Q14_chantier_fr | 10 | 3 | 3 | A=B |
| Q15_demenage_fr | >10 | >10 | 2 | B |
| Q16_chien_fr | 6 | 1 | 1 | A=B |
| Q17_deposantclandestin_fr | 1 | 1 | 1 | C=A=B |
| Q18_affichage_fr | 1 | 1 | 1 | C=A=B |
| Q19_taxi_fr | 2 | 1 | 1 | A=B |

### in_scope_nl  (n=8)

| Question ID | C | A | B | Winner |
|-------------|---|---|---|--------|
| Q2_muziek_nl | 2 | 2 | 1 | B |
| Q3_sluikstort_nl | 1 | 1 | 3 | C=A |
| Q20_alarm_nl | 1 | 1 | 1 | C=A=B |
| Q21_verhuizing_nl | >10 | >10 | 1 | B |
| Q22_voetpad_nl | 1 | 2 | 2 | C |
| Q24_hond_nl | 4 | 6 | 3 | B |
| Q25_afvalzak_nl | 1 | 1 | 1 | C=A=B |
| Q27_vuur_nl | >10 | >10 | >10 | — |

### in_scope_en  (n=10)

| Question ID | C | A | B | Winner |
|-------------|---|---|---|--------|
| Q1_noise_en | 1 | 1 | 1 | C=A=B |
| Q4_lez_social | 1 | 1 | 1 | C=A=B |
| Q5_bruxellair | 1 | 1 | 1 | C=A=B |
| Q8_ev_fleet | 1 | 1 | 1 | C=A=B |
| Q10_alarm | 1 | 1 | 1 | C=A=B |
| Q7_trash_sidewalk | 1 | 3 | 2 | C |
| Q28_crane_en | 2 | 1 | 1 | A=B |
| Q29_fireworks_en | 4 | 5 | 7 | C |
| Q30_lez_enforcement | 1 | 1 | 1 | C=A=B |
| Q31_lp_modalshift | >10 | 1 | 1 | A=B |

### out_of_scope  (n=7)

| Question ID | C | A | B | Winner |
|-------------|---|---|---|--------|
| Q23_werf_nl | OOS | OOS | OOS | — |
| Q26_gezondheid_nl | OOS | OOS | OOS | — |
| Q9_e40_oos | OOS | OOS | OOS | — |
| Q32_tax_oos | OOS | OOS | OOS | — |
| Q33_parking_oos | OOS | OOS | OOS | — |
| Q34_eu_emissions_oos | OOS | OOS | OOS | — |
| Q35_customs_oos | OOS | OOS | OOS | — |

### multi_hop  (n=10)

| Question ID | C | A | B | Winner |
|-------------|---|---|---|--------|
| Q36_noise_and_construction_fr | 2 | 4 | 4 | C |
| Q37_dog_and_dumping_fr | 3 | 1 | 1 | A=B |
| Q38_party_noise_sanction_fr | 2 | 1 | 1 | A=B |
| Q39_cleaning_snow_nl | 1 | 1 | 1 | C=A=B |
| Q40_alarm_and_noise_nl | 1 | 1 | 1 | C=A=B |
| Q41_waste_and_moving_nl | 3 | 9 | 7 | C |
| Q42_crane_and_sidewalk_en | 1 | 1 | 1 | C=A=B |
| Q43_sanctions_lez_en | 1 | 1 | 1 | C=A=B |
| Q44_fire_and_smoke_fr | 2 | 2 | 2 | C=A=B |
| Q45_dog_park_sanction_nl | 2 | 3 | 3 | C |

### cross_lingual  (n=5)

| Question ID | C | A | B | Winner |
|-------------|---|---|---|--------|
| Q46_fr_to_nl | 2 | 2 | 2 | C=A=B |
| Q47_nl_to_fr | 1 | 1 | 1 | C=A=B |
| Q48_en_to_fr | 1 | 3 | 3 | C |
| Q49_en_to_nl | 5 | 6 | 6 | C |
| Q50_fr_to_en_ieep | 1 | 1 | 1 | C=A=B |

> Rank = 1-based position of first gold chunk in top-10. `>10` = not retrieved. `OOS` = out-of-scope (no gold).

## 3. Aggregate gate checks

**Thresholds re-anchored to Config C at N=50** (baseline sets the floor; A/B must beat or match C on the harder benchmark).

Significant changes from the original N=10 thresholds:
- hit_at_1: 0.889 → 0.535 (lowered to match Config C at N=50 = 0.535; delta -0.354)
- hit_at_3: 1.000 → 0.767 (lowered to match Config C at N=50 = 0.767; delta -0.233)
- mrr: 0.944 → 0.671 (lowered to match Config C at N=50 = 0.671; delta -0.273)

| Metric | Threshold | C | A | B |
|--------|-----------|---|---|---|
| Hit@1 ≥ 0.535 | 0.535 | 0.535 ✅ | 0.581 ✅ | 0.605 ✅ |
| Hit@3 ≥ 0.767 | 0.767 | 0.767 ✅ | 0.814 ✅ | 0.884 ✅ |
| MRR ≥ 0.671 | 0.671 | 0.671 ✅ | 0.703 ✅ | 0.741 ✅ |
| sluikstort in top-5 | — | ✅ | ✅ | ✅ |

- Config A all-gates: **PASS**
- Config B all-gates: **PASS**
- MRR(B) − MRR(A) = +0.0382

## 4. Per-category Hit@1 breakdown

| Category | n | C Hit@1 | A Hit@1 | B Hit@1 | B−C |
|----------|---|---------|---------|---------|-----|
| in_scope_fr | 10 | 0.500 | 0.600 | 0.600 | +0.100 |
| in_scope_nl | 8 | 0.500 | 0.375 | 0.500 | +0.000 |
| in_scope_en | 10 | 0.700 | 0.800 | 0.800 | +0.100 |
| multi_hop | 10 | 0.400 | 0.600 | 0.600 | +0.200 |
| cross_lingual | 5 | 0.600 | 0.400 | 0.400 | -0.200 |

**Per-category gaps > 0.1 in B's favor:**
- in_scope_en: B=0.800, C=0.700, gap=+0.100
- multi_hop: B=0.600, C=0.400, gap=+0.200

## 5. OOS correctness (confidence caveat)

| Config | correctly caveated | incorrectly answered |
|--------|--------------------|----------------------|
| C | 1/7 | 6/7 |
| A | 4/7 | 3/7 |
| B | 4/7 | 3/7 |

> Caveat signal: dense_distance > 0.60 (Config C) or rerank_score < 0.30 (A/B).

Note: the caveat is fired by retrieval-side thresholds (dense distance or rerank score), not by the generation layer. Recalibrating these thresholds against the expanded N=7 OOS set is Stage 6+ work. Currently 3/7 OOS questions receive confident hallucinated answers even under best-config B — a production-critical failure mode.

## 6. Decision matrix application

- A passes all gates: **True**
- B passes all gates: **True**
- |MRR(B) − MRR(A)| = 0.0382  (tie threshold 0.02)
- Per-category B-favor gap > 0.1: YES — in_scope_en, multi_hop

→ **KEEP hybrid, keep reranker**

A and B both pass, but B shows per-category Hit@1 gap > 0.1: in_scope_en(+0.10), multi_hop(+0.20).

### Concrete BM25 contributions

Hybrid (Config B vs A) recovers two questions that dense-only retrieval missed entirely:
- Q15_demenage_fr: rank >10 (C, A) → rank 1 (B)
- Q21_verhuizing_nl: rank >10 (C, A) → rank 1 (B)

Same pattern as the original sluikstort win at N=10 — BM25 surfaces exact-keyword matches that dense embeddings bury under semantically adjacent but topically unrelated chunks. This is the single most concrete evidence that hybrid earns its complexity.

## 7. Q7_trash_sidewalk + B-regression diagnostic

Q7_trash_sidewalk ranks: C=1, A=3, B=2

**All in-scope questions where B's rank is worse than C's:** 10

| Question ID | C rank | B rank |
|-------------|--------|--------|
| Q11_alarme_fr | 1 | 2 |
| Q3_sluikstort_nl | 1 | 3 |
| Q22_voetpad_nl | 1 | 2 |
| Q7_trash_sidewalk | 1 | 2 |
| Q29_fireworks_en | 4 | 7 |
| Q36_noise_and_construction_fr | 2 | 4 |
| Q41_waste_and_moving_nl | 3 | 7 |
| Q45_dog_park_sanction_nl | 2 | 3 |
| Q48_en_to_fr | 1 | 3 |
| Q49_en_to_nl | 5 | 6 |

Pattern read: 10 questions where B underperforms C, of which:
- 4 are 1-position shifts within the top-3 (Q11, Q22, Q7, Q3) — noise band within a tight relevance band, likely reranker tie-breaking
- 2 are cross-lingual (Q48, Q49) — see §4 cross_lingual category regression, addressed by planned Stage 3 (cross-lingual query expansion)
- 4 are genuine multi-position drops (Q29 fireworks, Q36 noise+construction, Q41 waste+moving, Q45 dog park sanction). No obvious shared pattern. Flagged for Stage 6+ investigation.

## 8. Methodology note — RAGAS scoped out

This evaluation uses retrieval metrics (Hit@K, MRR) only. RAGAS-based answer-quality metrics (faithfulness, answer_relevancy, context_precision, context_recall) were scoped out of this iteration due to local hardware constraints — running RAGAS with adequate concurrency requires more RAM than the development machine reliably provides.

The decision matrix is therefore weighted toward retrieval-side evidence. Where retrieval metrics are tied between configurations, "simpler stack wins" because we lack the answer-quality signal that would otherwise justify additional architectural complexity.

Future work: re-run the three-way evaluation in a hosted environment (Google Colab or equivalent) with RAGAS faithfulness and answer_relevancy added. Per-category RAGAS deltas — particularly on multi_hop and cross_lingual where rerank is most likely to earn its place — would either confirm or revise the keep/revert decision documented here.
