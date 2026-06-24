# QA REPORT — Project 15 (Phase 9)

## Subnational Fertility Transition & Adolescent RH — Ghana | Date: 2026-06-23 | AIPOCH v6.5

Mode: pre-Phase-10 QA on the 4 present deliverables (Manuscript · Poster · Dashboard · Master data).
GitHub repo (5th deliverable) is Phase 10 — its PRE-COMMIT 9-gate QA runs at push.

## QA-0 — Output completeness
| Deliverable | Present | Notes |
|---|---|---|
| Manuscript | ✓ | manuscript/*.md + *.docx (local-only, git-excluded) |
| Poster | ✓ | poster/Ghana_Fertility_Transition_Poster.html (HI-EI, offline) |
| Dashboard | ✓ | dashboard/Ghana_Fertility_Transition_Dashboard.html (HI-EI, offline) |
| Master data | ✓ | outputs/data/master_panel_wide.csv + district_cross_section_2022_261.csv |
| GitHub repo | ⧖ | Phase 10 |

## QA-1 — Manuscript peer review (epid-council 5-advisor dialectical)
Run and presented inline. Priority findings resolved in the draft:
- Forecasting overclaim softened: "beat/underperforms deep learning" → "the LSTM did not outperform …
  in this implementation" + single-hold-out, not-significance-tested caveat (Abstract, Results §3.3,
  Discussion ¶2, Limitations).
- Adolescent-determinant null reframed as power-limited (n=16) — not a definitive null (Abstract,
  Results §3.5, Discussion ¶3).
- Policy recommendation given an [IMPLEMENTATION GAP] caveat (northern health-system constraints;
  rests on cited evidence, not a cost-effectiveness analysis).
- MIS/DHS survey-design heterogeneity added to Limitations.
- One [UNRESOLVED DEBATE] logged: method-ranking not significance-tested → Diebold-Mariano / rolling-
  origin evaluation flagged as future work (not run).

## QA — Figure/Table cross-reference audit
✓ All 6 figures and all 5 tables cited in-text in BOTH Results AND Discussion (verified by regex).

## QA — Citation integrity (Vancouver)
✓ 28 in-text citations correspond to 28 references; first-appearance order; ascending.

## QA — Cross-artefact value reconciliation
✓ PASS. National mean TFR 4.32, max 6.6 (North East), min 2.9 (Greater Accra), Moran's I 0.68,
LeeCarter TFR RMSE 0.52, LSTM TFR 0.60, BayesHier ASFR 22.7, LOROCV R² 0.70/0.02, 261 districts,
16 regions — all consistent across manuscript ↔ dashboard ↔ poster ↔ source data.

## QA — Submission-grade formatting
✓ Figures: vector PDF + 600-DPI PNG, distinct titles, legends outside the plot, panel labels A/B/…,
colourblind-safe. ✓ Tables: three-line/booktabs, right-aligned numerics.

## VERDICT
**CONDITIONAL PASS** — analytical and manuscript deliverables pass QA. Remaining gate: Phase 10
(14-section README + /github-publish PRE-COMMIT 9-gate + push). Manuscript is never pushed (Tenet 20).
