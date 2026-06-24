# Evidence — METHODOLOGY (forecasting & irregular-interval handling)

- **Time-aware LSTM (T-LSTM) — primary architecture anchor.** Baytas et al. 2017 (KDD/ACM,
  DOI 10.1145/3097983.3097997) introduce T-LSTM with a Δt time-decay that subspace-decomposes the
  memory cell to handle **non-uniform intervals** between observations — exactly the 5/5/5/5/6/2/3/3-yr
  DHS spacing. Scite: 358 Smart Citations (6 supporting / 1 contrasting / 350 mentioning), 616 citing,
  **no retraction** [SUPPORTING, reliable].
- **LSTM vs ARIMA — SUPPORTING.** LSTM reduces forecast error 84–87% vs ARIMA. — Siami-Namini
  et al. 2018, IEEE ICMLA [abstract-level; body-confirm before quoting the figure].
- **LSTM vs ARIMA — CONTRASTING (critical for our N=9 series).** ARIMA most appropriate for
  **small univariate** datasets; deep learning "not yet at their best" on short series; KNN best at
  medium/long horizons. — Iaousse et al. 2023 [CONTRASTING].
- **Fertility-curve modelling.** Low-parameter SVD / Lee-Carter-style ASFR representation; SSA
  curves flatter/broader. — Pantazis & Clark 2018, *PLoS ONE*, DOI 10.1371/journal.pone.0190574
  (Scite 18 tally, 0 contrasting). Use as a statistical baseline / ASFR-shape reference.

## ML-vs-statistical debate — CRITICAL for an N=9 series (CONTRASTING, C-B resolved)
- **ML systematically dominated by statistical methods** across the M3 competition (1045 series),
  all horizons, at higher compute cost. — Makridakis et al. 2018, *PLoS ONE* [CONTRASTING].
- **"Size matters."** ML underperforms simple statistical methods **only at very low sample size**;
  relative performance improves as N grows. — Cerqueira et al. 2019, *ArXiv* [CONTRASTING; directly
  motivates our N=9 caution].
- Survey across M1–M6 competitions (incl. LSTM/TCN/transformers). — Tjøstheim 2025, *Entropy*.

## Bayesian probabilistic projection — the incumbent comparator
- **UN Bayesian hierarchical TFR projection** (MCMC), validated out-of-sample. — Raftery et al.
  2012, *PNAS*. Extended to account for **uncertainty in past values** (key for high-fertility,
  survey-reliant countries like Ghana). — Liu et al. 2018, *Ann Appl Stat*. **Subnational** Bayesian
  probabilistic projection (county-level), well-calibrated intervals. — Yu et al. 2023, *Demography*.
- IMPLICATION: T-LSTM must be benchmarked against Bayesian/Lee-Carter, not only ARIMA; report
  prediction-interval coverage honestly given S14/S15/S20–S22.

## Spatial methodology precedent (2022 cross-section)
- **Ghana 2022 DHS district-level spatial analysis** of unmet need with Getis-Ord hotspots (north)
  — validates our 2022 spatial stream. — Okyere et al. 2025, *Reprod Health*,
  DOI 10.1186/s12978-024-01935-6. Uganda spatial mCPR (choropleth/network). — Towongo & Kelepile
  2024, *Contracept Reprod Med*, DOI 10.1186/s40834-024-00288-6.

DESIGN IMPLICATION: benchmark T-LSTM against ARIMA / random-walk / exponential-smoothing /
Lee-Carter / Bayesian (Raftery-style); report prediction-interval coverage; the short 9-timestep
series is a stated limitation (S5/S15). Δt is fed as an input feature; MIS-wave (2016/2019) flagged.
