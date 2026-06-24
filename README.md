# Subnational fertility transition and adolescent reproductive health in Ghana: regional trajectories 1988–2022 and probabilistic forecasting to 2030

[![CI](https://github.com/valentineghanem-bit/ghana-fertility-transition-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/valentineghanem-bit/ghana-fertility-transition-forecasting/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/) [![R 4.3+](https://img.shields.io/badge/R-4.3+-blue.svg)](https://www.r-project.org/) [![ORCID](https://img.shields.io/badge/ORCID-0009--0002--8332--0220-green.svg)](https://orcid.org/0009-0002-8332-0220)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
**Affiliation:** Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**Reporting standard:** STROBE · RECORD-Spatial · TRIPOD+AI
**Date:** 2026
**Status:** Manuscript in preparation

## 1. Abstract

Ghana's fertility decline is widely described as having slowed, yet evidence for a true national "stall" is contested and subnational trajectories for the country's current 16-region structure have not been assembled in one harmonised series. Using nine DHS/MIS waves (1988–2022) harmonised to the 2022 16-region frame and a 261-district 2022 cross-section, this study characterises the regional fertility and adolescent-reproductive-health transition, maps its spatial structure, and produces probabilistic forecasts to 2030. National TFR fell from ~6.4 to 4.32 (2022) but the decline is shallow and persistent in the north (North East 6.6 vs Greater Accra 2.9). Demography-native forecasters (Bayesian hierarchical, Lee-Carter) were not outperformed by a time-aware LSTM benchmark on these short series. District structural deprivation predicted overall fertility out-of-region (LOROCV R² = 0.70) but not adolescent fertility (R² ≈ 0.02).

## 2. Research Question & Aims

How has subnational fertility and adolescent reproductive health changed across Ghana's regions from 1988 to 2022, how does it cluster spatially in 2022, and what are the credible regional trajectories to 2030? Aims: (i) assemble a harmonised 16-region 1988–2022 panel; (ii) characterise 2022 spatial structure across 261 districts; (iii) develop and honestly benchmark probabilistic forecasts to 2025/2030; (iv) examine district structural determinants of fertility.

## 3. Methods Summary

| Method | Tool | Purpose |
|--------|------|---------|
| Region harmonisation (ancestry rule) | Python (pandas/numpy) | Collapse DHS hierarchy to the 16-region 2022 frame |
| Bayesian hierarchical forecast | Python (numpy, Gibbs sampler) | Partial-pooling probabilistic TFR/ASFR forecasts + credible intervals |
| Lee-Carter | Python (numpy SVD) | Coherent age-schedule forecast (TFR + adolescent ASFR) |
| Statistical baselines (RWD, log-linear, linear) | Python (numpy) | Year-aware benchmark forecasters |
| Time-aware LSTM (benchmark) | Python (numpy, BPTT) | Deep-learning comparator (gradient-checked) |
| Global Moran's I + LISA | Python (numpy, sklearn KNN) | Spatial autocorrelation / clusters (261 districts) |
| Global Moran's I + LISA (reference) | R 4.3 (spdep) | Independent spatial-autocorrelation cross-check |
| Gradient boosting + permutation importance (LOROCV) | scikit-learn | District structural determinants of fertility |

## 4. Data Sources

| Source | Variables | Year | Access |
|--------|-----------|------|--------|
| DHS Program StatCompiler (DHS/MIS) | TFR, ASFR, contraception, unmet need | 1988–2022 (9 waves) | Public ([dhsprogram.com](https://dhsprogram.com)) |
| Ghana Population & Housing Census | District poverty, illiteracy, uninsurance, age–sex structure | 2021 | Public (Ghana Statistical Service) |
| Ghana district GeoJSON | District geometry (260 polygons; frame = 261 districts) | — | Public |

## 5. Key Findings

| Metric | Value |
|--------|-------|
| National mean TFR (2022) | 4.32 (from ~6.4 in 1988) |
| North–South TFR gap (2022) | 2.3× (North East 6.6 vs Greater Accra 2.9) |
| Region-level TFR spatial autocorrelation | Global Moran's I = 0.68 (p < 0.001) |
| District illiteracy spatial autocorrelation | Global Moran's I = 0.76 (p < 0.001) |
| Best forecaster (backtest) | Lee-Carter (TFR RMSE 0.52) / Bayesian hierarchical (ASFR RMSE 22.7); LSTM did not outperform |
| Projected national TFR (2030) | ≈ 3.7–3.9 (north–south gap intact) |
| District determinants of fertility (LOROCV R²) | TFR 0.70; adolescent ASFR ≈ 0.02 |
| Scope | 16 regions · 261 districts · 9 survey waves · 28 references |

## 6. Repository Structure

```
ghana-fertility-transition-forecasting/
  analysis/          harmonize_regions, build_master_panel, build_district_cross_section,
                     spatial_lisa, bayes_hier, lee_carter, lstm_numpy, run_forecast,
                     compare_forecasts, district_ml, make_figures, md_to_docx (Python)
                     spatial_diagnostics.R (R/spdep spatial cross-check)
  data/raw/          DHS/MIS subnational CSVs, Census Master Sheet, district GeoJSON
  data/processed/    master_panel_long.csv, district_cross_section_2022_261.csv, datalog
  outputs/data/      master_panel_wide.csv, forecasts_2025_2030.csv
  outputs/tables/    backtest metrics, Moran's I, LISA clusters, district-ML importance
  outputs/figures/   fig1–fig6 (300/600-DPI PNG)
  dashboard/         Ghana_Fertility_Transition_Dashboard.html (interactive, offline)
  poster/            Ghana_Fertility_Transition_Poster.html (A0, print-ready)
  evidence/          section-stratified evidence extractions (28-source bank)
  qa/                QA report + badge
  tests/             pytest suite
  README.md  CITATION.cff  LICENSE  requirements.txt  run_all.sh  Dockerfile  .github/workflows/ci.yml
```

## 7. Reproducibility

### 7.1 Requirements
Python 3.12 with numpy, scipy, pandas, scikit-learn, matplotlib, python-docx (see `requirements.txt`) — all forecasting/ML models are pure-NumPy (no GPU). R 4.3+ with `spdep` is optional, used only by `analysis/spatial_diagnostics.R` to independently cross-check the spatial statistics. A `Dockerfile` provides the full Python + R environment (`docker build -t fertility-ghana .`).

### 7.2 Clone & install
```bash
git clone https://github.com/valentineghanem-bit/ghana-fertility-transition-forecasting.git
cd ghana-fertility-transition-forecasting
pip install -r requirements.txt
```

### 7.3 Run the analytical pipeline
```bash
bash run_all.sh
# or step-by-step:
python analysis/build_master_panel.py
python analysis/build_district_cross_section.py
python analysis/spatial_lisa.py
python analysis/run_forecast.py && python analysis/compare_forecasts.py
python analysis/district_ml.py
python analysis/make_figures.py
```

### 7.4 Run the test suite
```bash
pytest tests/ -q
```

### 7.5 Interactive dashboard (no server required)
The dashboard is a single self-contained offline HTML file (inline ECharts) — no Dash server or network needed. Open it directly (see 7.6).

### 7.6 Open the static HTML dashboard
```bash
# macOS
open dashboard/Ghana_Fertility_Transition_Dashboard.html
# Windows
start dashboard/Ghana_Fertility_Transition_Dashboard.html
# Linux
xdg-open dashboard/Ghana_Fertility_Transition_Dashboard.html
```

## 8. Outputs

| Output | Description |
|--------|-------------|
| `outputs/data/master_panel_wide.csv` | Harmonised 16-region × 9-wave indicator panel |
| `outputs/data/forecasts_2025_2030.csv` | Per-region 2025/2030 forecasts (6 methods) + 80% intervals |
| `outputs/tables/forecast_backtest_metrics.csv` | 2022 hold-out RMSE/MAE/coverage by method |
| `outputs/tables/spatial_global_moran_2022.csv` · `lisa_clusters_2022_261.csv` | Spatial statistics |
| `outputs/tables/district_ml_*.csv` | LOROCV performance + permutation importances |
| `outputs/figures/fig1–fig6` | Publication figures (300/600 DPI) |

## 8a. Downloadable Artefacts (HTML)

| Artefact | View on GitHub | Live preview | Direct download (raw HTML) |
|----------|---------------|--------------|---------------------------|
| Interactive dashboard | [View](https://github.com/valentineghanem-bit/ghana-fertility-transition-forecasting/blob/main/dashboard/Ghana_Fertility_Transition_Dashboard.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/ghana-fertility-transition-forecasting/blob/main/dashboard/Ghana_Fertility_Transition_Dashboard.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/ghana-fertility-transition-forecasting/main/dashboard/Ghana_Fertility_Transition_Dashboard.html) |
| Conference poster | [View](https://github.com/valentineghanem-bit/ghana-fertility-transition-forecasting/blob/main/poster/Ghana_Fertility_Transition_Poster.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/ghana-fertility-transition-forecasting/blob/main/poster/Ghana_Fertility_Transition_Poster.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/ghana-fertility-transition-forecasting/main/poster/Ghana_Fertility_Transition_Poster.html) |

> **Tip:** The dashboard works fully offline once downloaded. The poster is print-ready at A0 (841 × 1189 mm).

## 9. Reporting Standard

This study follows the **STROBE** guideline for observational studies. The spatial component follows **RECORD-Spatial**; the prediction/forecasting and machine-learning components follow **TRIPOD+AI**.

## 10. Ethical Statement

The study uses de-identified, publicly available aggregate survey (DHS/MIS) and census data; no individual records were accessed and no ethical approval was required.

## 11. Citation

**APA:**
Ghanem, V. G. (2026). *Subnational fertility transition and adolescent reproductive health in Ghana: regional trajectories 1988–2022 and probabilistic forecasting to 2030.* GitHub. https://github.com/valentineghanem-bit/ghana-fertility-transition-forecasting

**BibTeX:**
```bibtex
@misc{ghanem2026fertilityghana,
  author = {Ghanem, Valentine Golden},
  title  = {Subnational fertility transition and adolescent reproductive health in Ghana: regional trajectories 1988--2022 and probabilistic forecasting to 2030},
  year   = {2026},
  url    = {https://github.com/valentineghanem-bit/ghana-fertility-transition-forecasting}
}
```
A machine-readable citation is provided in `CITATION.cff`.

## 12. License

Code is released under the **MIT License** — see [LICENSE](LICENSE). Outputs and figures: **CC BY 4.0**.

## 13. Author & Contact

**Valentine Golden Ghanem**
Ghana COCOBOD Cocoa Clinic, Accra, Ghana
Email: valentineghanem@gmail.com
ORCID: [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)

## 14. Acknowledgements

The DHS Program (ICF) and the Ghana Statistical Service for open access to the underlying survey and census data.
