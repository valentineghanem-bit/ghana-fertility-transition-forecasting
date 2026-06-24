#!/usr/bin/env bash
# One-shot analytical pipeline for the Ghana subnational fertility transition study.
set -euo pipefail
python analysis/build_master_panel.py
python analysis/build_district_cross_section.py
python analysis/spatial_lisa.py
python analysis/run_forecast.py
python analysis/compare_forecasts.py
python analysis/district_ml.py
python analysis/make_figures.py
echo "Pipeline complete — see outputs/."
