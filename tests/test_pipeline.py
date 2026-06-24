"""Local test suite (data-dependent; not run in CI). Validates key invariants."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))


def test_lstm_gradient():
    from lstm_numpy import gradient_check
    assert gradient_check() < 1e-4


def test_region_frame_is_16():
    from harmonize_regions import ANCESTRY
    assert len(ANCESTRY) == 16


def test_master_panel_complete():
    w = pd.read_csv("outputs/data/master_panel_wide.csv")
    assert w.shape[0] == 144 and w["region"].nunique() == 16  # 16 regions x 9 waves


def test_district_frame_is_261():
    d = pd.read_csv("data/processed/district_cross_section_2022_261.csv")
    assert len(d) == 261 and d["district"].nunique() == 261


def test_forecasts_nonnegative():
    fc = pd.read_csv("outputs/data/forecasts_2025_2030.csv")
    assert (fc[["point", "lo80", "hi80"]] >= 0).all().all()
