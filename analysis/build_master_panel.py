"""
build_master_panel.py — Project 15 PHASE 3 (data wrangling).
Assemble the harmonised 16-region x 9-wave MULTI-INDICATOR master panel using the verified
ancestry rule in harmonize_regions.py (NOT the discredited LevelRank==1.0 filter).

Outputs:
  data/processed/master_panel_long.csv   (region, year, indicator, value, unit, source_label,
                                           inherited, survey_type, dt_years)
  outputs/data/master_panel_wide.csv      (one row per region-year, one column per indicator)
  data/processed/master_panel_datalog.md  (/datalog provenance + per-indicator completeness)

Structural missingness is represented as NaN (e.g., unmet need pre-1993, education pre-1993,
contraception in MIS years) — never imputed before an indicator's documented introduction (Tenet 4).
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from harmonize_regions import load_dhs_region_long, harmonize_indicator, WAVES, ANCESTRY  # noqa: E402

RAW = "data/raw"
MIS_WAVES = {2016, 2019}  # Malaria Indicator Surveys — sparser modules

# (short_name, source_csv, exact source indicator string, unit, theme)
REGISTRY = [
    ("tfr_15_49",          "fertility-rates",            "Total fertility rate 15-49",                                   "children/woman", "fertility"),
    ("asfr_15_19",         "fertility-rates",            "Age specific fertility rate: 15-19",                           "births/1000",    "adolescent"),
    ("mcpr_all_women",     "fp2020",                     "Current use of any modern method of contraception (all women)","percent",        "contraception"),
    ("mcpr_married",       "fp2020",                     "Married women currently using any modern method of contraception","percent",     "contraception"),
    ("any_contra_married", "mics-indicators",            "Married women currently using any method of contraception",    "percent",        "contraception"),
    ("unmet_need",         "mics-indicators",            "Unmet need for family planning",                               "percent",        "contraception"),
    ("edu_secondary_women","select-education-indicators","Women with secondary or higher education",                     "percent",        "covariate"),
    ("age_first_birth",    "dhs-mobile",                 "Median age at first birth for women age 25-49",                "years",          "covariate"),
    ("age_first_marriage", "dhs-mobile",                 "Median age at first marriage [Women]: 25-49",                  "years",          "covariate"),
]

DT = {w: (w - WAVES[i - 1] if i else None) for i, w in enumerate(WAVES)}  # years since previous wave


def build():
    long_frames, datalog = [], []
    loaded = {}
    for short, csv, indicator, unit, theme in REGISTRY:
        path = f"{RAW}/{csv}_subnational_gha.csv"
        if path not in loaded:
            loaded[path] = load_dhs_region_long(path)
        panel = harmonize_indicator(loaded[path], indicator)
        panel["indicator"], panel["unit"], panel["theme"], panel["source_csv"] = short, unit, theme, csv
        long_frames.append(panel)
        present_waves = sorted(panel.loc[panel["value"].notna(), "year"].unique())
        datalog.append({
            "indicator": short, "source_csv": csv, "source_indicator": indicator, "unit": unit,
            "waves_present": len(present_waves), "first_wave": present_waves[0] if present_waves else None,
            "cells_filled": int(panel["value"].notna().sum()), "cells_total": len(panel),
            "inherited_cells": int(panel["inherited"].sum()),
        })

    long = pd.concat(long_frames, ignore_index=True)
    # DHS Value column reads as object (numeric strings); coerce. Report any non-numeric loss.
    before = long["value"].notna().sum()
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    coerced_out = int(before - long["value"].notna().sum())
    if coerced_out:
        print(f"WARNING: {coerced_out} non-numeric value(s) coerced to NaN — inspect before trusting panel")
    long["survey_type"] = long["year"].map(lambda y: "MIS" if y in MIS_WAVES else "DHS")
    long["dt_years"] = long["year"].map(DT)
    long = long[["region", "year", "indicator", "value", "unit", "theme", "source_csv",
                 "source_label", "inherited", "survey_type", "dt_years"]]

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/data", exist_ok=True)
    long.to_csv("data/processed/master_panel_long.csv", index=False)

    wide = long.pivot_table(index=["region", "year"], columns="indicator", values="value",
                            aggfunc="first").reset_index()
    wide.to_csv("outputs/data/master_panel_wide.csv", index=False)

    dl = pd.DataFrame(datalog)
    with open("data/processed/master_panel_datalog.md", "w", encoding="utf-8") as fh:
        fh.write("# /datalog — Project 15 master panel\n\n")
        fh.write("Source: DHS StatCompiler subnational exports (Ghana). Frame: 16 regions (2022) x 9 waves.\n")
        fh.write("Harmonisation: analysis/harmonize_regions.py (verified ancestry rule). Extracted 2026-06-22.\n")
        fh.write("Structural missingness = NaN; never imputed before an indicator's first documented wave (Tenet 4).\n\n")
        cols = list(dl.columns)
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, r in dl.iterrows():
            fh.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    return long, wide, dl


if __name__ == "__main__":
    long, wide, dl = build()
    print("master_panel_long.csv rows:", len(long), "| indicators:", long['indicator'].nunique())
    print("master_panel_wide.csv shape:", wide.shape, "(expect 144 rows = 16x9)")
    print()
    print(dl.to_string(index=False))
    print("\nfabrication check: rows with value but no source_label =",
          int(((long['value'].notna()) & (long['source_label'].isna())).sum()), "(must be 0)")
