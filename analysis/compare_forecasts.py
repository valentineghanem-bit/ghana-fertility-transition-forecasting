"""
compare_forecasts.py — Project 15 PHASE 5 (reframed). Unified multi-method comparison.
Adds the demography-native methods (Lee-Carter, Bayesian hierarchical) to the existing baseline + LSTM
backtest, and produces the final 2025/2030 forecasts for all methods. The 2022-holdout backtest
(RMSE/MAE + 80% interval coverage) decides the recommended operational forecaster per indicator.

Reuses outputs/tables/forecast_backtest_metrics.csv and outputs/data/forecasts_2025_2030.csv
(baselines + LSTM, already computed) so the slow LSTM ensemble is not retrained.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import bayes_hier  # noqa: E402
import lee_carter as lc  # noqa: E402

PANEL = "outputs/data/master_panel_wide.csv"
IND_MAP = {"tfr_15_49": "tfr", "asfr_15_19": "asfr15"}   # panel column -> lee_carter key
TARGETS = [2025, 2030]
Z = 1.2816


def actual_2022(panel, ind):
    out = {}
    for r in sorted(panel["region"].unique()):
        s = panel[panel["region"] == r].sort_values("year")
        out[r] = float(pd.to_numeric(s[ind], errors="coerce").values[-1])
    return out


def metrics(rows, ind, method):
    e = np.array([p - a for p, a, lo, hi in rows])
    cov = np.mean([lo <= a <= hi for p, a, lo, hi in rows])
    return {"indicator": ind, "method": method,
            "RMSE": round(float(np.sqrt((e ** 2).mean())), 3),
            "MAE": round(float(np.abs(e).mean()), 3),
            "coverage80": round(float(cov), 3)}


def main():
    panel = pd.read_csv(PANEL)
    regions = sorted(panel["region"].unique())
    years = sorted(panel["year"].unique())
    pre = [y for y in years if y < 2022]
    rat = lc.region_age_time(lc.load_dhs_region_long(lc.SRC))

    bt_new, fc_new = [], []
    for ind, lckey in IND_MAP.items():
        act = actual_2022(panel, ind)

        # ---- Lee-Carter ----
        bt_rows = []
        for r in regions:
            yr, M = rat[r]
            mask = yr < 2022
            d = lc.fit_forecast(yr[mask], M[:, mask], [2022])[2022][lckey]
            bt_rows.append((d[0], act[r], d[1], d[2]))
        bt_new.append(metrics(bt_rows, ind, "LeeCarter"))
        for r in regions:
            yr, M = rat[r]
            f = lc.fit_forecast(yr, M, TARGETS)
            for ty in TARGETS:
                m, lo, hi = f[ty][lckey]
                fc_new.append({"region": r, "indicator": ind, "method": "LeeCarter",
                               "year": ty, "point": round(m, 3), "lo80": round(lo, 3), "hi80": round(hi, 3)})

        # ---- Bayesian hierarchical ----
        bbt = bayes_hier.fit(panel, ind, pre, [2022])
        bt_rows = [(bbt[r][2022][0], act[r], bbt[r][2022][1], bbt[r][2022][2]) for r in regions]
        bt_new.append(metrics(bt_rows, ind, "BayesHier"))
        bfc = bayes_hier.fit(panel, ind, years, TARGETS)
        for r in regions:
            for ty in TARGETS:
                m, lo, hi = bfc[r][ty]
                fc_new.append({"region": r, "indicator": ind, "method": "BayesHier",
                               "year": ty, "point": round(m, 3), "lo80": round(lo, 3), "hi80": round(hi, 3)})

    # merge with existing baseline+LSTM results
    bt_old = pd.read_csv("outputs/tables/forecast_backtest_metrics.csv")
    bt = pd.concat([bt_old, pd.DataFrame(bt_new)], ignore_index=True)
    bt = bt.sort_values(["indicator", "RMSE"]).reset_index(drop=True)
    bt.to_csv("outputs/tables/forecast_backtest_metrics.csv", index=False)

    fc_old = pd.read_csv("outputs/data/forecasts_2025_2030.csv")
    fc = pd.concat([fc_old, pd.DataFrame(fc_new)], ignore_index=True)
    fc[["point", "lo80", "hi80"]] = fc[["point", "lo80", "hi80"]].clip(lower=0).round(3)
    fc.to_csv("outputs/data/forecasts_2025_2030.csv", index=False)

    print("=== UNIFIED BACKTEST (hold out 2022; lower RMSE better; coverage80 target ~0.80) ===")
    print(bt.to_string(index=False))
    print("\n=== Best method by RMSE per indicator ===")
    for ind in IND_MAP:
        sub = bt[bt.indicator == ind].sort_values("RMSE")
        best = sub.iloc[0]
        print(f"  {ind}: {best['method']} (RMSE {best['RMSE']}, coverage {best['coverage80']})")


if __name__ == "__main__":
    main()
