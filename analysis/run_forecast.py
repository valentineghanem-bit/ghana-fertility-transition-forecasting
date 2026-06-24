"""
run_forecast.py — Project 15 PHASE 5. Forecast 16-region TFR (15-49) and adolescent ASFR (15-19)
to 2025 and 2030, benchmarking a from-scratch time-aware LSTM (lstm_numpy.py) against year-aware
statistical baselines. Honest backtest (hold out 2022) reports RMSE/MAE + 80% prediction-interval
coverage per method (Tenet 6 — uncertainty before interpretation).

Outputs:
  outputs/tables/forecast_backtest_metrics.csv   (per indicator x method: RMSE, MAE, coverage80)
  outputs/data/forecasts_2025_2030.csv           (region x indicator x method x year: point, lo80, hi80)
  docs/Phase5_Forecasting.md is written by hand from these results.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lstm_numpy import TimeAwareLSTM, gradient_check  # noqa: E402

PANEL = "outputs/data/master_panel_wide.csv"
INDICATORS = ["tfr_15_49", "asfr_15_19"]
TARGET_YEARS = [2025, 2030]
Z80 = 1.2816  # 80% PI half-width in SDs
M_ENS, EPOCHS, HIDDEN = 12, 500, 8


# ---------- year-aware statistical baselines: return (point, sd) at a target year ----------
def rwd(yr, y, t):
    mu = (y[-1] - y[0]) / (yr[-1] - yr[0])
    gaps = np.diff(yr)
    innov = (np.diff(y) - mu * gaps) / np.sqrt(gaps)
    s2 = np.mean(innov ** 2)
    h = t - yr[-1]
    return y[-1] + mu * h, np.sqrt(max(s2, 1e-9) * h)


def _ols(xv, yv, x0):
    n = len(xv)
    xm, ym = xv.mean(), yv.mean()
    b = ((xv - xm) * (yv - ym)).sum() / ((xv - xm) ** 2).sum()
    a = ym - b * xm
    resid = yv - (a + b * xv)
    s = np.sqrt((resid ** 2).sum() / max(n - 2, 1))
    se = s * np.sqrt(1 + 1 / n + (x0 - xm) ** 2 / ((xv - xm) ** 2).sum())
    return a + b * x0, se


def linear(yr, y, t):
    return _ols(yr.astype(float), y, float(t))


def loglinear(yr, y, t):
    p, se = _ols(yr.astype(float), np.log(y), float(t))
    point = np.exp(p)
    return point, point * se  # delta-method sd on original scale


BASELINES = {"RWD": rwd, "Linear": linear, "LogLinear": loglinear}


# ---------- LSTM ensemble (standardised, time-aware) ----------
def _standardise(panel, ind):
    vals = panel[ind].values.astype(float)
    return vals.mean(), vals.std()


def _region_series(panel, region, ind):
    s = panel[panel["region"] == region].sort_values("year")
    return s["year"].values.astype(float), s[ind].values.astype(float)


def lstm_forecast(panel, regions, ind, fit_years, target_years, vmu, vsd, dmu, dsd, seeds):
    """Train an ensemble on fit_years; return {region: {ty: (mean, lo, hi)}} for target_years."""
    seqs = []
    for r in regions:
        yr, y = _region_series(panel, r, ind)
        mask = np.isin(yr, fit_years)
        yr, y = yr[mask], y[mask]
        ys = (y - vmu) / vsd
        gaps = np.diff(yr)
        X = np.column_stack([ys[:-1], (gaps - dmu) / dsd])
        seqs.append((r, yr, y, ys, X, (yr[-1])))
    train = [(X, ys[1:]) for (_, _, _, ys, X, _) in seqs]

    preds = {r: {ty: [] for ty in target_years} for r in regions}
    for sd in seeds:
        net = TimeAwareLSTM(hidden=HIDDEN, seed=sd)
        net.fit(train, epochs=EPOCHS, lr=0.05)
        for (r, yr, y, ys, X, last_yr) in seqs:
            fdts = []
            prev = last_yr
            for ty in target_years:
                fdts.append((ty - prev - dmu) / dsd)
                prev = ty
            out_std = net.forecast(X, ys[-1], fdts)
            for ty, v in zip(target_years, out_std):
                preds[r][ty].append(v * vsd + vmu)
    res = {}
    for r in regions:
        res[r] = {ty: (float(np.mean(preds[r][ty])),
                       float(np.percentile(preds[r][ty], 10)),
                       float(np.percentile(preds[r][ty], 90))) for ty in target_years}
    return res


def main():
    gc = gradient_check()
    assert gc < 1e-4, f"LSTM gradient check failed ({gc:.1e})"
    panel = pd.read_csv(PANEL)
    regions = sorted(panel["region"].unique())
    all_years = sorted(panel["year"].unique())
    fit_years_bt = [y for y in all_years if y < 2022]          # backtest: hold out 2022
    bt_rows, fc_rows = [], []

    for ind in INDICATORS:
        vmu, vsd = _standardise(panel, ind)
        gaps_all = np.diff(np.array(all_years, float))
        dmu, dsd = gaps_all.mean(), gaps_all.std()

        # ---- BACKTEST: fit on <2022, predict 2022 ----
        actual22 = {r: _region_series(panel, r, ind)[1][-1] for r in regions}
        for name, fn in BASELINES.items():
            errs, cov = [], []
            for r in regions:
                yr, y = _region_series(panel, r, ind)
                yrb, yb = yr[:-1], y[:-1]
                pt, sd = fn(yrb, yb, 2022)
                a = actual22[r]
                errs.append(pt - a)
                cov.append(abs(a - pt) <= Z80 * sd)
            errs = np.array(errs)
            bt_rows.append({"indicator": ind, "method": name,
                            "RMSE": round(float(np.sqrt((errs ** 2).mean())), 3),
                            "MAE": round(float(np.abs(errs).mean()), 3),
                            "coverage80": round(float(np.mean(cov)), 3)})
        # LSTM backtest — standardise on PRE-2022 ONLY (no train/test leak; the final forecast below
        # legitimately uses full-data vmu/vsd)
        pre_vals = pd.to_numeric(panel[panel["year"].isin(fit_years_bt)][ind], errors="coerce")
        vmu_bt, vsd_bt = pre_vals.mean(), pre_vals.std()
        gaps_pre = np.diff(np.array(fit_years_bt, float))
        dmu_bt, dsd_bt = gaps_pre.mean(), gaps_pre.std()
        lr = lstm_forecast(panel, regions, ind, fit_years_bt, [2022], vmu_bt, vsd_bt, dmu_bt, dsd_bt,
                           seeds=range(M_ENS))
        errs, cov = [], []
        for r in regions:
            mean, lo, hi = lr[r][2022]
            a = actual22[r]
            errs.append(mean - a)
            cov.append(lo <= a <= hi)
        errs = np.array(errs)
        bt_rows.append({"indicator": ind, "method": "LSTM",
                        "RMSE": round(float(np.sqrt((errs ** 2).mean())), 3),
                        "MAE": round(float(np.abs(errs).mean()), 3),
                        "coverage80": round(float(np.mean(cov)), 3)})

        # ---- FORECAST 2025 & 2030 on FULL data ----
        for name, fn in BASELINES.items():
            for r in regions:
                yr, y = _region_series(panel, r, ind)
                for ty in TARGET_YEARS:
                    pt, sd = fn(yr, y, ty)
                    fc_rows.append({"region": r, "indicator": ind, "method": name, "year": ty,
                                    "point": round(float(pt), 3),
                                    "lo80": round(float(pt - Z80 * sd), 3),
                                    "hi80": round(float(pt + Z80 * sd), 3)})
        lf = lstm_forecast(panel, regions, ind, all_years, TARGET_YEARS, vmu, vsd, dmu, dsd,
                           seeds=range(M_ENS))
        for r in regions:
            for ty in TARGET_YEARS:
                mean, lo, hi = lf[r][ty]
                fc_rows.append({"region": r, "indicator": ind, "method": "LSTM", "year": ty,
                                "point": round(mean, 3), "lo80": round(lo, 3), "hi80": round(hi, 3)})

    os.makedirs("outputs/tables", exist_ok=True)
    bt = pd.DataFrame(bt_rows)
    bt.to_csv("outputs/tables/forecast_backtest_metrics.csv", index=False)
    fc = pd.DataFrame(fc_rows)
    # TFR and ASFR are non-negative — floor point + interval at 0 (Linear PIs can dip below 0 at low rates)
    fc[["point", "lo80", "hi80"]] = fc[["point", "lo80", "hi80"]].clip(lower=0).round(3)
    fc.to_csv("outputs/data/forecasts_2025_2030.csv", index=False)

    print("=== BACKTEST (hold out 2022; lower RMSE/MAE better; coverage80 target ~0.80) ===")
    print(bt.to_string(index=False))
    print("\n=== National (pop-unweighted mean) 2030 forecast by method ===")
    fc = pd.DataFrame(fc_rows)
    for ind in INDICATORS:
        sub = fc[(fc.indicator == ind) & (fc.year == 2030)]
        print(f"  {ind}: " + " | ".join(f"{m}={sub[sub.method==m]['point'].mean():.2f}"
                                        for m in ["RWD", "Linear", "LogLinear", "LSTM"]))


if __name__ == "__main__":
    main()
