"""
lee_carter.py — Project 15 PHASE 5 (reframed). Lee-Carter forecasting on the FULL age-specific
fertility schedule (ASFR 15-19 ... 45-49) per region. The demography-native method: it forecasts the
whole age schedule coherently, yielding TFR (sum of ASFR) AND adolescent ASFR (15-19) together.

Model:  log m_{x,t} = a_x + b_x * k_t,  with sum_x b_x = 1, sum_t k_t = 0  (SVD rank-1).
Forecast k_t by year-aware random walk with drift; reconstruct rates; simulate for prediction
intervals. Year-aware drift handles the irregular DHS spacing (no interpolation).

Pure numpy. Uses harmonize_regions for the 16-region frame (verified ancestry rule).
"""
import numpy as np
import pandas as pd

from harmonize_regions import load_dhs_region_long, harmonize_indicator, ANCESTRY

AGES = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
AGE_INDS = [f"Age specific fertility rate: {a}" for a in AGES]
SRC = "data/raw/fertility-rates_subnational_gha.csv"


def region_age_time(long_df):
    """Return {region: (years[T], M[A,T] ASFR per 1000)} using the harmonised 16-region frame."""
    panels = {ind: harmonize_indicator(long_df, ind) for ind in AGE_INDS}
    out = {}
    for r in ANCESTRY:
        years = sorted(panels[AGE_INDS[0]].query("region==@r")["year"].unique())
        M = np.zeros((len(AGES), len(years)))
        for ai, ind in enumerate(AGE_INDS):
            p = pd.to_numeric(panels[ind].query("region==@r").set_index("year")["value"], errors="coerce")
            M[ai] = [p.get(y, np.nan) for y in years]
        out[r] = (np.array(years, float), M)
    return out


def fit_forecast(years, M, target_years, n_sim=1000, seed=42):
    """Lee-Carter fit + simulated forecast. Returns dict target_year -> {tfr:(m,lo,hi), asfr15:(m,lo,hi)}.
    Missing age cells (mostly the oldest ages) are filled by EM-SVD rank-1 imputation so all waves are
    retained (dropping incomplete waves would bias the trend by discarding recent surveys)."""
    years = np.asarray(years, float)
    if M.shape[1] < 4 or np.isnan(M).all(axis=1).any():
        raise ValueError("Lee-Carter not estimable (need >=4 waves and no fully-missing age)")
    logM = np.log(np.clip(M, 1e-3, None))     # NaN preserved
    mask = np.isnan(logM)
    filled = np.where(mask, np.nanmean(logM, axis=1)[:, None], logM)
    for _ in range(200):                       # EM-SVD imputation
        a = filled.mean(axis=1)
        U, S, Vt = np.linalg.svd(filled - a[:, None], full_matrices=False)
        recon = a[:, None] + S[0] * np.outer(U[:, 0], Vt[0])
        new = np.where(mask, recon, logM)
        if np.max(np.abs(new - filled)) < 1e-7:
            filled = new
            break
        filled = new
    a = filled.mean(axis=1)
    U, S, Vt = np.linalg.svd(filled - a[:, None], full_matrices=False)
    b = U[:, 0]
    k = S[0] * Vt[0]
    if b.sum() < 0:          # sign convention: sum_x b_x = 1
        b, k = -b, -k
    k = k * b.sum()
    b = b / b.sum()
    # year-aware RWD on k_t
    drift = (k[-1] - k[0]) / (years[-1] - years[0])
    gaps = np.diff(years)
    innov = (np.diff(k) - drift * gaps) / np.sqrt(gaps)
    s_k = np.sqrt(max(np.mean(innov ** 2), 1e-9))
    # Lee-Miller jump-off correction: anchor to the LAST OBSERVED (imputed) schedule and apply the
    # LC-estimated rate of change, so forecasts start from 2022 rather than the rank-1 fitted level
    # (removes the reversion that otherwise makes recently-declining regions forecast upward).
    last_log = filled[:, -1]
    rng = np.random.default_rng(seed)
    res = {}
    for ty in target_years:
        h = ty - years[-1]
        sims_tfr, sims_a15 = [], []
        for _ in range(n_sim):
            dk = drift * h + s_k * np.sqrt(h) * rng.standard_normal()
            asfr = np.exp(last_log + b * dk)
            sims_tfr.append(5.0 * asfr.sum() / 1000.0)   # 5-yr age groups
            sims_a15.append(asfr[0])
        # point = MEDIAN of simulations (robust to lognormal skew; the mean inflates with horizon
        # via Jensen's inequality, which spuriously makes longer-horizon point forecasts rise)
        res[ty] = {
            "tfr": (float(np.median(sims_tfr)), float(np.percentile(sims_tfr, 10)), float(np.percentile(sims_tfr, 90))),
            "asfr15": (float(np.median(sims_a15)), float(np.percentile(sims_a15, 10)), float(np.percentile(sims_a15, 90))),
        }
    return res


def point_forecast(years, M, t):
    """Deterministic point forecast (mean) of TFR & ASFR15 at year t — for backtesting."""
    r = fit_forecast(years, M, [t], n_sim=200)
    return r[t]["tfr"][0], r[t]["asfr15"][0]


if __name__ == "__main__":
    long = load_dhs_region_long(SRC)
    rat = region_age_time(long)
    # validate: reconstructed TFR from observed ASFR matches reported TFR
    tfr_rep = harmonize_indicator(long, "Total fertility rate 15-49")
    errs = []
    for r, (yr, M) in rat.items():
        recon = 5 * M.sum(axis=0) / 1000
        rep = pd.to_numeric(tfr_rep.query("region==@r").set_index("year")["value"], errors="coerce").reindex(yr).values
        errs.append(np.nanmean(np.abs(recon - rep)))
    print(f"ASFR->TFR reconstruction mean abs error vs reported TFR: {np.nanmean(errs):.3f} (small = schedule consistent)")
    demo = fit_forecast(*rat["Northern"], [2025, 2030])
    print("Northern Lee-Carter:", {ty: {k: tuple(round(x, 2) for x in v) for k, v in d.items()} for ty, d in demo.items()})
