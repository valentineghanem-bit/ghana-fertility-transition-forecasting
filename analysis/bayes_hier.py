"""
bayes_hier.py — Project 15 PHASE 5 (reframed headline method). Bayesian hierarchical log-linear trend
forecaster with PARTIAL POOLING across Ghana's 16 regions, fit by a Gibbs sampler (pure NumPy — no
PyMC/Stan installed). This is the demographically-appropriate small-sample approach (cf. UN/bayesTFR,
Raftery 2012): each region's intercept and slope are shrunk toward the national mean, so noisy /
short regional series borrow strength, and forecasts carry calibrated credible intervals.

Model (per indicator, on log-rate; x = year - mean(year)):
  y_{r,i} = alpha_r + beta_r * x_i + eps,  eps ~ N(0, sigma2)
  alpha_r ~ N(mu_a, ta2),  beta_r ~ N(mu_b, tb2)            # hierarchy => partial pooling
  mu_a, mu_b ~ N(0, 1e4);  1/sigma2, 1/ta2, 1/tb2 ~ Gamma(0.01, 0.01)
Forecast = posterior predictive median + 10/90 credible interval at the target year.
"""
import numpy as np
import pandas as pd

PANEL = "outputs/data/master_panel_wide.csv"


def _series(panel, region, ind):
    s = panel[panel["region"] == region].sort_values("year")
    y = pd.to_numeric(s[ind], errors="coerce").values
    yr = s["year"].values.astype(float)
    ok = ~np.isnan(y)
    return yr[ok], y[ok]


def fit(panel, ind, fit_years, target_years, iters=4000, burn=1500, seed=42):
    """Gibbs-fit the hierarchy on fit_years; return {region: {ty: (median, lo10, hi90)}}."""
    rng = np.random.default_rng(seed)
    regions = sorted(panel["region"].unique())
    xbar = np.mean([y for y in fit_years])
    data = {}
    for r in regions:
        yr, y = _series(panel, r, ind)
        m = np.isin(yr, fit_years)
        yr, y = yr[m], np.log(np.clip(y[m], 1e-6, None))
        X = np.column_stack([np.ones_like(yr), yr - xbar])
        data[r] = (X, y)
    R = len(regions)
    N = sum(len(y) for _, y in data.values())

    # init
    alpha = {r: data[r][1].mean() for r in regions}
    beta = {r: 0.0 for r in regions}
    mu_a, mu_b = np.mean(list(alpha.values())), 0.0
    sigma2, ta2, tb2 = 0.1, 1.0, 0.01
    a0 = b0 = 0.01
    draws = {r: {ty: [] for ty in target_years} for r in regions}

    for it in range(iters):
        # 1. region-level (alpha_r, beta_r) | rest  (conjugate Normal)
        Lam0 = np.diag([1.0 / ta2, 1.0 / tb2])
        mu0 = np.array([mu_a, mu_b])
        for r in regions:
            X, y = data[r]
            prec = Lam0 + (X.T @ X) / sigma2
            cov = np.linalg.inv(prec)
            mean = cov @ (Lam0 @ mu0 + (X.T @ y) / sigma2)
            th = rng.multivariate_normal(mean, cov)
            alpha[r], beta[r] = th[0], th[1]
        # 2. hyper-means mu_a, mu_b | region effects
        av = np.array([alpha[r] for r in regions]); bv = np.array([beta[r] for r in regions])
        pa = R / ta2 + 1e-4; mu_a = rng.normal((av.sum() / ta2) / pa, np.sqrt(1 / pa))
        pb = R / tb2 + 1e-4; mu_b = rng.normal((bv.sum() / tb2) / pb, np.sqrt(1 / pb))
        # 3. hyper-variances ta2, tb2 | region effects  (InvGamma)
        ta2 = 1.0 / rng.gamma(a0 + R / 2, 1.0 / (b0 + 0.5 * np.sum((av - mu_a) ** 2)))
        tb2 = 1.0 / rng.gamma(a0 + R / 2, 1.0 / (b0 + 0.5 * np.sum((bv - mu_b) ** 2)))
        # 4. residual variance sigma2 | rest
        sse = sum(np.sum((y - (alpha[r] + beta[r] * X[:, 1])) ** 2) for r, (X, y) in data.items())
        sigma2 = 1.0 / rng.gamma(a0 + N / 2, 1.0 / (b0 + 0.5 * sse))
        # collect posterior-predictive draws
        if it >= burn:
            sd = np.sqrt(sigma2)
            for r in regions:
                for ty in target_years:
                    mu = alpha[r] + beta[r] * (ty - xbar)
                    draws[r][ty].append(np.exp(mu + sd * rng.standard_normal()))

    res = {}
    for r in regions:
        res[r] = {ty: (float(np.median(draws[r][ty])),
                       float(np.percentile(draws[r][ty], 10)),
                       float(np.percentile(draws[r][ty], 90))) for ty in target_years}
    return res


if __name__ == "__main__":
    panel = pd.read_csv(PANEL)
    yrs = sorted(panel["year"].unique())
    r = fit(panel, "tfr_15_49", yrs, [2025, 2030], iters=2500, burn=1000)
    print("Bayesian hierarchical TFR (median [80% CrI]) — sample regions:")
    for reg in ["Greater Accra", "Ashanti", "Northern", "North East", "Savannah", "Volta"]:
        f25, f30 = r[reg][2025], r[reg][2030]
        print(f"  {reg:14s} 2025={f25[0]:.2f} [{f25[1]:.2f},{f25[2]:.2f}]  2030={f30[0]:.2f} [{f30[1]:.2f},{f30[2]:.2f}]")
