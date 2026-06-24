"""
spatial_lisa.py — Project 15 PHASE 4 (spatial analysis), pure numpy/scipy (no pysal/esda available).
Global Moran's I + Local Moran's I (LISA) on the 261-district 2022 cross-section.

Spatial weights: k-nearest-neighbour (k=6) from district CENTROIDS (haversine), row-standardised.
Centroid-based KNN is used deliberately so the spatial frame is the FULL 261 districts — every
district (including the 3 with no own GeoJSON polygon: Guan, Sagnarigu, Awutu Senya West) has a
centroid, so none is dropped. (Ghana has 261 districts.)

Run on genuinely district-varying Census covariates. Region-assigned DHS outcomes (16 distinct
values) are NOT valid targets for district LISA (ecological) and are excluded here (Tenet 4/22).

Inference: conditional permutation (999, seed=42). Outputs:
  outputs/tables/spatial_global_moran_2022.csv
  outputs/tables/lisa_clusters_2022_261.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

SEED, PERM, K = 42, 999, 6
PERM_G = 9999  # more permutations for the GLOBAL test (cheap) -> lower pseudo-p floor
SRC = "data/processed/district_cross_section_2022_261.csv"
VARS = ["poverty_rate", "illiteracy_rate", "uninsured_rate", "under15_share", "women_15_64_share"]


def knn_neighbors(lat, lon, k=K):
    """k-nearest-neighbour indices via haversine on centroids (row-standardised equal weights)."""
    coords = np.radians(np.c_[lat, lon])
    nn = NearestNeighbors(n_neighbors=k + 1, metric="haversine").fit(coords)
    _, idx = nn.kneighbors(coords)
    return idx[:, 1:]  # drop self


def moran_global(x, nbr, rng):
    """Global Moran's I with permutation inference.

    Returns the discriminating significance measures (z_sim + analytical p) alongside the empirical
    pseudo-p. NOTE: the pseudo-p has a hard floor of 1/(PERM_G+1); when 0 permutations exceed the
    observed I (very strong autocorrelation), every variable saturates at that floor and the pseudo-p
    no longer discriminates — the z-score / analytical p must be used for relative strength.
    """
    n = len(x)
    z = x - x.mean()
    zz = (z * z).sum()
    lag = z[nbr].mean(axis=1)                       # row-standardised KNN spatial lag
    I = (z * lag).sum() / zz
    perm_I = np.empty(PERM_G)
    for p in range(PERM_G):
        zp = rng.permutation(z)
        perm_I[p] = (zp * zp[nbr].mean(axis=1)).sum() / zz
    n_exceed = int(np.sum(np.abs(perm_I) >= abs(I)))
    p_sim = (n_exceed + 1) / (PERM_G + 1)
    floor = 1.0 / (PERM_G + 1)
    z_sim = (I - perm_I.mean()) / perm_I.std()      # std Moran z from permutation null
    p_analytic = float(2 * norm.sf(abs(z_sim)))     # two-sided normal approx — discriminates
    return {
        "morans_I": round(I, 4), "expected_I": round(-1.0 / (n - 1), 4),
        "z_sim": round(z_sim, 2), "p_analytic": p_analytic,
        "n_exceed": n_exceed, "n_perm": PERM_G, "p_sim": p_sim, "p_sim_floored": n_exceed == 0,
        "p_sim_report": (f"<{floor:.1e}" if n_exceed == 0 else f"{p_sim:.2e}"),
        "autocorrelation": "positive" if I > -1.0 / (n - 1) else "negative",
        "significant_0.05": p_analytic < 0.05,
    }


def moran_local(x, nbr, rng):
    n = len(x)
    z = x - x.mean()
    m2 = (z * z).sum() / n
    lag = z[nbr].mean(axis=1)
    Ii = z / m2 * lag
    # conditional permutation: for each i, draw k neighbour values from the other n-1
    p = np.empty(n)
    for i in range(n):
        others = np.delete(z, i)
        draws = rng.random((PERM, others.size)).argsort(axis=1)[:, :K]   # k without replacement
        lag_p = others[draws].mean(axis=1)
        Ii_p = z[i] / m2 * lag_p
        p[i] = (np.sum(np.abs(Ii_p) >= abs(Ii[i])) + 1) / (PERM + 1)
    # cluster type at p<0.05 from sign of z_i and lag
    cl = np.full(n, "NS", dtype=object)
    sig = p < 0.05
    cl[sig & (z > 0) & (lag > 0)] = "HH"
    cl[sig & (z < 0) & (lag < 0)] = "LL"
    cl[sig & (z > 0) & (lag < 0)] = "HL"
    cl[sig & (z < 0) & (lag > 0)] = "LH"
    return Ii, lag, p, cl


def main():
    df = pd.read_csv(SRC)
    assert len(df) == 261, "spatial frame must be 261 districts"
    nbr = knn_neighbors(df["lat"].values, df["lon"].values)

    g_rows, lisa = [], df[["district", "region", "is_structural_gap"]].copy()
    for v in VARS:
        x = df[v].values.astype(float)
        rng = np.random.default_rng(SEED)
        g = moran_global(x, nbr, rng)
        g_rows.append({"variable": v, **g})
        rng = np.random.default_rng(SEED)
        Ii, lag, lp, cl = moran_local(x, nbr, rng)
        lisa[f"{v}__LISA"] = cl
        lisa[f"{v}__p"] = lp.round(4)

    gdf = pd.DataFrame(g_rows)
    import os
    os.makedirs("outputs/tables", exist_ok=True)
    gdf.to_csv("outputs/tables/spatial_global_moran_2022.csv", index=False)
    lisa.to_csv("outputs/tables/lisa_clusters_2022_261.csv", index=False)

    print(f"=== Global Moran's I (k=6 KNN, 261 districts, {PERM_G} perms, seed=42) ===")
    print(gdf[["variable", "morans_I", "expected_I", "z_sim", "p_analytic",
               "n_exceed", "p_sim_report", "significant_0.05"]].to_string(index=False))
    if (gdf["p_sim_floored"]).all():
        print(f"\nNOTE: 0/{PERM_G} permutations exceeded the observed I for every variable -> pseudo-p is"
              f" pinned at its floor ({1/(PERM_G+1):.1e}); use z_sim / p_analytic for relative strength.")
    print("\n=== LISA cluster counts per variable ===")
    for v in VARS:
        vc = lisa[f"{v}__LISA"].value_counts().reindex(["HH", "LL", "HL", "LH", "NS"]).fillna(0).astype(int)
        print(f"  {v:20s} HH={vc.HH} LL={vc.LL} HL={vc.HL} LH={vc.LH} NS={vc.NS}")
    print("\nstructural-gap districts present in LISA output:",
          int(lisa["is_structural_gap"].sum()), "(must be 3 -> frame is 261)")


if __name__ == "__main__":
    main()
