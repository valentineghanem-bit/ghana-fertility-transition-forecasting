"""
district_ml.py — Project 15 PHASE 6b. Machine-learning of the structural determinants of subnational
fertility / adolescent fertility on the 261-district 2022 cross-section.

Gradient boosting (sklearn — xgboost/shap not installed) + permutation importance (model-agnostic
interpretability standing in for SHAP). Evaluated by LEAVE-ONE-REGION-OUT cross-validation (LOROCV):
because the fertility outcome is REGION-ASSIGNED (ecological — 16 distinct values mapped onto 261
districts), ordinary CV would leak region identity. LOROCV asks the honest question: can district-level
structural context predict a *held-out region's* fertility?

ECOLOGICAL CAVEAT (mandatory): the outcome varies only at region level; this quantifies the
association between district structural conditions and regional fertility, NOT a district-level causal
or individual relationship.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

SRC = "data/processed/district_cross_section_2022_261.csv"
FEATURES = ["poverty_rate", "poverty_intensity", "illiteracy_rate", "uninsured_rate",
            "employed_share", "under15_share", "women_15_64_share", "log_population", "urbanicity"]
TARGETS = {"asfr_15_19_region": "Adolescent ASFR 15-19", "tfr_15_49_region": "TFR 15-49"}
SEED = 42


def load():
    df = pd.read_csv(SRC)
    df["log_population"] = np.log(df["total_population"])
    df["urbanicity"] = df["class"].map({"District": 0, "Municipal": 1, "Metropolitan": 2})
    return df


def lorocv(df, target):
    """Leave-one-region-out CV; return region-level out-of-fold predictions vs actuals."""
    regions = sorted(df["region"].unique())
    rows = []
    for held in regions:
        tr = df[df["region"] != held]
        te = df[df["region"] == held]
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=SEED)
        m.fit(tr[FEATURES], tr[target])
        pred_region = m.predict(te[FEATURES]).mean()   # region-level prediction
        rows.append({"region": held, "actual": te[target].iloc[0], "pred": pred_region})
    return pd.DataFrame(rows)


def main():
    df = load()
    imp_rows, cv_summary = [], []
    for target, label in TARGETS.items():
        cv = lorocv(df, target)
        ss_res = ((cv["actual"] - cv["pred"]) ** 2).sum()
        ss_tot = ((cv["actual"] - cv["actual"].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt((ss_res / len(cv)))
        cv_summary.append({"target": label, "LOROCV_R2": round(float(r2), 3),
                           "LOROCV_RMSE": round(float(rmse), 3), "n_regions": len(cv)})
        # permutation importance on full-data model (interpretability)
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=SEED).fit(df[FEATURES], df[target])
        pi = permutation_importance(m, df[FEATURES], df[target], n_repeats=30, random_state=SEED)
        for f, mean, sd in sorted(zip(FEATURES, pi.importances_mean, pi.importances_std),
                                  key=lambda t: -t[1]):
            imp_rows.append({"target": label, "feature": f,
                             "perm_importance": round(float(mean), 4), "sd": round(float(sd), 4)})

    import os
    os.makedirs("outputs/tables", exist_ok=True)
    pd.DataFrame(cv_summary).to_csv("outputs/tables/district_ml_lorocv.csv", index=False)
    imp = pd.DataFrame(imp_rows)
    imp.to_csv("outputs/tables/district_ml_importance.csv", index=False)

    print("=== LOROCV (can district structural context predict a held-out region's fertility?) ===")
    print(pd.DataFrame(cv_summary).to_string(index=False))
    print("\n=== Permutation importance — top 5 determinants per outcome ===")
    for label in TARGETS.values():
        top = imp[imp.target == label].head(5)
        print(f"  {label}:")
        for _, r in top.iterrows():
            print(f"     {r['feature']:20s} {r['perm_importance']:.3f} (±{r['sd']:.3f})")


if __name__ == "__main__":
    main()
