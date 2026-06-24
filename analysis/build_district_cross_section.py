"""
build_district_cross_section.py — Project 15 PHASE 3b (data wrangling, district stream).
Assemble the 2022 cross-sectional dataset on the FULL 261-district frame (Ghana has 261 districts —
the 260-polygon GeoJSON is a rendering artefact, NOT the analytical frame).

E10 resolved: all 261 Master-Sheet districts are retained as rows; the 3 structural-gap districts
(Guan, Sagnarigu Municipal, Awutu Senya West — created after the GeoJSON vintage) keep full data and
are flagged `is_structural_gap=True`, mapped to their parent polygon for rendering only.

Outputs:
  data/processed/district_cross_section_2022_261.csv
  outputs/data/district_cross_section_2022_261.csv

NOTE (ecological flag): district-level fertility/contraception columns are REGION-assigned 2022 DHS
values (suffix `_region`), not independently measured at district level. They carry only 16 distinct
values. District-level spatial inference must use the genuinely district-varying Census covariates;
the region-assigned outcomes are for context/choropleth only (Tenet 4/22).
"""
import pandas as pd

MS = "data/raw/Master Sheet.xlsx"
CW = "docs/district_crosswalk_261_to_260.csv"
PANEL = "outputs/data/master_panel_wide.csv"
DCOL = "Metropolitan, Municipal, and District Assemblies (MMDA's)"


def build():
    ms = pd.read_excel(MS)
    assert len(ms) == 261, f"Master Sheet must have 261 districts, got {len(ms)}"
    ms = ms.rename(columns={DCOL: "district", "Region": "region"})

    tot = ms["Total Population"]
    out = pd.DataFrame({
        "district": ms["district"].str.strip(),
        "region": ms["region"].str.strip(),
        "class": ms["Class"],
        "lat": ms["Latitude"], "lon": ms["Longitude"],
        "poverty_rate": ms["Incidence of Poverty"],          # already %
        "poverty_intensity": ms["Intensity of Poverty"],     # already %
        "illiteracy_rate": (ms["Illiterate Population"] / tot * 100).round(2),
        "uninsured_rate": (ms["Uninsured Population"] / tot * 100).round(2),
        "employed_share": (ms["Employed Population"] / tot * 100).round(2),
        "under15_share": ((ms["Male Population 0-14"] + ms["Female Population 0-14"]) / tot * 100).round(2),
        "women_15_64_share": (ms["Female Population 15-64"] / tot * 100).round(2),
        "total_population": tot.astype(int),
    })

    # crosswalk -> polygon mapping + structural-gap flag (all 261 retained)
    cw = pd.read_csv(CW)
    cw["district"] = cw["master_sheet_district"].str.strip()
    out = out.merge(cw[["district", "geojson_district", "match_method"]], on="district", how="left")
    out["is_structural_gap"] = out["match_method"].eq("structural_gap")

    # region-assigned 2022 DHS outcomes (ECOLOGICAL — region-level, suffixed _region)
    panel = pd.read_csv(PANEL)
    p22 = panel[panel["year"] == 2022].copy()
    keep = ["tfr_15_49", "asfr_15_19", "mcpr_married", "unmet_need", "edu_secondary_women"]
    p22 = p22[["region"] + keep].rename(columns={k: f"{k}_region" for k in keep})
    out = out.merge(p22, on="region", how="left")

    assert len(out) == 261 and out["district"].nunique() == 261, "frame must be 261 districts"
    out.to_csv("data/processed/district_cross_section_2022_261.csv", index=False)
    return out


if __name__ == "__main__":
    df = build()
    print("district cross-section rows:", len(df), "| districts:", df["district"].nunique(),
          "| regions:", df["region"].nunique())
    print("structural-gap districts (retained):",
          df.loc[df["is_structural_gap"], "district"].tolist())
    print("\ndistrict-varying covariate ranges:")
    print(df[["poverty_rate", "illiteracy_rate", "uninsured_rate", "under15_share",
              "women_15_64_share"]].describe().T[["min", "max", "mean"]].round(2).to_string())
    print("\nregion-assigned outcome distinct values (expect <=16):",
          {c: df[c].nunique() for c in df.columns if c.endswith("_region")})
