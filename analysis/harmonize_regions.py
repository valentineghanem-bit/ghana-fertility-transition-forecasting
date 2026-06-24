"""
harmonize_regions.py — Project 15 (Ghana Fertility Transition / LSTM)
Phase 0/1 VERIFIED region-harmonisation module.

Purpose: collapse the DHS StatCompiler subnational region hierarchy (which mixes
legacy, intermediate, and 2022 administrative units across waves) into a single
16-region (2022 administrative frame) x 9-wave panel, with NO fabricated values.

This SUPERSEDES the original Phase-0 filter rule
    `CharacteristicCategory=='Region' AND LevelRank==1.0`
which was shown to be WRONG: every region row in the source carries LevelRank==1.0,
so that filter removes nothing and leaves the redundant combined aggregate
("Northern, Upper West, Upper East") co-occurring with its children.

Correct rule (verified 2026-06-22 against fertility-rates_subnational_gha.csv):
  1. Drop the HXL hashtag row (SurveyYear == '#date+year').
  2. Strip the DHS indent prefixes ('..', '....') from CharacteristicLabel.
  3. For each of the 16 target regions, take the value from the FINEST available
     reporting unit each wave, falling back along a documented ancestry chain.
  4. The combined "Northern, Upper West, Upper East" label is used ONLY as the
     1988 fallback for the five northern-zone descendants; it is never emitted
     as its own panel row.

Observed reporting frame (verified):
  1988          : 8 units, northern zone fully combined
  1993-2019     : 10 real regions (combined-North label is a redundant duplicate)
  2022          : full 16 regions (+5 legacy parents present but unused)
"""
import re
import pandas as pd

WAVES = [1988, 1993, 1998, 2003, 2008, 2014, 2016, 2019, 2022]
COMBINED_NORTH = "Northern, Upper West, Upper East"

# Target region -> ancestry chain (finest available first, coarsest fallback last).
# Sibling regions created by the 2018 splits inherit their pre-2018 parent before
# their own 2022 value exists -> identical trajectories pre-2022 by construction.
ANCESTRY = {
    "Ashanti":        ["Ashanti"],
    "Central":        ["Central"],
    "Greater Accra":  ["Greater Accra"],
    "Eastern":        ["Eastern"],
    "Upper East":     ["Upper East", COMBINED_NORTH],      # own value from 1993; combined only 1988
    "Upper West":     ["Upper West", COMBINED_NORTH],
    "Western":        ["Western (post 2022)", "Western (pre 2022)"],
    "Western North":  ["Western North", "Western (pre 2022)"],
    "Volta":          ["Volta (post 2022)", "Volta (pre 2022)"],
    "Oti":            ["Oti", "Volta (pre 2022)"],
    "Ahafo":          ["Ahafo", "Brong-Ahafo"],
    "Bono":           ["Bono", "Brong-Ahafo"],
    "Bono East":      ["Bono East", "Brong-Ahafo"],
    "Northern":       ["Northern(post 2022)", "Northern (pre 2022)", COMBINED_NORTH],
    "Savannah":       ["Savannah", "Northern (pre 2022)", COMBINED_NORTH],
    "North East":     ["Northeast", "Northern (pre 2022)", COMBINED_NORTH],
}
assert len(ANCESTRY) == 16


def _strip_prefix(label: str) -> str:
    """Remove DHS '..'/'....' indent prefixes that encode hierarchy depth."""
    return re.sub(r"^\.+", "", str(label)).strip()


def load_dhs_region_long(path: str) -> pd.DataFrame:
    """Load a DHS StatCompiler subnational CSV, drop HXL row, clean region labels."""
    df = pd.read_csv(path, low_memory=False)
    df = df[df["SurveyYear"].astype(str) != "#date+year"].copy()
    df["SurveyYear"] = df["SurveyYear"].astype(int)
    df["region_raw"] = df["CharacteristicLabel"].apply(_strip_prefix)
    return df


def harmonize_indicator(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """Return a long 16-region x 9-wave panel for one indicator.

    Columns: region, year, value, source_label, inherited (bool).
    `inherited=True` flags a value carried from a coarser ancestor (a structural,
    documented limitation -- NOT imputation/fabrication).
    """
    sub = df[df["Indicator"] == indicator]
    lookup = {(r.SurveyYear, r.region_raw): r.Value for r in sub.itertuples()}
    out = []
    for region, chain in ANCESTRY.items():
        for year in WAVES:
            value, source = None, None
            for cand in chain:
                if (year, cand) in lookup:
                    value, source = lookup[(year, cand)], cand
                    break
            out.append({
                "region": region, "year": year, "value": value,
                "source_label": source,
                "inherited": source is not None and source != region,
            })
    return pd.DataFrame(out)


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/fertility-rates_subnational_gha.csv"
    ind = sys.argv[2] if len(sys.argv) > 2 else "Total fertility rate 15-49"
    long = load_dhs_region_long(src)
    panel = harmonize_indicator(long, ind)
    n_missing = int(panel["value"].isna().sum())
    print(f"indicator={ind!r}  cells={len(panel)} (expect 144)  missing={n_missing}")
    print(panel.pivot(index="region", columns="year", values="value").round(2).to_string())
