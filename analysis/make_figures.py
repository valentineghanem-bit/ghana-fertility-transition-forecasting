"""
make_figures.py — Project 15 PHASE 6. Q1 submission-grade figures.
Each figure has a DISTINCT title and a legend placed OUTSIDE the plot area (never overlapping data).
Colourblind-safe Okabe-Ito; units on axes; bold panel labels A/B/… on multi-panel figures; minimal
chartjunk; journal column widths. Saved as BOTH a vector PDF (submission) and a 600-DPI PNG (.docx),
with embedded TrueType fonts (pdf.fonttype 42).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.grid": False, "figure.titlesize": 11.5, "figure.titleweight": "bold",
})

OUT = "outputs/figures"
PANEL = "outputs/data/master_panel_wide.csv"
FC = "outputs/data/forecasts_2025_2030.csv"
BT = "outputs/tables/forecast_backtest_metrics.csv"
LISA = "outputs/tables/lisa_clusters_2022_261.csv"

OK = {"north": "#D55E00", "middle": "#E69F00", "south": "#0072B2"}
ZONE = {**{r: "north" for r in ["Northern", "Savannah", "North East", "Upper East", "Upper West"]},
        **{r: "middle" for r in ["Bono", "Bono East", "Ahafo", "Oti", "Volta"]},
        **{r: "south" for r in ["Greater Accra", "Central", "Eastern", "Western", "Western North", "Ashanti"]}}
LISA_COL = {"HH": "#c0392b", "LL": "#2980b9", "HL": "#e67e22", "LH": "#82c0e8", "NS": "#d9d9d9"}
CONTRAST = ["Greater Accra", "Ashanti", "Northern", "North East"]


def _save(fig, stem):
    fig.savefig(f"{OUT}/{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT}/{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _panel(ax, lab, dx=-0.16):
    ax.text(dx, 1.04, lab, transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom")


def _series(panel, region, ind):
    s = panel[panel["region"] == region].sort_values("year")
    return s["year"].values, pd.to_numeric(s[ind], errors="coerce").values


def fig_trajectories(panel, ind, ylab, title, stem):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for r in sorted(panel["region"].unique()):
        yr, y = _series(panel, r, ind)
        ax.plot(yr, y, color=OK[ZONE[r]], alpha=0.55, lw=1.1)
    nat = panel.groupby("year")[ind].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    ax.plot(nat.index, nat.values, "k--", lw=2.0, label="National (unweighted) mean")
    for z, c in OK.items():
        ax.plot([], [], color=c, lw=2, label=z.capitalize() + " regions")
    ax.set_xlabel("Survey year"); ax.set_ylabel(ylab)
    ax.set_title(title, fontweight="bold", fontsize=11, pad=8)
    ax.spines[["top", "right"]].set_visible(False); ax.margins(x=0.01)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    _save(fig, stem)


def fig_forecast_fans(panel, fc, ind, ylab, title, stem):
    bh = fc[(fc.method == "BayesHier") & (fc.indicator == ind)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True)
    for pl, (ax, r) in zip("ABCD", zip(axes.ravel(), CONTRAST)):
        _panel(ax, pl, dx=-0.18)
        yr, y = _series(panel, r, ind)
        ax.plot(yr, y, "-o", color="#222222", ms=3, lw=1.3)
        sub = bh[bh.region == r].sort_values("year")
        fx = [2022] + list(sub.year); last = y[-1]
        ax.plot(fx, [last] + list(sub.point), "--", color=OK["south"], lw=1.6)
        ax.fill_between(fx, [last] + list(sub.lo80), [last] + list(sub.hi80), color=OK["south"], alpha=0.2)
        ax.set_title(r, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylab)
    for ax in axes[1, :]:
        ax.set_xlabel("Year")
    handles = [mlines.Line2D([], [], color="#222222", marker="o", ms=3, lw=1.3, label="Observed"),
               mlines.Line2D([], [], color=OK["south"], ls="--", lw=1.6, label="Forecast median"),
               mpatches.Patch(color=OK["south"], alpha=0.2, label="80% credible interval")]
    fig.suptitle(title, y=0.995)
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=3)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    _save(fig, stem)


def fig_backtest(bt, title, stem):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for pl, (ax, ind, lab) in zip("AB", zip(axes, ["tfr_15_49", "asfr_15_19"],
                                            ["TFR 15-49 (RMSE)", "Adolescent ASFR 15-19 (RMSE)"])):
        _panel(ax, pl, dx=-0.22)
        s = bt[bt.indicator == ind].sort_values("RMSE")
        colors = ["#009E73" if m in ("BayesHier", "LeeCarter") else ("#D55E00" if m == "LSTM" else "#9e9e9e")
                  for m in s.method]
        ax.barh(s.method, s.RMSE, color=colors); ax.invert_yaxis(); ax.set_xlabel(lab)
        for i, v in enumerate(s.RMSE):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [mpatches.Patch(color="#009E73", label="Headline (Bayes / Lee-Carter)"),
               mpatches.Patch(color="#9e9e9e", label="Statistical baseline"),
               mpatches.Patch(color="#D55E00", label="LSTM (benchmark)")]
    fig.suptitle(title, y=0.995)
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=3)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    _save(fig, stem)


def _norm(s):
    import re
    return re.sub(r"[^A-Z0-9 ]", "", str(s).upper()).strip()


def fig_lisa_map(title, stem, var="illiteracy_rate"):
    import json
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    li = pd.read_csv(LISA); cw = pd.read_csv("docs/district_crosswalk_261_to_260.csv")
    cls = dict(zip(li["district"], li[f"{var}__LISA"]))
    geo2cls = {_norm(r["geojson_district"]): cls.get(r["master_sheet_district"], "NS")
               for _, r in cw.iterrows() if pd.notna(r["geojson_district"])}
    gj = json.load(open("data/raw/Ghana_New_260_District.geojson", encoding="utf-8"))
    patches, colors = [], []
    for ft in gj["features"]:
        name = ft["properties"].get("DISTRICT") or ft["properties"].get("name")
        cl = geo2cls.get(_norm(name), "NS")
        geom = ft["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in parts:
            patches.append(MplPoly(np.array(poly[0]), closed=True)); colors.append(LISA_COL.get(cl, "#d9d9d9"))
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="white", linewidth=0.2))
    ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=11, pad=8)
    counts = li[f"{var}__LISA"].value_counts()
    names = {"HH": "High-High", "LL": "Low-Low", "HL": "High-Low", "LH": "Low-High", "NS": "Not significant"}
    handles = [mlines.Line2D([], [], marker="s", linestyle="", markersize=9,
                             markerfacecolor=LISA_COL[k], markeredgecolor="white",
                             label=f"{k}  {names[k]}  (n={int(counts.get(k, 0))})") for k in ["HH", "LL", "HL", "LH", "NS"]]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="center left",
              bbox_to_anchor=(1.0, 0.5), title="LISA cluster", title_fontsize=8.5)
    _save(fig, stem)


def fig_district_importance(title, stem):
    imp = pd.read_csv("outputs/tables/district_ml_importance.csv")
    cv = pd.read_csv("outputs/tables/district_ml_lorocv.csv").set_index("target")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for pl, (ax, label) in zip("AB", zip(axes, ["TFR 15-49", "Adolescent ASFR 15-19"])):
        _panel(ax, pl, dx=-0.22)
        s = imp[imp.target == label].sort_values("perm_importance").tail(6)
        ax.barh(s.feature, s.perm_importance, xerr=s.sd, color="#009E73", error_kw={"lw": 0.7})
        ax.set_xlabel(f"Permutation importance  (LOROCV R² = {cv.loc[label, 'LOROCV_R2']:.2f})")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, stem)


def main():
    os.makedirs(OUT, exist_ok=True)
    panel = pd.read_csv(PANEL); fc = pd.read_csv(FC); bt = pd.read_csv(BT)
    fig_trajectories(panel, "tfr_15_49", "Total fertility rate (children/woman)",
                     "Regional total fertility rate, 1988–2022", "fig1_tfr_trajectories")
    fig_forecast_fans(panel, fc, "tfr_15_49", "TFR (children/woman)",
                      "Probabilistic TFR forecasts to 2030 (Bayesian hierarchical)", "fig2_tfr_forecast_fans")
    fig_forecast_fans(panel, fc, "asfr_15_19", "ASFR 15-19 (births/1000)",
                      "Adolescent fertility forecasts to 2030 (Bayesian hierarchical)", "fig3_asfr_forecast_fans")
    fig_backtest(bt, "Forecast backtest accuracy by method (2022 hold-out)", "fig4_backtest_rmse")
    fig_lisa_map("Spatial clustering of district illiteracy, 2022", "fig5_lisa_illiteracy_map")
    fig_district_importance("District structural determinants of fertility (LOROCV)", "fig6_district_importance")
    print("wrote png+pdf for fig1-6 (titles distinct, legends outside)")


if __name__ == "__main__":
    main()
