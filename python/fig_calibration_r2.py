"""
Supplemental figure — distribution of calibration-curve R2 across target peptides.

Requested twice during review: the Results quote only a median R2 of 0.9 for the
linearity of the assay, without showing how that value is distributed.

The curves are refitted here rather than read back from an export because the
application does not persist per-curve R2. The fit deliberately mirrors
calibrate_group() in app_v6.py: weighted least squares of concentration on
reporter-ion intensity with w = 1/x^2, the lowest below-LOD standard clamped to
zero and any other below-LOD standard dropped, and a weighted R2. Reproducing
that logic here keeps the figure runnable without Streamlit; any change to the
application's fit must be mirrored below or the figure will drift from the data.

Input : data/2025-0825_TMT-quant-template_gene.csv
Output: plots/SupplementalFigure_calibration_R2.{png,pdf}
        plots/calibration_r2_per_curve.csv
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT_DIR = "plots"

STANDARD_CONCENTRATION_BY_CHANNEL = {
    "TMT-126": 20.0, "TMT-127N": 3.2, "TMT-128N": 0.8,
    "TMT-129N": 0.1, "TMT-130N": 0.02,
}
BLANK_CHANNELS = {"TMT-131L", "TMT-135"}

BLUE = "#4292c6"
DARK_BLUE = "#084594"
GREEN = "#41ab5d"


def load_reporter_measurements():
    df = pd.read_csv(f"{DATA_DIR}/2025-0825_TMT-quant-template_gene.csv")
    df = df[df["Fragment Ion"].str.startswith("TMT")].copy()
    df["Transition Result"] = pd.to_numeric(df["Transition Result"], errors="coerce")
    return df.dropna(subset=["Transition Result"])


def select_standards_above_lod(group, baseline):
    """Applies the application's below-LOD policy to the calibration standards."""
    known = group[group["Fragment Ion"].isin(STANDARD_CONCENTRATION_BY_CHANNEL)].copy()
    known["conc"] = known["Fragment Ion"].map(STANDARD_CONCENTRATION_BY_CHANNEL)
    below = known["Transition Result"] < baseline
    if below.any():
        lowest = known.loc[below, "Transition Result"].idxmin()
        known.loc[lowest, "Transition Result"] = 0.0
        known = known.loc[~(below & (known.index != lowest))]
    return known


def weighted_r2_of_calibration_curve(group):
    blanks = group[group["Fragment Ion"].isin(BLANK_CHANNELS)]["Transition Result"]
    baseline = blanks.median() if len(blanks) else 0.0
    if pd.isna(baseline):
        baseline = 0.0

    known = select_standards_above_lod(group, baseline)
    if len(known) < 2:
        return None

    x = known["Transition Result"].to_numpy(float)
    y = known["conc"].to_numpy(float)
    # A curve whose standards are all at one intensity carries no slope
    # information; polyfit would return a degenerate fit rather than fail.
    if np.ptp(x) == 0:
        return None
    w = 1.0 / np.maximum(x, 1e-12) ** 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            slope, intercept = np.polyfit(x, y, 1, w=w)
        except Exception:
            return None
    if not np.isfinite([slope, intercept]).all():
        return None

    residual = y - (slope * x + intercept)
    ss_res = float(np.sum(w * residual ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    if not ss_tot > 0:
        return None
    return slope, intercept, 1 - ss_res / ss_tot, len(known)


def fit_all_curves(df):
    records = []
    skipped = 0
    for (run, protein, peptide, precursor), group in df.groupby(
            ["Replicate", "Protein", "Peptide", "Precursor"], sort=False):
        fit = weighted_r2_of_calibration_curve(group)
        if fit is None:
            skipped += 1
            continue
        slope, intercept, r2, n_standards = fit
        records.append(dict(run=run, protein=protein, peptide=peptide,
                            precursor=precursor, slope=slope, intercept=intercept,
                            r2=r2, n_standards=n_standards))
    return pd.DataFrame(records), skipped


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.1)
    ax.tick_params(labelsize=14, width=1.1, length=5, colors="black")


def build_figure(per_peptide_r2, per_curve_r2):
    fig, (ax_hist, ax_cumulative) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    ax_hist.hist(per_peptide_r2, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
    ax_hist.axvline(np.median(per_peptide_r2), color=DARK_BLUE, linewidth=2.0)
    ax_hist.set_xlabel("Calibration curve $R^2$", fontsize=16, fontweight="bold")
    ax_hist.set_ylabel("Target peptides", fontsize=16, fontweight="bold")
    ax_hist.set_xlim(0, 1)
    style_axis(ax_hist)

    ordered = np.sort(per_peptide_r2)
    fraction_at_or_above = 1.0 - np.arange(len(ordered)) / len(ordered)
    ax_cumulative.plot(ordered, 100 * fraction_at_or_above, color=GREEN, linewidth=2.4)
    ax_cumulative.set_xlabel("Calibration curve $R^2$", fontsize=16, fontweight="bold")
    ax_cumulative.set_ylabel("Target peptides at or above (%)",
                             fontsize=16, fontweight="bold")
    ax_cumulative.set_xlim(0, 1)
    ax_cumulative.set_ylim(0, 100)
    style_axis(ax_cumulative)

    for ax, tag in ((ax_hist, "A"), (ax_cumulative, "B")):
        ax.text(-0.20, 1.07, tag, transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="top")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(f"{OUT_DIR}/SupplementalFigure_calibration_R2.png", dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT_DIR}/SupplementalFigure_calibration_R2.pdf",
                bbox_inches="tight", facecolor="white")


def main():
    curves, skipped = fit_all_curves(load_reporter_measurements())
    curves.to_csv(f"{OUT_DIR}/calibration_r2_per_curve.csv", index=False)

    per_peptide = curves.groupby(["protein", "peptide"])["r2"].median()
    build_figure(per_peptide.to_numpy(), curves["r2"].to_numpy())

    print("--- numbers for the calibration R2 figure legend ---")
    print(f"runs                        : {curves['run'].nunique()}")
    print(f"curves fitted               : {len(curves)}")
    print(f"curves skipped (degenerate) : {skipped}")
    print(f"target peptides             : {len(per_peptide)}")
    print(f"proteins                    : {curves['protein'].nunique()}")
    print(f"median R2 per curve         : {curves['r2'].median():.3f}")
    print(f"median R2 per peptide       : {per_peptide.median():.3f}")
    for threshold in (0.8, 0.9, 0.95, 0.99):
        print(f"peptides with R2 >= {threshold:<5}   : "
              f"{(per_peptide >= threshold).sum()} ({100 * (per_peptide >= threshold).mean():.1f}%)")


main()
