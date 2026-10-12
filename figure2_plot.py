# -*- coding: utf-8 -*-
"""figure2_plot.py — renders Figure 2 (two panels) for the VB10.7 manuscript.

Panel A: adjusted odds ratio (solid line) with 95% CI (shaded area);
         reference value 86.27 mmHg·min (dotted line).
Panel B: predicted probability of liver injury (solid line);
         dashed line = overall event rate (13.7%).
All curves are read from figure2_rcs_values.xlsx (columns tib/or/lo/hi/prob) and
the Wald P values from step3_result.json — both written by step3_rcs.py into
config.OUT — so the figure is always consistent with the values table.
Outputs: figure2_rcs.pdf, figure2_rcs.png (300 dpi). Run AFTER step3_rcs.py.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data_loader

FIG_PDF = os.path.join(config.OUT, "figure2_rcs.pdf")
FIG_PNG = os.path.join(config.OUT, "figure2_rcs.png")

def main():
    # ---- values table produced by step3_rcs.py ----
    vals = pd.read_excel(os.path.join(config.OUT, "figure2_rcs_values.xlsx"))
    for col in ("tib", "or", "lo", "hi", "prob"):
        assert col in vals.columns, \
            "figure2_rcs_values.xlsx missing column: %s " \
            "(expected: tib/or/lo/hi/prob)" % col
    tib_g = vals["tib"].to_numpy(dtype=float)
    or_g = vals["or"].to_numpy(dtype=float)
    lo_g = vals["lo"].to_numpy(dtype=float)
    hi_g = vals["hi"].to_numpy(dtype=float)
    p_g = vals["prob"].to_numpy(dtype=float)

    # ---- reference: median TIB = 86.27 mmHg·min (Figure 2 legend) ----
    REF = float(config.EXPECTED["tib_median"])

    # ---- overall event rate: 195/1427 = 13.7% (Panel B dashed line) ----
    df, _ = data_loader.load()
    ev_rate = float((df["liver_injury"] == 1).mean())

    # ---- Wald P values from step3_result.json (config.save writes to OUT) ----
    with open(os.path.join(config.OUT, "step3_result.json"), encoding="utf-8") as f:
        s3 = json.load(f)
    gp, nlp = float(s3["global_p"]), float(s3["nonlinear_p"])

    # ---- figure ----
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    # Panel A: adjusted OR with 95% CI
    ax = axes[0]
    ax.fill_between(tib_g, lo_g, hi_g, alpha=0.20, color="steelblue",
                    label="95% CI")
    ax.plot(tib_g, or_g, color="steelblue", linewidth=1.8, label="Adjusted OR")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(REF, color="black", linestyle=":", linewidth=1.2)
    ax.text(REF, 1.02, "Reference 86.27", fontsize=8, rotation=90, va="bottom")
    ax.text(0.03, 0.95, "A", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")
    ax.set_xscale("log")
    ax.set_xlim(tib_g.min(), tib_g.max())
    ax.set_xlabel("TIB (mmHg·min)")
    ax.set_ylabel("Adjusted odds ratio")
    ax.legend(loc="upper right", fontsize=8)

    # Panel B: predicted probability
    ax2 = axes[1]
    ax2.plot(tib_g, p_g, color="steelblue", linewidth=1.8)
    ax2.axhline(ev_rate, color="gray", linestyle="--", linewidth=1.2,
                label="Overall event rate (%.1f%%)" % (ev_rate * 100))
    ax2.axvline(REF, color="black", linestyle=":", linewidth=1.2)
    ax2.text(0.03, 0.95, "B", transform=ax2.transAxes, fontsize=12,
             fontweight="bold", va="top")
    ax2.set_xscale("log")
    ax2.set_xlim(tib_g.min(), tib_g.max())
    ax2.set_xlabel("TIB (mmHg·min)")
    ax2.set_ylabel("Predicted probability")
    ax2.legend(loc="lower left", fontsize=8)

    fig.suptitle("Global P = %.3f; non-linearity P = %.3f" % (gp, nlp), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG, dpi=300)
    print("[Figure 2] saved %s and %s" % (FIG_PDF, FIG_PNG))
    print("[Figure 2] global P = %.3f, non-linearity P = %.3f" % (gp, nlp))
    print("[Figure 2] TIB grid %.2f to %.1f mmHg·min (manuscript observed "
          "exposure range 0.6 to 819)" % (tib_g.min(), tib_g.max()))

if __name__ == "__main__":
    main()