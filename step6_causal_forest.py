"""step6_causal_forest.py — causal forest heterogeneity (Step 6, Figure 3 / Table 3).
EconML GRF, 500 trees / max_depth 15 / min_samples_leaf 10 / 5-fold cross-fitting.
Treatment T = log(TIB + 1); outcome = liver_injury; covariates = the 19 A' set.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from econml.grf import CausalForest          # noqa: E402  (heavy import, kept late)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import econml
import config
from data_loader import load
import covariates

def main():
    print("[API] econml version = %s" % econml.__version__)

    df, _ = load()
    df = covariates.build(df)
    y = (df["liver_injury"] == 1).astype(float).to_numpy()
    tib = df["tib"].to_numpy(dtype=float)
    ok = np.isfinite(tib)
    y, tib = y[ok], tib[ok]
    cov = df.loc[ok, covariates.COVARS].astype(float)
    assert cov.isna().sum().sum() == 0
    X, T = cov.to_numpy(), np.log(tib + 1.0)
    n = len(y)

    kw = dict(n_estimators=500, max_depth=15, min_samples_leaf=10, random_state=config.SEED)
    kf = KFold(n_splits=config.N_FOLD, shuffle=True, random_state=config.SEED)

    cates = np.full(n, np.nan)
    lbs = np.full(n, np.nan)
    ubs = np.full(n, np.nan)
    ci_ok, imps = False, []
    for tr, va in kf.split(X):
        cf = CausalForest(**kw)
        cf.fit(X[tr], y[tr], T[tr])
        try:
            pr, lb, ub = cf.predict(X[va], interval=True)
            pr, lb, ub = map(lambda a: np.asarray(a).reshape(-1), (pr, lb, ub))
            assert len(pr) == len(va)
            lbs[va], ubs[va] = lb, ub
            ci_ok = True
        except (TypeError, ValueError):
            pr = np.asarray(cf.predict(X[va])).reshape(-1)
        cates[va] = pr
        fi = getattr(cf, "feature_importances_", None)
        if fi is not None:
            imps.append(np.asarray(fi, dtype=float))

    # [reconstructed] variable importance averaged over the folds and expressed as a
    # percentage of the total (contract: top3 = 34.2 / 33.1 / 11.0).
    if imps:
        imp = np.mean(np.vstack(imps), axis=0)
    else:
        imp = np.zeros(X.shape[1])
    imp = np.clip(imp, 0, None)
    tot = imp.sum() if imp.sum() > 0 else 1.0
    imp_df = pd.DataFrame({"Variable": list(covariates.COVARS),
                           "Importance": imp / tot})
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    top3 = [round(float(v) * 100, 1) for v in imp_df["Importance"].head(3)]
    print("[importance] top-3: " + ", ".join(
        "%s %.1f%%" % (r["Variable"], r["Importance"] * 100)
        for _, r in imp_df.head(3).iterrows()))

    # [reconstructed] overall ATE with the forest prediction-interval bounds, as
    # described in the supplementary footnote (pointwise interval means).
    ate = float(np.mean(cates))
    if ci_ok and np.isfinite(lbs).all():
        ate_ci = (float(np.mean(lbs)), float(np.mean(ubs)))
    else:
        se = float(np.std(cates, ddof=1) / np.sqrt(n))
        ate_ci = (ate - 1.96 * se, ate + 1.96 * se)

    # [reconstructed] four pre-specified modifiers (median split for continuous
    # variables, reported categories for surgery type); anaesthesia type excluded.
    def _stratum(name, mask, label):
        if mask.sum() == 0:
            return None
        c = cates[mask]
        lo = float(np.mean(lbs[mask])) if ci_ok and np.isfinite(lbs[mask]).all() else np.nan
        hi = float(np.mean(ubs[mask])) if ci_ok and np.isfinite(ubs[mask]).all() else np.nan
        return {"Subgroup": name, "Stratum": label, "n": int(mask.sum()),
                "Events": int(y[mask].sum()), "Event Rate": round(100 * float(y[mask].mean()), 1),
                "CATE": round(float(np.mean(c)), 3),
                "CI_lower": round(lo, 3) if np.isfinite(lo) else np.nan,
                "CI_upper": round(hi, 3) if np.isfinite(hi) else np.nan}

    rows = []
    alt = cov["preop_alt"].to_numpy()
    age = cov["age"].to_numpy()
    dm = cov["preop_dm"].to_numpy()
    gs = cov["op_General surgery"].to_numpy()
    ur = cov["op_Urology"].to_numpy()
    for r in (_stratum("Preoperative ALT", alt >= np.median(alt), ">= median"),
              _stratum("Preoperative ALT", alt < np.median(alt), "< median"),
              _stratum("Age", age >= np.median(age), ">= median"),
              _stratum("Age", age < np.median(age), "< median"),
              _stratum("Diabetes mellitus", dm == 1, "Yes"),
              _stratum("Diabetes mellitus", dm == 0, "No"),
              _stratum("Surgery type", gs == 1, "General surgery"),
              _stratum("Surgery type", ur == 1, "Urology"),
              _stratum("Surgery type", (gs == 0) & (ur == 0), "Other")):
        if r is not None:
            rows.append(r)
    sub_df = pd.DataFrame(rows)
    pd.DataFrame(rows).to_csv(os.path.join(config.RESULT, "step6_subgroups.csv"), index=False)

    print("[ATE] %+.4f (95%% CI %+.4f to %+.4f)" % (ate, *ate_ci))

    # ---- Figure 3A: variable importance ----
    d = imp_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(d["Variable"], d["Importance"] * 100, color="#4C72B0")
    ax.set_xlabel("Variable importance (%)")
    for yy, v in zip(range(len(d)), d["Importance"] * 100):
        ax.text(v + 0.4, yy, "%.1f%%" % v, va="center", fontsize=8)
    ax.set_xlim(0, max(d["Importance"] * 100) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(config.OUT, "figure3a_variable_importance.pdf"))
    fig.savefig(os.path.join(config.OUT, "figure3a_variable_importance.png"), dpi=300)
    plt.close(fig)

    # [reconstructed] ---- Figure 3B: subgroup CATEs ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    yy = np.arange(len(sub_df))
    ax.errorbar(sub_df["CATE"], yy,
                xerr=[sub_df["CATE"] - sub_df["CI_lower"], sub_df["CI_upper"] - sub_df["CATE"]],
                fmt="o", color="#4C72B0", ecolor="#4C72B0", capsize=3)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_yticks(yy)
    ax.set_yticklabels(["%s: %s" % (r["Subgroup"], r["Stratum"])
                        for _, r in sub_df.iterrows()], fontsize=8)
    ax.set_xlabel("Conditional average treatment effect (CATE)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.OUT, "figure3b_subgroup_cate.pdf"))
    fig.savefig(os.path.join(config.OUT, "figure3b_subgroup_cate.png"), dpi=300)
    plt.close(fig)

    # ---- Table S2 ----
    with pd.ExcelWriter(os.path.join(config.OUT, "tableS2_forest.xlsx")) as xw:
        imp_df.assign(Importance_pct=imp_df["Importance"] * 100).to_excel(
            xw, sheet_name="Variable importance", index=False)
        sub_df.to_excel(xw, sheet_name="Subgroup CATE", index=False)
        pd.DataFrame([dict(ATE=round(ate, 3), CI_low=round(ate_ci[0], 3),
                           CI_high=round(ate_ci[1], 3), N=n, Events=int(y.sum()),
                           trees=500, max_depth=15, min_samples_leaf=10,
                           folds=config.N_FOLD)]).to_excel(
            xw, sheet_name="Overall", index=False)

    md = ["# Step 6 — causal forest heterogeneity", "",
          "EconML GRF: 500 trees, max depth 15, minimum leaf size 10, %d-fold "
          "cross-fitting. Treatment log(TIB+1); outcome liver injury; 19 covariates."
          % config.N_FOLD, "",
          "| Subgroup | Stratum | n | Events | Event Rate | CATE | 95% CI |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s | %d | %d | %.1f | %+.3f | %+.3f to %+.3f |"
                  % (r["Subgroup"], r["Stratum"], r["n"], r["Events"], r["Event Rate"],
                     r["CATE"], r["CI_lower"], r["CI_upper"]))
    md += ["", "| **Overall** | **ATE** | **%d** | **%d** | **%.1f** | **%+.3f** | **%+.3f to %+.3f** |"
           % (n, int(y.sum()), 100 * y.sum() / n, ate, ate_ci[0], ate_ci[1]), ""]
    with open(os.path.join(config.OUT, "step6_causal_forest_results.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))
    print("Figure 3 and Table S2 written to %s" % config.OUT)

    out = {"top3": top3, "ATE": (round(ate, 3), ate_ci[0], ate_ci[1]),
           "trees": 500, "max_depth": 15, "min_leaf": 10, "folds": config.N_FOLD}
    import gates
    gates.assert_step(6, out)
    return out

if __name__ == "__main__":
    main()