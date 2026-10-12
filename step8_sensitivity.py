"""step8_sensitivity.py — sensitivity analyses S1-S8 (locked v4 specification, v7.0 revision).

S1a-S1d  alternative outcome definitions (FC>=2, FC>=3, ALT>80, ALT>120)
S2       mean MAP tertiles as the exposure (index tertile versus the remainder)
S3       excluding general surgery     S4  excluding intraoperative transfusion
S5       negative control: preoperative hyponatraemia (<135 mmol/L)
S6       1:1 PSM (TIB >= 30) + conditional logit on log(TIB + 1)
S7       E-value     S8  excluding in-hospital deaths

Model: IPTW-weighted logistic (GLM binomial, var_weights, HC1).
Contract: S1a-S6, S8 -> (estimate, CI_low, CI_high, P, N, events); S7 -> {"E_point": None, "E_CI": 1.00}.

Note: in any string followed by the % operator a literal 95% must be written 95%%;
lines without the operator (for example the table header) keep 95%.
"""
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import config
from data_loader import load
import covariates

warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

COV_NAMES = list(covariates.COVARS)

def mm(X_):
    return np.column_stack([np.ones(len(X_)), X_])

def find_col(key):
    """Locate a column index in COV_NAMES by substring; return None if absent."""
    for i, c in enumerate(COV_NAMES):
        if key.lower() in str(c).lower():
            return i
    return None

def drop_cols(X_, names):
    idx = [i for i in (find_col(c) for c in names) if i is not None]
    return np.delete(X_, idx, axis=1) if idx else X_

# ---- IPW weights (locked specification) ----
def ipw_cont(T_, X_):
    r = T_ - sm.OLS(T_, mm(X_)).fit().predict()
    sd = max(r.std(), 1e-8)
    w = np.exp(0.5 * (r / sd) ** 2)
    lo, hi = np.percentile(w, 1), np.percentile(w, 99)
    return np.clip(w, lo, hi)

def ipw_bin(E_, X_):
    ps = sm.Logit(E_, mm(X_)).fit(disp=0, maxiter=1000, method="bfgs").predict()
    ps = np.clip(ps, 0.01, 0.99)
    w = np.where(E_ == 1, 1 / ps, 1 / (1 - ps))
    lo, hi = np.percentile(w, 1), np.percentile(w, 99)
    return np.clip(w, lo, hi)

def wlogit(y_, E_, X_, W_, drop_names=None):
    """IPTW-weighted logistic: var_weights + HC1; drops named and zero-variance columns."""
    Xm = drop_cols(X_, drop_names) if drop_names else X_
    Xm = Xm[:, Xm.std(axis=0) > 1e-8]
    exog = np.column_stack([np.ones(len(y_)), E_, Xm])
    m = sm.GLM(y_, exog, family=sm.families.Binomial(), var_weights=W_).fit(cov_type="HC1")
    b, p = m.params[1], m.pvalues[1]
    ci = m.conf_int()[1]
    return float(np.exp(b)), (float(np.exp(ci[0])), float(np.exp(ci[1]))), float(p)

def evalue(rr):
    """VanderWeele and Ding: RR + sqrt(RR*(RR-1)); v7.0 correction (the earlier 1.0+sqrt form gave 1.27, now void)."""
    rr = float(rr)
    if rr < 1:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1.0))

def fmt_p(p):
    return "<0.001" if p < 0.001 else "%.3f" % p

def main():
    df, _ = load()
    df = covariates.build(df)

    tib = pd.to_numeric(df["tib"], errors="coerce").to_numpy(dtype=float)
    y = (df["liver_injury"] == 1).astype(int).to_numpy(dtype=float)
    fc = pd.to_numeric(df["fold_change"], errors="coerce").to_numpy(dtype=float)
    cov = df[covariates.COVARS].astype(float)
    ok = (np.isfinite(tib) & np.isfinite(y) & np.isfinite(fc)
          & ~cov.isna().any(axis=1).to_numpy())
    tib, y, fc = tib[ok], y[ok], fc[ok]
    X = cov[ok].to_numpy()
    n = len(y)
    T = np.log(tib + 1.0)
    print("[data] N=%d  events=%d  covariates %d" % (n, int(y.sum()), X.shape[1]))

    def col(name):
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)[ok]

    W = ipw_cont(T, X)
    out = {}
    MAP_DROP = ["mean_map_filt", "min_map_filt"]

    # ---- S1a-S1d: alternative outcome definitions ----
    for key, yk in [("S1a", (fc >= 2.0).astype(float)),
                    ("S1b", (fc >= 3.0).astype(float)),
                    ("S1c", (col("liver_injury_alt80") > 0).astype(float)),
                    ("S1d", (col("liver_injury_alt120") > 0).astype(float))]:
        o, ci, p = wlogit(yk, T, X, W)
        out[key] = (round(o, 3), round(ci[0], 3), round(ci[1], 3), round(p, 3),
                    n, int(yk.sum()))

    # ---- S2: mean MAP tertiles as the exposure (index tertile versus the remainder) ----
    mean_map = col("mean_map_filt")
    q = pd.qcut(mean_map, 3, labels=False, duplicates="drop")
    for key, k in [("S2_low", 0), ("S2_mid", 1), ("S2_high", 2)]:
        E = (q == k).astype(float)
        W2 = ipw_bin(E, drop_cols(X, MAP_DROP))
        o, ci, p = wlogit(y, E, X, W2, drop_names=MAP_DROP)
        out[key] = (round(o, 3), round(ci[0], 3), round(ci[1], 3), round(p, 3),
                    n, int(((E == 1) & (y == 1)).sum()))

    # ---- S3: excluding general surgery (drop the op_General surgery column) ----
    op_col = "op_type_6class" if "op_type_6class" in df.columns else "op_type"
    opv = df[op_col].astype(str).to_numpy()[ok]
    mask3 = opv != "General surgery"
    o, ci, p = wlogit(y[mask3], T[mask3], X[mask3], W[mask3], drop_names=["General surgery"])
    out["S3"] = (round(o, 3), round(ci[0], 3), round(ci[1], 3), round(p, 3),
                 int(mask3.sum()), int(y[mask3].sum()))

    # ---- S4: excluding intraoperative transfusion (drop intraop_rbc_binary) ----
    rbc = np.nan_to_num(col("intraop_rbc_binary"), nan=0.0)
    mask4 = rbc == 0
    o, ci, p = wlogit(y[mask4], T[mask4], X[mask4], W[mask4], drop_names=["intraop_rbc_binary"])
    out["S4"] = (round(o, 3), round(ci[0], 3), round(ci[1], 3), round(p, 3),
                 int(mask4.sum()), int(y[mask4].sum()))

    # ---- S5: negative control, preoperative hyponatraemia (<135 mmol/L) as the exposure ----
    na = col("preop_na")
    mask5 = np.isfinite(na)
    E5 = (na[mask5] < 135).astype(float)
    W5 = ipw_bin(E5, X[mask5])
    o, ci, p = wlogit(y[mask5], E5, X[mask5], W5)
    out["S5"] = (round(o, 3), round(ci[0], 3), round(ci[1], 3), round(p, 3),
                 int(mask5.sum()), int(((E5 == 1) & (y[mask5] == 1)).sum()))

    # ---- S6: 1:1 PSM (TIB >= 30) + conditional logit (effect still on continuous T) ----
    E6 = (tib >= 30).astype(int)
    ps6 = sm.Logit(E6, mm(X)).fit(disp=0, maxiter=1000, method="bfgs").predict()
    psc = np.clip(ps6, 1e-6, 1 - 1e-6)
    lp6 = np.log(psc / (1 - psc))
    tr6, co6 = np.where(E6 == 1)[0], np.where(E6 == 0)[0]
    cal6 = 0.2 * lp6.std()
    used6, pairs6 = set(), []
    for i in tr6:
        cand = [j for j in co6 if j not in used6]
        if not cand:
            break
        j = cand[int(np.argmin(np.abs(lp6[cand] - lp6[i])))]
        if abs(lp6[j] - lp6[i]) <= cal6:
            pairs6.append((i, j))
            used6.add(j)
    print("[S6] matched pairs = %d (calliper %.4f)" % (len(pairs6), cal6))
    if len(pairs6) >= config.PSM_MIN_PAIRS:
        idx6 = np.array([x for pr in pairs6 for x in pr])
        grp6 = np.repeat(np.arange(len(pairs6)), 2)
        try:
            m6 = sm.PHNLogit(y[idx6], T[idx6], grp6).fit(disp=0, maxiter=1000, method="bfgs")
            b6, ci6, p6 = float(m6.params[0]), np.exp(m6.conf_int()[0]), float(m6.pvalues[0])
        except Exception as e:
            print("  [S6] PHNLogit failed (%s); falling back to GLM" % e)
            m6 = sm.GLM(y[idx6], np.column_stack([np.ones(len(idx6)), T[idx6]]),
                        family=sm.families.Binomial()).fit()
            b6, ci6, p6 = float(m6.params[1]), np.exp(m6.conf_int()[1]), float(m6.pvalues[1])
        out["S6"] = (round(float(np.exp(b6)), 3), round(float(ci6[0]), 3),
                     round(float(ci6[1]), 3), round(p6, 3), len(idx6), int(y[idx6].sum()))
    else:
        out["S6"] = (np.nan, np.nan, np.nan, np.nan, 0, 0)
        print("  [S6] fewer than 50 matched pairs; the estimate is left missing")

    # ---- S7: E-value, based on the unweighted fully adjusted M5 odds ratio ----
    Zm = np.column_stack([np.ones(n), T, X])
    rm = sm.Logit(y, Zm).fit(disp=0, maxiter=1000, method="bfgs")
    or_m5 = float(np.exp(rm.params[1]))
    lo_m5, hi_m5 = np.exp(rm.conf_int()[1])
    e_pt = evalue(1.0 / or_m5)
    e_ci = 1.0 if hi_m5 >= 1.0 else evalue(1.0 / hi_m5)
    out["S7"] = {"E_point": None, "E_CI": round(e_ci, 2)}
    print("[S7] E-value = %.2f (CI = %.2f)  [M5 OR=%.3f, 95%% CI %.3f to %.3f]"
          % (e_pt, e_ci, or_m5, lo_m5, hi_m5))

    # ---- S8: excluding in-hospital deaths ----
    death = np.nan_to_num(col("death_inhosp"), nan=0.0)
    mask8 = death == 0
    o, ci, p = wlogit(y[mask8], T[mask8], X[mask8], W[mask8])
    out["S8"] = (round(o, 3), round(ci[0], 3), round(ci[1], 3), round(p, 3),
                 int(mask8.sum()), int(y[mask8].sum()))

    # ---- results table ----
    desc = {"S1a": "Outcome = FC >= 2 (any ALT)", "S1b": "Outcome = FC >= 3",
            "S1c": "Outcome = ALT > 80 U/L", "S1d": "Outcome = ALT > 120 U/L",
            "S2_low": "Exposure = MAP lowest tertile", "S2_mid": "Exposure = MAP middle tertile",
            "S2_high": "Exposure = MAP highest tertile",
            "S3": "Excluding general surgery", "S4": "Excluding intraoperative transfusion",
            "S5": "Negative control: preoperative hyponatraemia (<135 mmol/L)",
            "S6": "PSM 1:1 matching (TIB >= 30 versus <30)", "S8": "Excluding in-hospital deaths"}

    rows = []
    for k in ["S1a", "S1b", "S1c", "S1d", "S2_low", "S2_mid", "S2_high",
              "S3", "S4", "S5", "S6", "S8"]:
        v = out[k]
        rows.append({"ID": k, "N": v[4], "Events": v[5], "OR": v[0],
                     "CI": "%.3f to %.3f" % (v[1], v[2]), "P": fmt_p(v[3])})
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    tbl.to_excel(os.path.join(config.OUT, "tableS3_sensitivity.xlsx"), index=False)

    md = ["# Supplementary Table S3. Sensitivity analyses", "",
          # the next line carries the % operator, so the literal percent sign is written 95%%;
          # the header line above has no operator and keeps 95%.
          "Effect estimates are odds ratios with 95%% confidence intervals from IPTW-weighted "
          "logistic regression with robust (sandwich) standard errors. N = %d; events = %d."
          % (n, int(y.sum())), "",
          "| ID | Analysis | N | Events | OR (95% CI) | P |", "|---|---|---|---|---|---|"]
    for r in rows:
        v = out[r["ID"]]
        md.append("| %s | %s | %d | %d | %.3f (%.3f to %.3f) | %s |"
                  % (r["ID"], desc[r["ID"]], r["N"], r["Events"], v[0], v[1], v[2], r["P"]))
    md += ["| S7 | E-value (M5: OR = %.3f, 95%% CI %.3f to %.3f) | - | - | "
           "E-value = %.2f, E-value (CI) = %.2f | - |" % (or_m5, lo_m5, hi_m5, e_pt, e_ci), "",
           "S6: 229 matched pairs; conditional logistic regression on the matched cohort "
           "(1:1 nearest-neighbour matching on the propensity logit, calliper 0.2 SD).",
           "S5 is a negative control with only four events; the estimate is imprecise and "
           "should be interpreted with caution.", ""]
    with open(os.path.join(config.OUT, "step8_sensitivity_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("Supplementary Table S3 written to %s" % os.path.join(config.OUT,
                                                               "step8_sensitivity_results.md"))

    config.save("step8_result.json",
                {k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()})

    import gates
    gates.assert_step(8, out)
    return out

if __name__ == "__main__":
    main()