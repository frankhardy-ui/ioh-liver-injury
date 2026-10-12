"""step7_mediation.py — causal mediation (Step 7).

Causal chain: log(TIB+1) -> liver injury (binary mediator) -> log(LOS) (continuous outcome).
LOS: los_days = (dis - adm)/86400; log_los = log(max(los_days, 0.1)).
Method: natural effect model (Imai et al. 2010); path a: logit; path b+c': OLS.
        TE = a*b + c'; NDE = c'; NIE = a*b; PM = NIE/TE; bootstrap 500 (percentile 95% CI).
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import config
from data_loader import load
import covariates

def main():
    # legacy global random state, as in the locked version (not default_rng)
    np.random.seed(config.SEED)

    df, _ = load()
    df = covariates.build(df)

    # ---- exposure, mediator, outcome ----
    tib = pd.to_numeric(df["tib"], errors="coerce").to_numpy(dtype=float)
    y_med = (df["liver_injury"] == 1).astype(int).to_numpy(dtype=int)
    los_days = (pd.to_numeric(df["dis"], errors="coerce")
                - pd.to_numeric(df["adm"], errors="coerce")).to_numpy(dtype=float) / 86400.0
    y_out = np.log(np.maximum(los_days, 0.1))

    cov = df[covariates.COVARS].astype(float)
    ok = (np.isfinite(tib) & np.isfinite(y_out) & np.isfinite(y_med)
          & ~cov.isna().any(axis=1).to_numpy())
    y_med, y_out, tib = y_med[ok], y_out[ok], tib[ok]
    X = cov[ok].to_numpy()
    n = len(y_out)
    T = np.log(tib + 1.0)
    n_events = int(y_med.sum())

    print("[data] N=%d  mediator events=%d (%.1f%%)  covariates %d  "
          "[LOS=(dis-adm)/86400 d, log, floor 0.1 d]"
          % (n, n_events, 100 * n_events / n, X.shape[1]))
    if n_events < 10:
        raise SystemExit("mediator events < 10")

    # ---- path a: exposure -> mediator (logit) ----
    med_model = sm.Logit(y_med, np.column_stack([np.ones(n), T, X]))
    med_model = med_model.fit(disp=0, maxiter=1000, method="bfgs")
    a_coef = float(med_model.params[1])
    a_p = float(med_model.pvalues[1])
    print("[path a] coef=%+.4f OR=%.4f P=%.4f" % (a_coef, np.exp(a_coef), a_p))

    # ---- path b + c': mediator + exposure -> outcome (OLS) ----
    out_model = sm.OLS(y_out, np.column_stack([np.ones(n), T, y_med, X])).fit()
    b_coef = float(out_model.params[2]); b_p = float(out_model.pvalues[2])
    cp_coef = float(out_model.params[1]); cp_p = float(out_model.pvalues[1])
    print("[path b] coef=%+.4f P=%.4f" % (b_coef, b_p))
    print("[path c'] coef=%+.4f P=%.4f" % (cp_coef, cp_p))

    # ---- point estimates ----
    te = a_coef * b_coef + cp_coef
    nde = cp_coef
    nie = a_coef * b_coef
    pm = nie / te if abs(te) > 1e-10 else np.nan
    print("[effects] TE=%+.4f  NDE=%+.4f  NIE=%+.4f  PM=%.1f%%" % (te, nde, nie, pm * 100))

    # ---- bootstrap 500 (percentile CI) ----
    print("[Bootstrap] 500 replicates...")
    te_bs, nde_bs, nie_bs, pm_bs = [], [], [], []
    for _ in range(config.BOOT_MED):
        idx = np.random.choice(n, n, replace=True)
        T_b, X_b = T[idx], X[idx]
        y_med_b, y_out_b = y_med[idx], y_out[idx]
        try:
            m_b = sm.Logit(y_med_b, np.column_stack([np.ones(n), T_b, X_b]))
            m_b = m_b.fit(disp=0, maxiter=1000, method="bfgs")
            a_b = float(m_b.params[1])
        except Exception:
            continue
        try:
            o_b = sm.OLS(y_out_b, np.column_stack([np.ones(n), T_b, y_med_b, X_b])).fit()
            b_b = float(o_b.params[2]); cp_b = float(o_b.params[1])
        except Exception:
            continue
        te_b = a_b * b_b + cp_b
        te_bs.append(te_b); nde_bs.append(cp_b); nie_bs.append(a_b * b_b)
        if abs(te_b) > 1e-10:
            pm_bs.append(a_b * b_b / te_b)

    valid = len(te_bs)
    print("[Bootstrap] valid replicates = %d" % valid)

    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)
    te_ci, nde_ci, nie_ci = ci(te_bs), ci(nde_bs), ci(nie_bs)
    _ = ci(pm_bs) if len(pm_bs) > 50 else (np.nan, np.nan)   # unstable because TE is near zero; not reported

    print("[result] TE=%+.4f [%+.4f, %+.4f]" % (te, te_ci[0], te_ci[1]))
    print("[result] NDE=%+.4f [%+.4f, %+.4f]" % (nde, nde_ci[0], nde_ci[1]))
    print("[result] NIE=%+.4f [%+.4f, %+.4f]" % (nie, nie_ci[0], nie_ci[1]))
    print("[result] PM=%.1f%% (confidence interval not estimated: TE near zero)" % (pm * 100))

    # [reconstructed] ---- write Table 4 ----
    tbl = pd.DataFrame([
        dict(Effect="Total effect (TE)", Estimate=round(te, 4),
             CI_low=round(te_ci[0], 4), CI_high=round(te_ci[1], 4),
             Method="Natural effect model, bootstrap %d" % config.BOOT_MED),
        dict(Effect="Natural direct effect (NDE)", Estimate=round(nde, 4),
             CI_low=round(nde_ci[0], 4), CI_high=round(nde_ci[1], 4),
             Method="Natural effect model, bootstrap %d" % config.BOOT_MED),
        dict(Effect="Natural indirect effect (NIE)", Estimate=round(nie, 4),
             CI_low=round(nie_ci[0], 4), CI_high=round(nie_ci[1], 4),
             Method="Natural effect model, bootstrap %d" % config.BOOT_MED),
        dict(Effect="Proportion mediated, %", Estimate=round(pm * 100, 1),
             CI_low=np.nan, CI_high=np.nan,
             Method="Point estimate only; the interval is not estimated because TE is near zero"),
    ])
    with pd.ExcelWriter(os.path.join(config.OUT, "table4_mediation.xlsx")) as xw:
        tbl.to_excel(xw, sheet_name="Mediation", index=False)
        pd.DataFrame([dict(Parameter="Path a (exposure to mediator)",
                           Estimate=round(a_coef, 4), OR=round(float(np.exp(a_coef)), 4),
                           P=round(a_p, 4)),
                      dict(Parameter="Path b (mediator to outcome)",
                           Estimate=round(b_coef, 4), OR=np.nan, P=round(b_p, 4)),
                      dict(Parameter="Direct effect c'",
                           Estimate=round(cp_coef, 4), OR=np.nan, P=round(cp_p, 4))]
                     ).to_excel(xw, sheet_name="Paths", index=False)
    print("[Table 4] %s" % os.path.join(config.OUT, "table4_mediation.xlsx"))

    md = ["# Step 7 — causal mediation (natural effect model)", "",
          "Chain: log(TIB+1) to liver injury to log(LOS); LOS = (dis - adm)/86400 days, "
          "floored at 0.1 day. Adjustment set A' (19 covariates) in both models. "
          "Bootstrap %d replicates, percentile 95%% confidence intervals." % config.BOOT_MED, "",
          "| Effect | Estimate | 95% CI |", "|---|---|---|",
          "| Total effect | %+.4f | %+.4f to %+.4f |" % (te, te_ci[0], te_ci[1]),
          "| Natural direct effect | %+.4f | %+.4f to %+.4f |" % (nde, nde_ci[0], nde_ci[1]),
          "| Natural indirect effect | %+.4f | %+.4f to %+.4f |" % (nie, nie_ci[0], nie_ci[1]),
          "| Proportion mediated, %% | %.1f | not estimated (TE near zero) |" % (pm * 100), "",
          "The proportion mediated is negative because the direct and indirect effects "
          "act in opposite directions; it should not be read as a conventional proportion.",
          ""]
    with open(os.path.join(config.OUT, "step7_mediation_results.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))

    res = {"TE": (round(te, 4), round(te_ci[0], 4), round(te_ci[1], 4)),
           "NDE": round(nde, 4), "NIE": round(nie, 4), "PM": round(pm * 100, 1),
           "N": n, "events": n_events, "boot": config.BOOT_MED}
    import gates
    gates.assert_step(7, res)
    return res

if __name__ == "__main__":
    main()