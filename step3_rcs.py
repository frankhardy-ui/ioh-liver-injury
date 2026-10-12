"""step3_rcs.py - RCS dose-response (Step 3, Figure 2).

Global and non-linearity P values are WALD tests, matching the manuscript Methods:
the Wald statistic uses the exposure block (linear + spline term) for the global test
and the spline term alone for the non-linearity test.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import config
from data_loader import load
import covariates

def main():
    df, _ = load()
    df = covariates.build(df)
    y = (df["liver_injury"] == 1).astype(float).to_numpy()
    tib = df["tib"].to_numpy(dtype=float)
    ok = np.isfinite(tib)
    y, tib = y[ok], tib[ok]
    cov = df.loc[ok, covariates.COVARS].astype(float)
    assert cov.isna().sum().sum() == 0

    x = np.log(tib + 1.0)
    t1, t2, t3 = np.percentile(x, [10, 50, 90])

    def rcs3(u):
        return (u,
                np.clip(u - t1, 0, None) ** 3
                - np.clip(u - t2, 0, None) ** 3 * (t3 - t1) / (t3 - t2)
                + np.clip(u - t3, 0, None) ** 3 * (t2 - t1) / (t3 - t2))

    x1, x2 = rcs3(x)

    # columns: [const] + covariates + [tib_lin, tib_spline]
    Xc = sm.add_constant(cov.to_numpy())
    X = np.column_stack([Xc, x1, x2])
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    b = np.asarray(fit.params)
    V = np.asarray(fit.cov_params())
    idx_all = [X.shape[1] - 2, X.shape[1] - 1]   # linear + spline
    idx_non = [X.shape[1] - 1]                   # spline only
    w_all = b[idx_all] @ np.linalg.inv(V[np.ix_(idx_all, idx_all)]) @ b[idx_all]
    w_non = b[idx_non] @ np.linalg.inv(V[np.ix_(idx_non, idx_non)]) @ b[idx_non]

    global_p = float(1 - stats.chi2.cdf(w_all, len(idx_all)))
    nonlin_p = float(1 - stats.chi2.cdf(w_non, len(idx_non)))

    print("[RCS] knots: %.1f / %.1f / %.1f" % tuple(np.exp([t1, t2, t3]) - 1))
    print("[RCS] global P = %.3f   non-linearity P = %.3f (Wald)" % (global_p, nonlin_p))

    REF_TIB = (config.EXPECTED["tib_median"] if config.REF_MODE == "median"
               else float(np.quantile(tib, .10)))
    lr_ref = np.log(REF_TIB + 1)
    r1, r2 = rcs3(np.array([lr_ref]))
    cm = np.asarray(cov.mean().values)
    x_ref = np.r_[1.0, cm, r1[0], r2[0]]
    eta_ref = x_ref @ b

    glog = np.linspace(np.log(np.percentile(tib, 1) + 1),
                       np.log(np.percentile(tib, 99) + 1), 300)
    gtib = np.exp(glog) - 1
    g1, g2 = rcs3(glog)
    Xg = np.column_stack([np.ones(len(glog)), np.tile(cm, (len(glog), 1)), g1, g2])
    eta_g = Xg @ b
    se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", Xg - x_ref, V, Xg - x_ref), 0, None))

    pd.DataFrame({"tib": gtib, "or": np.exp(eta_g - eta_ref),
                  "lo": np.exp(eta_g - eta_ref - 1.96 * se),
                  "hi": np.exp(eta_g - eta_ref + 1.96 * se),
                  "prob": 1 / (1 + np.exp(-eta_g))}).to_excel(
        os.path.join(config.OUT, "figure2_rcs_values.xlsx"), index=False)
    knots_tib = [float(np.exp(k) - 1) for k in (t1, t2, t3)]
    config.save("step3_result.json", dict(global_p=global_p, nonlinear_p=nonlin_p,
                knots=knots_tib, ref_tib=REF_TIB, ref_mode=config.REF_MODE))

    import gates
    gates.assert_step(3, {"N": len(y), "events": int(y.sum()),
                          "knots": [round(k, 1) for k in knots_tib],
                          "ref": REF_TIB, "global_P": global_p,
                          "nonlinear_P": nonlin_p})
    return {"N": len(y), "events": int(y.sum()), "ref": REF_TIB,
            "global_P": global_p, "nonlinear_P": nonlin_p}

if __name__ == "__main__":
    main()