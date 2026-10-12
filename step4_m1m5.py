"""step4_m1m5.py — sequential adjustment M1 → M5 (Step 4)."""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import config
from data_loader import load
import covariates

BLOCKS = [("M1: Unadjusted", []),
          ("M2: +Demographics", ["age", "sex", "bmi", "asa"]),
          ("M3: +Comorbidities", ["preop_htn", "preop_dm"]),
          ("M4: +Preop laboratory", ["preop_pt", "preop_alb", "preop_ast",
                                     "preop_alt", "preop_bun", "preop_cr"]),
          ("M5: +Intraoperative", ["op_duration_min", "vaso_active", "intraop_rbc_binary",
                                   "mean_map_filt", "min_map_filt",
                                   "op_General surgery", "op_Urology"])]

def main():
    df, _ = load(); df = covariates.build(df)
    y = (df["liver_injury"] == 1).astype(int).to_numpy()
    tib = df["tib"].to_numpy(dtype=float)
    ok = np.isfinite(tib)
    y, tib = y[ok], tib[ok]
    cov = df.loc[ok, covariates.COVARS].astype(float)
    assert cov.isna().sum().sum() == 0
    log_tib = np.log(tib + 1.0)
    e = int(y.sum())
    res, acc = {}, []

    for name, blk in BLOCKS:
        acc += blk
        X = sm.add_constant(pd.DataFrame(
            np.column_stack([log_tib] + [cov[c].to_numpy() for c in acc]),
            columns=["log_tib"] + list(acc)))
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit()   # model-based SE
        lo, hi = m.conf_int().loc["log_tib"]
        key = name.split(":")[0]
        res[key] = (round(float(np.exp(m.params["log_tib"])), 3), round(float(np.exp(lo)), 3),
                    round(float(np.exp(hi)), 3), round(float(m.pvalues["log_tib"]), 3),
                    round(float(m.aic), 2), round(float(roc_auc_score(y, m.predict(X))), 3))
        print("[%s] OR=%.3f (%.3f–%.3f) P=%.3f AIC=%.2f C=%.3f EPV=%.1f"
              % (key, *res[key][:4], res[key][4], res[key][5], e / (len(m.params) - 1)))
    pd.DataFrame([dict(Model=k, N=len(y), Events=e, OR="%.3f (%.3f–%.3f)" % v[:3],
                       P=v[3], AIC=v[4], C_stat=v[5]) for k, v in res.items()]).to_excel(
        os.path.join(config.OUT, "table2_m1m5.xlsx"), index=False)
    import gates; gates.assert_step(4, res)
    return res

if __name__ == "__main__":
    main()