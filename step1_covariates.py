"""step1_covariates.py - audit of A-prime (Step 1)."""
import pandas as pd
import config
from data_loader import load
import covariates

def main():
    df, _ = load(); df = covariates.build(df)
    cov = df[covariates.COVARS].astype(float)
    assert cov.isna().sum().sum() == 0, "adjustment set contains missing values"
    assert len(covariates.COVARS) == config.EXPECTED["n_covariates"]
    zero = [c for c in covariates.COVARS if cov[c].std() < 1e-8]
    corr = cov.corr().abs()
    hi = [(a, b) for i, a in enumerate(covariates.COVARS)
          for b in covariates.COVARS[i + 1:] if corr.loc[a, b] > 0.9]
    print(pd.DataFrame({"sd": cov.std(), "min": cov.min(), "max": cov.max()}).to_string())
    print("zero-variance:", zero, "| |r| > 0.9:", hi)
    config.save("step1_covariate_report.json", {"zero_variance": zero,
                "high_correlation": [list(x) for x in hi]})
    n_ev = int(df["liver_injury"].sum())
    epv = n_ev / float(len(covariates.COVARS))
    config.save("step1_result.json", {"n_covariates": len(covariates.COVARS),
                                      "events": n_ev, "EPV": round(epv, 3)})
    print("[step1] complete-case N=%d events=%d EPV=%.3f" % (len(df), n_ev, epv))
    return {"n_covariates": 19}

if __name__ == "__main__":
    main()