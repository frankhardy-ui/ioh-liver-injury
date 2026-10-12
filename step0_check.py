"""step0_check.py — data integrity (Step 0)."""
import config
from data_loader import load

def main():
    df, src = load()
    n, ev = len(df), int(df["liver_injury"].sum())
    print("STEP 0 — DATA INTEGRITY\nsource: %s" % src)
    print("N = %d   events = %d   rate = %.1f%%" % (n, ev, 100 * ev / n))
    print("TIB median = %.2f   IQR = %.2f–%.2f" % (df["tib"].median(),
          df["tib"].quantile(.25), df["tib"].quantile(.75)))
    print("anchor caseid=1: tib = %.4f" % df.loc[df["caseid"] == 1, "tib"].iloc[0])
    print("three-category: " + ", ".join("%s = %d" % (k, v) for k, v
          in df["tib_group"].value_counts().sort_index().items()))
    if "coverage_ratio" in df.columns:
        print("coverage ≥ 0.90: %d (%.1f%%)" % (int((df["coverage_ratio"] >= .9).sum()),
              100 * (df["coverage_ratio"] >= .9).mean()))
    return {"N": n, "events": ev}

if __name__ == "__main__":
    main()