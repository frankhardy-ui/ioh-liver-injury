"""step2_table1.py — baseline characteristics by outcome (Step 2, Table 1)."""
import os
import numpy as np
import pandas as pd
from scipy import stats
import config
from data_loader import load

def smd(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return 0.0 if s == 0 else abs(a.mean() - b.mean()) / s

def main():
    df, _ = load()
    df["sex"] = pd.to_numeric(df["sex"].replace({"M": 1, "F": 0}), errors="coerce")
    ev = df.groupby("op_type_6class")["liver_injury"].sum()
    df["op_type_merged"] = df["op_type_6class"].replace(ev[ev < 5].index.tolist(), "Other")
    inj = df["liver_injury"].astype(int) == 1
    N_ALL, N_NO, N_YES = len(df), int((~inj).sum()), int(inj.sum())
    assert (N_ALL, N_YES) == (config.EXPECTED["N"], config.EXPECTED["events"])

    def mq(s, dp):
        s = pd.Series(s).dropna()
        return ("%.{}f [%.{}f, %.{}f]".format(dp, dp, dp)) % (
            s.median(), s.quantile(.25), s.quantile(.75))

    def cat_ind(ind):
        d = pd.DataFrame({"v": pd.Series(ind, index=df.index).astype(float),
                          "y": df["liver_injury"].astype(int)}).dropna()
        tab = pd.crosstab(d["v"], d["y"])
        p = (stats.fisher_exact(tab).pvalue if (tab.values < 5).any()
             else stats.chi2_contingency(tab).pvalue)
        f = lambda s: "%d (%.1f)" % (int(s["v"].sum()), 100 * s["v"].mean())
        return (f(d), f(d[d["y"] == 0]), f(d[d["y"] == 1]), round(p, 3),
                round(smd(d.loc[d["y"] == 1, "v"], d.loc[d["y"] == 0, "v"]), 3))

    rows = []
    CONT = [("age", "Age, years", 0), ("bmi", "BMI, kg m-2", 1), ("asa", "ASA classification", 0),
            ("preop_hb", "Preoperative haemoglobin, g dL-1", 1),
            ("preop_plt", "Preoperative platelet count, 10^3 uL-1", 0),
            ("preop_pt", "Preoperative PT, INR", 2),
            ("preop_alb", "Preoperative albumin, g dL-1", 2),
            ("preop_ast", "Preoperative AST, IU L-1", 1),
            ("preop_alt", "Preoperative ALT, IU L-1", 1),
            ("preop_bun", "Preoperative urea, mg dL-1", 1),
            ("preop_cr", "Preoperative creatinine, mg dL-1", 2),
            ("op_duration_min", "Operation duration, min", 0),
            ("intraop_crystalloid", "Intraoperative crystalloid, mL", 0),
            ("intraop_ebl", "Intraoperative estimated blood loss, mL", 0),
            ("intraop_uo", "Intraoperative urine output, mL", 0),
            ("mean_map_filt", "Intraoperative mean MAP, mm Hg", 1),
            ("min_map_filt", "Intraoperative minimum MAP, mm Hg", 1),
            ("tib", "TIB, mm Hg.min", 2)]
    for var, lab, dp in CONT:
        if var not in df.columns:
            continue
        a, b = df.loc[~inj, var].dropna(), df.loc[inj, var].dropna()
        rows.append(dict(Variable=lab, Overall=mq(df[var], dp), No_Injury=mq(a, dp),
                         Injury=mq(b, dp), P=round(stats.mannwhitneyu(a, b).pvalue, 3),
                         SMD=round(smd(a, b), 3)))

    for var, lab in [("sex", "Sex, male"), ("preop_htn", "Preoperative hypertension"),
                     ("preop_dm", "Preoperative diabetes mellitus"),
                     ("vaso_active", "Vasoactive agent use"),
                     ("intraop_rbc_binary", "Intraoperative red blood cell transfusion")]:
        if var in df.columns:
            o, n0, n1, p, s = cat_ind(df[var])
            rows.append(dict(Variable=lab, Overall=o, No_Injury=n0, Injury=n1, P=p, SMD=s))
    for lev in ["General surgery", "Urology", "Other"]:
        o, n0, n1, p, s = cat_ind(df["op_type_merged"].eq(lev))
        rows.append(dict(Variable="Surgery type: " + lev, Overall=o, No_Injury=n0,
                         Injury=n1, P=p, SMD=s))
    for lev in ["<30", "30-60", ">=60"]:
        o, n0, n1, p, s = cat_ind(df["tib_group"].eq(lev))
        rows.append(dict(Variable="TIB category, mm Hg.min: " + lev, Overall=o,
                         Injury=n1, No_Injury=n0, P=p, SMD=s))

    t1 = pd.DataFrame(rows)
    t1.columns = ["Variable", "All (n=%d)" % N_ALL, "No liver injury (n=%d)" % N_NO,
                  "Liver injury (n=%d)" % N_YES, "P value", "SMD"]
    t1.to_excel(os.path.join(config.OUT, "table1.xlsx"), index=False)
    t1.to_csv(os.path.join(config.RESULT, "step2_table1.csv"), index=False)
    print(t1.to_string(index=False))
    return {"N": N_ALL, "events": N_YES}

if __name__ == "__main__":
    main()