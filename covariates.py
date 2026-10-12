"""covariates.py - covariate derivation and the locked 19-variable set."""
import pandas as pd

BASE = ["age", "bmi", "asa", "preop_htn", "preop_dm", "preop_pt",
        "preop_alb", "preop_ast", "preop_alt", "preop_bun", "preop_cr",
        "op_duration_min", "vaso_active", "intraop_rbc_binary",
        "mean_map_filt", "min_map_filt"]

OP_COLS = ["op_General surgery", "op_Urology"]

COVARS = BASE + ["sex"] + OP_COLS
assert len(COVARS) == 19

def build(df):
    df = df.copy()
    df["sex"] = (df["sex"].astype(str).str.strip().str.upper()
                 .map({"M": 1, "MALE": 1, "F": 0, "FEMALE": 0})
                 .fillna(pd.to_numeric(df["sex"], errors="coerce")).astype(int))
    src = "op_type_6class" if "op_type_6class" in df.columns else "op_type"
    op = df[src].map(lambda s: s if s in ("General surgery", "Urology") else "Other")
    for c in OP_COLS:
        df[c] = (op == c.replace("op_", "")).astype(int)
    missing = [c for c in COVARS if c not in df.columns]
    if missing:
        raise KeyError(f"covariates absent: {missing}")
    return df