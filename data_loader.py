"""data_loader.py — single gated entry point and the TIB pipeline."""
import os
import numpy as np
import pandas as pd
import config

def _trapz(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)

def _load_npz(cid):
    p = os.path.join(config.P0_CACHE, "case_%06d.npz" % cid)
    if not os.path.isfile(p):
        return None
    z = np.load(p)
    return z["t"].astype(float), z["v"].astype(float)

def build_tib_from_cache():
    """Programme 0: 1-s grid MAP track via vitaldb.load_case(interval=1); no
    resampling, interpolation or filtering; artefact window [20, 250];
    trapezoidal integral of (65 - MAP)+ / 60."""
    ds = pd.read_excel(config.P0_DS) if os.path.isfile(config.P0_DS) else None
    caseids = ds["caseid"].astype(int).tolist() if ds is not None else []
    rows = []
    for cid in caseids:
        r = dict(caseid=cid, tib=np.nan, min_map_filt=np.nan,
                 mean_map_filt=np.nan, n_kept=np.nan, median_dt_s=np.nan)
        z = _load_npz(cid)
        if z is None:
            rows.append(r); continue
        t, v = z
        m = np.isfinite(t) & np.isfinite(v)
        t, v = t[m], v[m]
        if len(t) < 2:
            rows.append(r); continue
        r.update(n_kept=int(len(t)), median_dt_s=float(np.median(np.diff(t))))
        m2 = (v >= config.TIB_LO) & (v <= config.TIB_HI)
        tv, vv = t[m2], v[m2]
        if len(tv) < 2:
            rows.append(r); continue
        r["tib"] = float(_trapz(np.clip(config.TIB_THR - vv, 0.0, None), tv) / 60.0)
        r["min_map_filt"] = float(vv.min()); r["mean_map_filt"] = float(vv.mean())
        rows.append(r)
    return pd.DataFrame(rows)

def _finalise(df, tib_df=None):
    if tib_df is not None and len(tib_df):
        df = df.merge(tib_df, on="caseid", how="left", validate="1:1", suffixes=("", "_tib"))
        for c in ("tib", "min_map_filt", "mean_map_filt"):
            if c + "_tib" in df.columns and df[c].isna().any():
                df[c] = df[c].fillna(df[c + "_tib"])
    df["log_tib_plus1"] = np.log(df["tib"] + 1.0)
    # right=False, so 30 falls in "30-60" and 60 falls in ">=60"
    df["tib_group"] = pd.cut(df["tib"], [-np.inf, 30.0, 60.0, np.inf],
                             labels=["<30", "30-60", ">=60"], right=False)
    return df

def _gate(df):
    E = config.EXPECTED
    n, ev = len(df), int(df["liver_injury"].sum())
    assert (n, ev) == (E["N"], E["events"]), f"cohort mismatch: N={n}, events={ev}"
    tib1 = float(df.loc[df.caseid == 1, "tib"].iloc[0])
    assert abs(tib1 - E["anchor_tib1"]) < config.TOL, f"anchor: {tib1}"
    assert abs(float(df["tib"].median()) - E["tib_median"]) < config.TOL
    for nm, got, exp in (("Q1", float(df["tib"].quantile(.25)), E["tib_q1"]),
                         ("Q3", float(df["tib"].quantile(.75)), E["tib_q3"])):
        assert abs(got - exp) < config.TOL, f"{nm}: {got}"
    got = {k: int(v) for k, v in df["tib_group"].value_counts().items()}
    assert got == E["tib_groups"], f"tib_group mismatch: {got}"
    return {"N": n, "events": ev, "anchor_tib1": tib1, "tib_median": float(df["tib"].median()),
            "tib_q1": float(df["tib"].quantile(.25)), "tib_q3": float(df["tib"].quantile(.75)),
            "tib_groups": got,
            "coverage_ok": float(df["coverage_ok"].mean()) if "coverage_ok" in df.columns else None}

def _find(cands):
    for d in config.SEARCH_DIRS:
        for f in cands:
            p = os.path.join(d, f)
            if os.path.exists(p):
                return p
    return None

def _read(p):
    return pd.read_excel(p) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p)

def load(force=False):
    if os.path.isfile(config.PKL) and not force:
        return pd.read_pickle(config.PKL), "cached"
    locked = _find(config.LOCKED_CANDS)
    if locked and config.MODE in ("auto", "locked"):
        df, source, path = _read(locked), "locked", locked
    else:
        if config.MODE == "locked":
            raise FileNotFoundError("locked dataset not found")
        tib_df = build_tib_from_cache()
        if tib_df.empty or tib_df["tib"].notna().sum() < 1400:
            raise FileNotFoundError("rebuild failed")
        base = _find(config.BASE_CANDS)
        df = _read(base).merge(tib_df, on="caseid", how="left",
                               validate="1:1", suffixes=("", "_tib"))
        source, path = "rebuilt", base
    df = _finalise(df)
    q = _gate(df); q.update(source=source, file=path)
    config.save("data_quality.json", q)
    df.to_pickle(config.PKL)
    return df, source