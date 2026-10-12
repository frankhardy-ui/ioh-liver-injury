"""gates.py — executable contract between code and manuscript."""
TOL_ABS, TOL_P = 5e-4, 5e-4

def close(a, b, tol=TOL_ABS):
    return abs(float(a) - float(b)) <= tol

CONTRACTS = {
    1: {"n_covariates": 19},
    2: {"N": 1427, "events": 195},
    3: {"N": 1427, "events": 195, "knots": [11.5, 86.3, 331.7], "ref": 86.27,
        "global_P": 0.613, "nonlinear_P": 0.493},
    4: {"M1": (1.166, 1.033, 1.317, 0.013, 1135.84, 0.548),
        "M2": (1.177, 1.040, 1.333, 0.010, 1140.80, 0.556),
        "M3": (1.178, 1.040, 1.334, 0.010, 1143.46, 0.558),
        "M4": (1.200, 1.055, 1.363, 0.005, 1136.82, 0.616),
        "M5": (0.936, 0.778, 1.126, 0.483, 1008.28, 0.771)},
    5: {"AIPW_RD": (-0.637, -2.705, 1.370, 0.549),
        "AIPW_cat_30_60": (-3.651, -14.069, 3.975, 0.434),
        "AIPW_cat_ge60": (-7.083, -16.996, 0.378, 0.106)},
    6: {"trees": 500, "max_depth": 15, "min_leaf": 10, "folds": 5,
        "top3": [34.2, 33.1, 11.0], "ATE": (0.187, -0.216, 0.590)},
    7: {"TE": (0.0136, -0.0262, 0.0460), "NDE": 0.0189, "NIE": -0.0053,
        "PM": -39.4, "N": 1427, "events": 195, "boot": 500},
    8: {"S1a": (0.923, 0.727, 1.172, 0.512, 1427, 300),
        "S1b": (0.959, 0.709, 1.297, 0.786, 1427, 171),
        "S1c": (0.832, 0.571, 1.214, 0.341, 1427, 86),
        "S1d": (0.819, 0.532, 1.259, 0.362, 1427, 39),
        "S2_low": (1.254, 0.851, 1.848, 0.252),
        "S2_mid": (0.743, 0.522, 1.055, 0.097),
        "S2_high": (1.110, 0.748, 1.649, 0.604),
        "S3": (0.611, 0.412, 0.906, 0.014, 1002, 73),
        "S4": (0.860, 0.643, 1.149, 0.307, 1302, 178),
        "S5": (0.602, 0.113, 3.201, 0.552, 1337, 4),
        "S6": (1.079, 0.863, 1.350, 0.505, 458, 63),
        "S7": {"E_point": None, "E_CI": 1.00},
        "S8": (0.889, 0.666, 1.187, 0.426, 1416, 190)},
    9: {"N": 1427, "events": 195, "tib_median": 86.27, "n_covariates": 19},
    10: {"EPV_events": 195, "n_covariates": 19},
}

def assert_step(step, results):
    spec = CONTRACTS[step]
    for k, exp in spec.items():
        if k not in results:
            raise AssertionError(f"step {step}: missing '{k}'")
        got = results[k]
        if isinstance(exp, dict) or isinstance(exp, bool):
            continue
        if isinstance(exp, (int, float)):
            if not close(got, exp):
                raise AssertionError(f"step {step}.{k}: {got} vs {exp}")
        elif isinstance(exp, (list, tuple)):
            for i, e in enumerate(exp):
                if e is None or i >= len(got) or got[i] is None:
                    continue
                tol = TOL_P if i == 3 else TOL_ABS
                if not close(got[i], e, tol):
                    raise AssertionError(f"step {step}.{k}[{i}]: {got[i]} vs {e}")
    return True