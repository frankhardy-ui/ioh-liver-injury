"""config.py — locked constants, paths, resolved switches."""
import json, os

SEED, N_FOLD, BOOT, BOOT_MED = 42, 5, 1000, 500
MODE = "auto"

TIB_LO, TIB_HI, TIB_THR = 20.0, 250.0, 65.0
TIB_UNIT = "mm Hg·min"

EXPECTED = {"N": 1427, "events": 195, "anchor_tib1": 216.1833,
            "tib_median": 86.27, "tib_q1": 35.71, "tib_q3": 179.54,
            "tib_groups": {"<30": 301, "30-60": 238, ">=60": 888},
            "n_covariates": 19}
TOL = 0.01
REF_MODE = "median"
S2_MAP_VAR = "mean_map_filt"
PSM_EXPOSURE = "tib_ge_30"
PSM_MIN_PAIRS = 50
NA_LOW = 135.0
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("VB106_DATA") or os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
RESULT = os.path.join(OUT, "_step_results")
P0_CACHE = os.path.join(DATA_DIR, "tib_cache")
P0_DS = os.path.join(DATA_DIR, "dataset000.xlsx")
SEARCH_DIRS = [HERE, DATA_DIR, os.getcwd()]
PKL = os.path.join(OUT, "data.pkl")

LOCKED_CANDS = ["1427analysis_dataset.xlsx", "1427analysis_dataset.csv",
                "analysis_dataset.csv", "analysis_dataset.xlsx"]
BASE_CANDS = ["analysis_dataset_base.xlsx", "vitaldb_base.xlsx", "analysis_dataset.xlsx"]
for _d in (OUT, RESULT):
    os.makedirs(_d, exist_ok=True)

def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)