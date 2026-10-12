"""step5_aipw.py — augmented IPW, doubly robust (Step 5)."""
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import config
from data_loader import load
import covariates

MIN_P, CLIP_MU = 0.01, (0.01, 0.99)

def main():
    df, _ = load(); df = covariates.build(df)
    y = (df["liver_injury"] == 1).astype(int).to_numpy()
    tib = df["tib"].to_numpy(dtype=float)
    ok = np.isfinite(tib)
    y, tib = y[ok], tib[ok]
    X = StandardScaler().fit_transform(
        df.loc[ok, covariates.COVARS].astype(float).to_numpy())
    T_cont = np.log(tib + 1.0)
    T_cat = np.select([tib < 30, tib < 60], [0, 1], default=2).astype(int)
    kf = KFold(n_splits=config.N_FOLD, shuffle=True, random_state=config.SEED)
    cp = lambda p: np.clip(p, MIN_P, 1 - MIN_P)
    cm = lambda p: np.clip(p, *CLIP_MU)

    def est_cont(Xs, Ts, Ys):
        psi = np.zeros(len(Ys))
        for tr, va in kf.split(Xs):
            m = LinearRegression().fit(Xs[tr], Ts[tr])
            sig2 = np.mean((Ts[tr] - m.predict(Xs[tr])) ** 2)
            mu = LogisticRegression(max_iter=5000, C=1.0).fit(
                np.c_[Xs[tr], Ts[tr]], Ys[tr])
            Xv, Tv, Yv = Xs[va], Ts[va], Ys[va]
            mv = cm(mu.predict_proba(np.c_[Xv, Tv])[:, 1])
            psi[va] = ((Tv - m.predict(Xv)) * (Yv - mv) / sig2
                       + mv * (1 - mv) * mu.coef_[0][-1])
        return psi.mean()

    def est_cat(Xs, Ts, Ys, k):
        psi = np.zeros(len(Ys))
        for tr, va in kf.split(Xs):
            pi = LogisticRegression(max_iter=5000, C=1.0).fit(Xs[tr], Ts[tr])
            mu = LogisticRegression(max_iter=5000, C=1.0).fit(
                np.c_[Xs[tr], (Ts[tr] == 1), (Ts[tr] == 2)], Ys[tr])
            Xv, Tv, Yv = Xs[va], Ts[va], Ys[va]
            piv = cp(pi.predict_proba(Xv))
            def muX(kk):
                D = np.zeros((len(Xv), 2)); D[:, 0] = (kk == 1); D[:, 1] = (kk == 2)
                return cm(mu.predict_proba(np.c_[Xv, D])[:, 1])
            Ik, I0 = (Tv == k).astype(float), (Tv == 0).astype(float)
            psi[va] = (Ik * (Yv - muX(k)) / piv[:, k] + muX(k)
                       - I0 * (Yv - muX(0)) / piv[:, 0] - muX(0))
        return psi.mean()

    def run(label, fn, k=None):
        mean = fn(X, T_cont, y) if k is None else fn(X, T_cat, y, k)
        rng = np.random.default_rng(config.SEED)      # reset for each run, as locked
        draws = np.zeros(config.BOOT)
        for b in range(config.BOOT):
            idx = rng.integers(0, len(y), len(y))
            draws[b] = (fn(X[idx], T_cont[idx], y[idx]) if k is None
                        else fn(X[idx], T_cat[idx], y[idx], k))
            if (b + 1) % 200 == 0:
                print("  [bootstrap] %d/%d done" % (b + 1, config.BOOT))
        se = draws.std(ddof=1); lo, hi = np.percentile(draws, [2.5, 97.5])
        p = 2 * (1 - norm.cdf(abs(mean / se)))
        n_ext = int(np.sum(np.abs(draws) > 0.5))
        print("[%s] RD = %.3f pp (95%% CI %.3f to %.3f), P=%.3f"
              % (label, mean * 100, lo * 100, hi * 100, p))
        print("  [diagnostic] bootstrap draws beyond 50 pp: %d; range [%.3f, %.3f]"
              % (n_ext, draws.min(), draws.max()))
        return (round(mean * 100, 3), round(lo * 100, 3),
                round(hi * 100, 3), round(p, 3))

    out = {"AIPW_RD": run("continuous log(TIB+1)", est_cont),
           "AIPW_cat_30_60": run("30-60 vs <30", est_cat, 1),
           "AIPW_cat_ge60": run(">=60 vs <30", est_cat, 2)}
    import gates; gates.assert_step(5, out)
    return out

if __name__ == "__main__":
    main()