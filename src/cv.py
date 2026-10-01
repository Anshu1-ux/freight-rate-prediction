"""Validation study: time-forward CV, baselines, and an unseen-city stress test.

Run:  python src/cv.py
"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from features import prepare, FEATURES
from model import HybridRateModel, LGB_PARAMS


def metrics(y, p):
    e = p - y
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e**2).mean()), MAPE=(np.abs(e)/y).mean()*100,
                WAPE=np.abs(e).sum()/y.sum()*100, R2=1-(e**2).sum()/((y-y.mean())**2).sum())


def load():
    df = prepare(pd.read_csv("data/train_test.csv")).sort_values("date").reset_index(drop=True)
    rpm = df.posted_rate/df.distance
    df["outlier"] = ((rpm > 5) | (rpm < 1)).astype(int)   # only used to REPORT clean-row metrics
    return df


# expanding-window folds; 2-month test blocks mimic the 2-month Nov-Dec hold-out
FOLDS = [("Jan-Apr -> May-Jun", lambda d: d.date < "2025-05-01", lambda d: (d.date >= "2025-05-01") & (d.date < "2025-07-01")),
         ("Jan-Jun -> Jul-Aug", lambda d: d.date < "2025-07-01", lambda d: (d.date >= "2025-07-01") & (d.date < "2025-09-01")),
         ("Jan-Aug -> Sep-Oct", lambda d: d.date < "2025-09-01", lambda d: d.date >= "2025-09-01")]


class NaiveRPM:
    """Median $/mile by equipment x distance."""
    def fit(self, df):
        self.m = (df.posted_rate/df.distance).groupby(df.equipment, observed=True).median(); return self
    def predict(self, df): return df.equipment.map(self.m).astype(float).values * df.distance.values


class PlainLGBM:
    """Single LightGBM on log(rate), all raw features incl. calendar (no trend stage)."""
    def __init__(self, obj="regression_l1"): self.obj = obj
    def fit(self, df):
        rpm = df.posted_rate/df.distance; t = df[(rpm >= 1) & (rpm <= 5)]
        self.m = lgb.LGBMRegressor(**{**LGB_PARAMS, "objective": self.obj}).fit(t[FEATURES], np.log(t.posted_rate)); return self
    def predict(self, df): return np.exp(self.m.predict(df[FEATURES]))


def evaluate(df, make_model, label, verbose=True):
    rows = []
    for name, trf, vaf in FOLDS:
        tr, va = df[trf(df)], df[vaf(df)]
        p = make_model().fit(tr).predict(va)
        ok = (va.outlier == 0).values
        a, c = metrics(va.posted_rate.values, p), metrics(va.posted_rate.values[ok], p[ok])
        rows.append(dict(fold=name, **a, MAE_clean=c["MAE"], MAPE_clean=c["MAPE"],
                         bias_clean=np.log(va.posted_rate.values[ok]/p[ok]).mean()))
    out = pd.DataFrame(rows); m = out.drop(columns="fold").mean()
    if verbose:
        print(f"{label:34s} MAE={m.MAE:6.1f} RMSE={m.RMSE:6.1f} MAPE={m.MAPE:5.2f} WAPE={m.WAPE:5.2f} R2={m.R2:.4f} | clean MAE={m.MAE_clean:5.1f} MAPE={m.MAPE_clean:4.2f} bias={m.bias_clean:+.4f}")
    return out, m


def unseen_city_check(df, n_hold=8, seed=0, reps=5):
    """Hold out all loads touching 8 cities (like the 8 cities that only appear in validation)."""
    rng = np.random.default_rng(seed); cities = sorted(set(df.pickup) | set(df.delivery)); res = []
    cut = df.date < "2025-09-01"
    for r in range(reps):
        hold = set(rng.choice(cities, n_hold, replace=False))
        touches = df.pickup.isin(hold) | df.delivery.isin(hold)
        tr = df[cut & ~touches]; te_unseen = df[~cut & touches]; te_seen = df[~cut & ~touches]
        mdl = HybridRateModel().fit(tr)
        for nm, te in (("unseen-city rows", te_unseen), ("seen-city rows", te_seen)):
            ok = (te.outlier == 0).values; p = mdl.predict(te)
            res.append(dict(rep=r, group=nm, n=len(te), MAPE_clean=metrics(te.posted_rate.values[ok], p[ok])["MAPE"]))
    out = pd.DataFrame(res).groupby("group")[["n", "MAPE_clean"]].mean(); return out


if __name__ == "__main__":
    df = load()
    print("=== Time-forward CV (mean over 3 folds; 'clean' = rows with credible $/mile) ===")
    summary = {}
    for label, mk in [("Naive $/mile by equipment", NaiveRPM), ("Plain LightGBM (L1, log target)", PlainLGBM),
                      ("Hybrid trend + LightGBM residual", HybridRateModel)]:
        folds, m = evaluate(df, mk, label); summary[label] = m
    pd.DataFrame(summary).T.round(4).to_csv("cv_summary.csv")
    folds, _ = evaluate(df, HybridRateModel, "Hybrid (per fold)", verbose=False)
    folds.round(4).to_csv("cv_hybrid_folds.csv", index=False); print("\n", folds.round(4).to_string(index=False))
    print("\n=== Unseen-city stress test (train on Jan-Aug w/o 8 cities; test Sep-Oct) ===")
    u = unseen_city_check(df); print(u.round(3)); u.round(3).to_csv("cv_unseen_city.csv")
