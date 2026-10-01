"""Model selection under time-forward validation (mirrors the Nov-Dec hold-out)."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import Ridge
from features import prepare, FEATURES, NUMERIC

df = prepare(pd.read_csv("data/train_test.csv")).sort_values("date").reset_index(drop=True)
df["y"] = np.log(df["posted_rate"])
df["rpm"] = df["posted_rate"] / df["distance"]
df["outlier"] = ((df.rpm > 5) | (df.rpm < 1)).astype(int)

def metrics(y, p, mask=None):
    if mask is not None: y, p = y[mask], p[mask]
    e = p - y
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e**2).mean()), MAPE=(np.abs(e)/y).mean()*100,
                WAPE=np.abs(e).sum()/y.sum()*100, R2=1-(e**2).sum()/((y-y.mean())**2).sum())

# expanding-window folds, 2-month test blocks (like the 2-month Nov-Dec validation)
folds = [("<=Apr","May-Jun", df.date<"2025-05-01", (df.date>="2025-05-01")&(df.date<"2025-07-01")),
         ("<=Jun","Jul-Aug", df.date<"2025-07-01", (df.date>="2025-07-01")&(df.date<"2025-09-01")),
         ("<=Aug","Sep-Oct", df.date<"2025-09-01", df.date>="2025-09-01")]

base = dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)

def fit_lgb(tr, feats, objective="regression", **kw):
    p = {**base, **kw}
    m = lgb.LGBMRegressor(objective=objective, **p)
    m.fit(tr[feats], tr["y"]); return m

def run(name, fn):
    rows=[]
    for tn, vn, tm, vm in folds:
        tr, va = df[tm], df[vm]
        pred_rate = np.exp(fn(tr, va))
        r = metrics(va.posted_rate.values, pred_rate.values if hasattr(pred_rate,'values') else pred_rate)
        rc = metrics(va.posted_rate.values, np.asarray(pred_rate), (va.outlier==0).values)
        rows.append({**{k:v for k,v in r.items()}, **{k+"_clean":v for k,v in rc.items() if k in("MAE","MAPE")}})
    out = pd.DataFrame(rows).mean()
    print(f"{name:34s} MAE={out.MAE:7.1f} RMSE={out.RMSE:7.1f} MAPE={out.MAPE:5.2f} WAPE={out.WAPE:5.2f} R2={out.R2:.4f} | clean MAE={out.MAE_clean:7.1f} MAPE={out.MAPE_clean:5.2f}")
    return out

# baselines
def naive_rpm(tr, va):
    m = tr.groupby("equipment", observed=True).rpm.median()
    return np.log(va.equipment.map(m).astype(float) * va.distance)
def ridge(tr, va):
    X = lambda d: pd.get_dummies(d[["log_distance","market_index","quote_signal","weight_frac","equipment","is_weekend"]], columns=["equipment"]).astype(float).fillna(0.6)
    Xt, Xv = X(tr), X(va); m = Ridge(1.0).fit(Xt, tr.y); return m.predict(Xv[Xt.columns])

run("naive $/mile by equipment", naive_rpm)
run("ridge (log-linear)", ridge)
run("LGBM L2 log-target", lambda tr,va: fit_lgb(tr, FEATURES).predict(va[FEATURES]))
run("LGBM Huber log-target", lambda tr,va: fit_lgb(tr, FEATURES, "huber", alpha=0.3).predict(va[FEATURES]))
run("LGBM L1 log-target", lambda tr,va: fit_lgb(tr, FEATURES, "regression_l1").predict(va[FEATURES]))
# robust: drop outliers from training only
run("LGBM L2, outliers dropped (train)", lambda tr,va: fit_lgb(tr[tr.outlier==0], FEATURES).predict(va[FEATURES]))
run("LGBM Huber, outliers dropped", lambda tr,va: fit_lgb(tr[tr.outlier==0], FEATURES, "huber", alpha=0.3).predict(va[FEATURES]))
