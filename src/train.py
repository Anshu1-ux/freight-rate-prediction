"""Fit on all labeled data, predict validation.csv and the fixed December chart inputs.

Run:  python src/train.py
"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from features import prepare, haversine_miles
from model import HybridRateModel

train = prepare(pd.read_csv("data/train_test.csv"))
val_raw = pd.read_csv("data/validation.csv")
val = prepare(val_raw)

model = HybridRateModel().fit(train)
print(f"Trained on {model.n_train_:,} rows ({model.n_dropped_} label outliers excluded from fitting).")

# ---- validation predictions ------------------------------------------------------------
val["predicted_rate"] = model.predict(val)
template = pd.read_csv("data/validation_predictions_template.csv")[["load_id"]]
out = template.merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
assert out.predicted_rate.notna().all() and (out.predicted_rate > 0).all() and len(out) == 12_000
out["predicted_rate"] = out["predicted_rate"].round(2)
out.to_csv("validation_predictions.csv", index=False)
print("Wrote validation_predictions.csv", out.shape, "| mean rate $%.0f" % out.predicted_rate.mean())

# ---- December chart --------------------------------------------------------------------
# The chart inputs only contain the date, but the model also needs the two daily market
# features. market_index is a *daily* market signal (std within a day ~0.025), so we take the
# December daily median from validation.csv's input features (no labels involved).
# quote_signal is load-level noise with no day structure, so we use its December median.
dec = pd.read_csv("data/december_chart_inputs.csv")
dec["date"] = pd.to_datetime(dec["date"])
coords = pd.concat([train, val])
pc = coords.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
dc = coords.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
daily_mi = val.groupby("date")["market_index"].median()
dec_q = float(val[val.date >= "2025-12-01"].quote_signal.median())

X = dec.copy()
X[["pickup_lat", "pickup_lon"]] = pc.loc[X.pickup].values
X[["delivery_lat", "delivery_lon"]] = dc.loc[X.delivery].values
X["market_index"] = X["date"].map(daily_mi)
assert X.market_index.notna().all(), "missing December market_index"
X["quote_signal"] = dec_q
X["weight_was_negative"] = 0; X["weight_capped"] = 0; X["weight_missing"] = 0; X["market_index_missing"] = 0
from features import add_features
X = add_features(X)
dec["predicted_rate"] = model.predict(X).round(2)
dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
dec.to_csv("december_predictions.csv", index=False)
print(dec[["date", "predicted_rate"]].describe().T if False else dec.predicted_rate.describe().round(1).to_dict())
