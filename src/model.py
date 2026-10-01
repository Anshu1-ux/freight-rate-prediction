"""Hybrid freight-rate model.

Stage 1: robust linear model on log(rate/mile) capturing the secular price trend (extrapolates
         in time, which trees cannot) plus market_index / quote_signal effects.
Stage 2: LightGBM (L1 objective) on the stage-1 residual using route, equipment, weight, weekday
         features only -- no calendar-level features, so nothing has to be extrapolated.
Final prediction = exp(stage1 + stage2 + log distance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import HuberRegressor

from features import NUMERIC, CATEGORICAL

START = pd.Timestamp("2025-01-01")
LIN_COLS = ["t", "lmi", "quote_signal", "log_distance", "inv_dist", "eq_flatbed", "eq_reefer", "weight_frac_f"]
TREE_NUMERIC = [c for c in NUMERIC if c not in ("month", "doy", "dom")]
TREE_FEATURES = TREE_NUMERIC + CATEGORICAL

LGB_PARAMS = dict(objective="regression_l1", n_estimators=600, learning_rate=0.03, num_leaves=31,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, verbose=-1, random_state=0)


def _lin_frame(df: pd.DataFrame, weight_fill: float) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["t"] = (df["date"] - START).dt.days / 30.0          # months since 2025-01-01 (linear trend)
    x["lmi"] = np.log(df["market_index"])
    x["quote_signal"] = df["quote_signal"]
    x["log_distance"] = df["log_distance"]
    x["inv_dist"] = 100.0 / df["distance"]                 # short-haul premium
    x["eq_flatbed"] = (df["equipment"] == "Flatbed").astype(float)
    x["eq_reefer"] = (df["equipment"] == "Reefer").astype(float)
    x["weight_frac_f"] = df["weight_frac"].fillna(weight_fill)
    return x[LIN_COLS]


class HybridRateModel:
    def __init__(self, lgb_params: dict | None = None, outlier_bounds=(1.0, 5.0)):
        self.lgb_params = {**LGB_PARAMS, **(lgb_params or {})}
        self.outlier_bounds = outlier_bounds  # $/mile band treated as credible training labels

    def fit(self, df: pd.DataFrame, rate_col: str = "posted_rate"):
        rpm = df[rate_col] / df["distance"]
        keep = (rpm >= self.outlier_bounds[0]) & (rpm <= self.outlier_bounds[1])
        tr = df[keep]
        y = np.log(tr[rate_col]) - tr["log_distance"]
        self.weight_fill_ = float(tr["weight_frac"].median())
        X = _lin_frame(tr, self.weight_fill_)
        self.mu_, self.sd_ = X.mean(), X.std().replace(0, 1)
        self.lin_ = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500).fit((X - self.mu_) / self.sd_, y)
        resid = y - self.lin_.predict((X - self.mu_) / self.sd_)
        self.gbm_ = lgb.LGBMRegressor(**self.lgb_params).fit(tr[TREE_FEATURES], resid)
        self.n_train_, self.n_dropped_ = int(keep.sum()), int((~keep).sum())
        return self

    def predict_log_rpm(self, df: pd.DataFrame) -> np.ndarray:
        X = (_lin_frame(df, self.weight_fill_) - self.mu_) / self.sd_
        return self.lin_.predict(X) + self.gbm_.predict(df[TREE_FEATURES])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log_rpm(df) + df["log_distance"].values)
