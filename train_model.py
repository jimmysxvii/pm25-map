"""Train + evaluate next-day PM2.5 models. Time-based split (no shuffling).

Compares:
  persistence   : tomorrow = today            (the baseline to beat)
  climatology   : average of that month in the training years
  ridge         : linear model
  hgb_no_fire   : gradient boosting WITHOUT hotspot features
  hgb_all       : gradient boosting WITH hotspot features
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEST_START = "2026-01-01"   # test period contains the whole 2026 burning season
BURN_MONTHS = (1, 2, 3, 4)

df = pd.read_csv("data/features.csv", parse_dates=["date"])
df = df.dropna(subset=["target"]).reset_index(drop=True)
df = pd.concat([df, pd.get_dummies(df["location"], prefix="prov", dtype=float)], axis=1)

train = df[df["date"] < TEST_START]
test = df[df["date"] >= TEST_START]
print(f"train: {len(train)} rows ({train['date'].min().date()}..{train['date'].max().date()})")
print(f"test : {len(test)} rows ({test['date'].min().date()}..{test['date'].max().date()})\n")
if len(train) < 500 or len(test) < 100:
    sys.exit("Not enough data for a split - check the date ranges above.")

prov = [c for c in df.columns if c.startswith("prov_")]
fire = [c for c in df.columns if c.startswith("fire_") or c == "frp_100km"]
base = ["pm25", "pm25_max", "pm10", "pm25_lag1", "pm25_lag2", "pm25_mean3", "pm25_mean7",
        "temp", "rh", "wind", "wind_u", "wind_v", "rain", "doy_sin", "doy_cos"] + prov
feats_all = base + fire
feats_nofire = base

def make_hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=0)

preds = {}
preds["persistence"] = test["pm25"].values
clim = train.assign(m=train["date"].dt.month).groupby(["location", "m"])["target"].mean()
preds["climatology"] = [clim.get((l, m), np.nan) for l, m in zip(test["location"], test["date"].dt.month)]

ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(train[feats_all], train["target"])
preds["ridge"] = ridge.predict(test[feats_all])
hgb_nf = make_hgb().fit(train[feats_nofire], train["target"])
preds["hgb_no_fire"] = hgb_nf.predict(test[feats_nofire])
hgb = make_hgb().fit(train[feats_all], train["target"])
preds["hgb_all"] = hgb.predict(test[feats_all])

def metrics(y, p):
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    return (mean_absolute_error(y[ok], p[ok]), np.sqrt(mean_squared_error(y[ok], p[ok])), r2_score(y[ok], p[ok]))

burn = test["date"].dt.month.isin(BURN_MONTHS).values
y = test["target"].values
print(f"{'model':<14}{'MAE':>8}{'RMSE':>8}{'R2':>8}   | burning season (Jan-Apr): {'MAE':>6}{'RMSE':>8}")
for name, p in preds.items():
    p = np.asarray(p, dtype=float)
    a = metrics(y, p)
    b = metrics(y[burn], p[burn])
    print(f"{name:<14}{a[0]:>8.2f}{a[1]:>8.2f}{a[2]:>8.2f}   |{'':>27}{b[0]:>6.2f}{b[1]:>8.2f}")

imp = permutation_importance(hgb, test[feats_all], y, n_repeats=5, random_state=0, scoring="neg_mean_absolute_error")
top = pd.Series(imp.importances_mean, index=feats_all).sort_values(ascending=False).head(10)
print("\nTop features (permutation importance, increase in MAE when shuffled):")
print(top.round(3).to_string())

out = test[["location", "date", "pm25", "target"]].copy()
for k, v in preds.items():
    out[k] = v
out.to_csv("data/predictions.csv", index=False)
joblib.dump({"model": hgb, "features": feats_all}, "data/model.joblib")
joblib.dump({"model": ridge, "features": feats_all}, "data/model_ridge.joblib")
print("\nsaved data/predictions.csv, data/model.joblib, data/model_ridge.joblib")
