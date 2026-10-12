"""Does hotspot information actually help?  Ridge with vs without fire features.

Rolling-origin evaluation: for each test year, train only on earlier years.
Shows the result for the whole year and for the burning season (Jan-Apr), so one lucky split cannot decide the story.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/features.csv", parse_dates=["date"]).dropna(subset=["target"]).reset_index(drop=True)
df = pd.concat([df, pd.get_dummies(df["location"], prefix="prov", dtype=float)], axis=1)

prov = [c for c in df.columns if c.startswith("prov_")]
fire = [c for c in df.columns if c.startswith("fire_") or c == "frp_100km"]
base = ["pm25", "pm25_max", "pm10", "pm25_lag1", "pm25_lag2", "pm25_mean3", "pm25_mean7",
        "temp", "rh", "wind", "wind_u", "wind_v", "rain", "doy_sin", "doy_cos"] + prov


def fit(cols, train):
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(train[cols], train["target"])


rows = []
for year in (2024, 2025, 2026):
    train = df[df["date"] < f"{year}-01-01"]
    test = df[(df["date"] >= f"{year}-01-01") & (df["date"] < f"{year + 1}-01-01")]
    if len(train) < 500 or len(test) < 100:
        continue
    m_no, m_yes = fit(base, train), fit(base + fire, train)
    p_no, p_yes = m_no.predict(test[base]), m_yes.predict(test[base + fire])
    burn = test["date"].dt.month.isin((1, 2, 3, 4)).values
    for label, mask in (("whole year", np.ones(len(test), bool)), ("burn Jan-Apr", burn)):
        y = test["target"].values[mask]
        mae_base = mean_absolute_error(y, test["pm25"].values[mask])
        mae_no = mean_absolute_error(y, p_no[mask])
        mae_yes = mean_absolute_error(y, p_yes[mask])
        rows.append(dict(test_year=year, period=label, n=int(mask.sum()), train_rows=len(train),
                         persistence=mae_base, ridge_no_fire=mae_no, ridge_with_fire=mae_yes,
                         fire_gain_pct=(mae_no - mae_yes) / mae_no * 100))

res = pd.DataFrame(rows)
if res.empty:
    raise SystemExit("Not enough data - run build_features.py first.")
pd.set_option("display.width", 200)
print(res.round(2).to_string(index=False))
helped = int((res["fire_gain_pct"] > 0).sum())
print(f"\nhotspot features lowered MAE in {helped} of {len(res)} comparisons "
      f"(mean change {res['fire_gain_pct'].mean():+.1f}%, positive = hotspots help)")
print("Note: 2024 is trained on 2023 only, so treat it as the weakest row.")
