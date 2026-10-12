"""Turn the raw SQLite tables into one daily feature table (data/features.csv).

One row = one province on one day (local time, UTC+7).
Features only use information available by the END of that day.
Target = the province's mean PM2.5 on the NEXT day.
"""
import numpy as np
import pandas as pd

from common import DB_PATH, LOCATIONS
import sqlite3

RADII_KM = (50, 100, 200)
UTC_OFFSET_H = 7


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load():
    con = sqlite3.connect(DB_PATH)
    aq = pd.read_sql("SELECT * FROM air_quality", con, parse_dates=["time"])
    wx = pd.read_sql("SELECT * FROM weather", con, parse_dates=["time"])
    hs = pd.read_sql("SELECT latitude, longitude, acq_date, acq_time, frp FROM hotspots", con)
    con.close()
    return aq, wx, hs


def daily_air_weather(aq, wx):
    aq["date"] = aq["time"].dt.normalize()
    wx["date"] = wx["time"].dt.normalize()
    a = aq.groupby(["location", "date"]).agg(pm25=("pm2_5", "mean"), pm25_max=("pm2_5", "max"),
                                             pm10=("pm10", "mean")).reset_index()
    # wind as vector components (direction is "coming from", in degrees)
    rad = np.radians(wx["wind_direction_10m"])
    wx["wind_u"] = -wx["wind_speed_10m"] * np.sin(rad)
    wx["wind_v"] = -wx["wind_speed_10m"] * np.cos(rad)
    w = wx.groupby(["location", "date"]).agg(
        temp=("temperature_2m", "mean"), rh=("relative_humidity_2m", "mean"),
        wind=("wind_speed_10m", "mean"), wind_u=("wind_u", "mean"), wind_v=("wind_v", "mean"),
        rain=("precipitation", "sum")).reset_index()
    return a.merge(w, on=["location", "date"], how="inner")


def daily_hotspots(hs):
    """Count hotspots near each province per LOCAL day (FIRMS gives UTC)."""
    t = pd.to_datetime(hs["acq_date"]) + pd.to_timedelta(
        hs["acq_time"].astype(str).str.zfill(4).str[:2].astype(int), unit="h") \
        + pd.to_timedelta(hs["acq_time"].astype(str).str.zfill(4).str[2:].astype(int), unit="m")
    hs = hs.assign(date=(t + pd.Timedelta(hours=UTC_OFFSET_H)).dt.normalize())
    out = []
    for loc, (lat, lon) in LOCATIONS.items():
        d = haversine_km(lat, lon, hs["latitude"].values, hs["longitude"].values)
        frames = []
        for r in RADII_KM:
            m = d <= r
            g = hs[m].groupby("date").size().rename(f"fire_{r}km")
            frames.append(g)
        frp = hs[d <= 100].groupby("date")["frp"].sum().rename("frp_100km")
        df = pd.concat(frames + [frp], axis=1, sort=False).reset_index()
        df["location"] = loc
        out.append(df)
    return pd.concat(out, ignore_index=True), hs["date"].min(), hs["date"].max()


def main():
    aq, wx, hs = load()
    day = daily_air_weather(aq, wx)
    fires, f_min, f_max = daily_hotspots(hs)
    df = day.merge(fires, on=["location", "date"], how="left")

    # days with no hotspot rows really mean zero fires, but only inside the hotspot coverage window
    fire_cols = [c for c in df.columns if c.startswith("fire_") or c == "frp_100km"]
    df = df[(df["date"] >= f_min) & (df["date"] <= f_max)].copy()
    df[fire_cols] = df[fire_cols].fillna(0)
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    g = df.groupby("location")
    df["pm25_lag1"] = g["pm25"].shift(1)
    df["pm25_lag2"] = g["pm25"].shift(2)
    df["pm25_mean3"] = g["pm25"].transform(lambda s: s.rolling(3).mean())
    df["pm25_mean7"] = g["pm25"].transform(lambda s: s.rolling(7).mean())
    for r in RADII_KM:
        c = f"fire_{r}km"
        df[f"{c}_sum3"] = g[c].transform(lambda s: s.rolling(3).sum())
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["target"] = g["pm25"].shift(-1)  # tomorrow's mean PM2.5

    df = df.dropna(subset=["pm25_mean7"]).reset_index(drop=True)
    df.to_csv("data/features.csv", index=False)
    print(f"features: {len(df)} rows, {df['date'].min().date()} .. {df['date'].max().date()}")
    print("rows without target (last day):", int(df['target'].isna().sum()))
    print("columns:", [c for c in df.columns if c not in ('location', 'date')])


if __name__ == "__main__":
    main()
