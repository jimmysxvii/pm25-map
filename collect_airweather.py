"""Collect hourly PM2.5 (CAMS model via Open-Meteo) + weather into SQLite.

Usage:
  python collect_airweather.py --start 2023-01-01          # backfill history
  python collect_airweather.py --recent                    # last few days (run hourly)
"""
import argparse
from datetime import date, timedelta

from common import LOCATIONS, TIMEZONE, get_db, http_get

AQ_ARCHIVE = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
WX_FORECAST = "https://api.open-meteo.com/v1/forecast"

AQ_VARS = ["pm2_5", "pm10"]
WX_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
           "wind_direction_10m", "precipitation"]


def save(con, table, loc, hourly, cols):
    times = hourly.get("time", [])
    rows = []
    for i, t in enumerate(times):
        rows.append((loc, t, *[hourly[c][i] if c in hourly else None for c in cols]))
    q = f"INSERT OR REPLACE INTO {table} (location, time, {', '.join(cols)}) VALUES (?, ?{', ?' * len(cols)})"
    con.executemany(q, rows)
    con.commit()
    return len(rows)


def chunks(start, end, days=180):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def backfill(con, start, end):
    for loc, (lat, lon) in LOCATIONS.items():
        for a, b in chunks(start, end):
            base = {"latitude": lat, "longitude": lon, "timezone": TIMEZONE,
                    "start_date": a.isoformat(), "end_date": b.isoformat()}
            aq = http_get(AQ_ARCHIVE, {**base, "hourly": ",".join(AQ_VARS)}).json()
            n1 = save(con, "air_quality", loc, aq["hourly"], AQ_VARS)
            wx = http_get(WX_ARCHIVE, {**base, "hourly": ",".join(WX_VARS)}).json()
            n2 = save(con, "weather", loc, wx["hourly"], WX_VARS)
            print(f"{loc} {a}..{b}: air={n1} rows, weather={n2} rows")


def recent(con, past_days=3):
    for loc, (lat, lon) in LOCATIONS.items():
        base = {"latitude": lat, "longitude": lon, "timezone": TIMEZONE,
                "past_days": past_days, "forecast_days": 1}
        aq = http_get(AQ_ARCHIVE, {**base, "hourly": ",".join(AQ_VARS)}).json()
        n1 = save(con, "air_quality", loc, aq["hourly"], AQ_VARS)
        wx = http_get(WX_FORECAST, {**base, "hourly": ",".join(WX_VARS)}).json()
        n2 = save(con, "weather", loc, wx["hourly"], WX_VARS)
        print(f"{loc} recent: air={n1}, weather={n2}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--recent", action="store_true")
    args = p.parse_args()
    con = get_db()
    if args.recent:
        recent(con)
    else:
        backfill(con, date.fromisoformat(args.start), date.fromisoformat(args.end))
    con.close()
