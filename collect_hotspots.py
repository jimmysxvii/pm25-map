"""Collect NASA FIRMS VIIRS hotspots for the bounding box into SQLite.

Needs a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
Set it as an environment variable FIRMS_MAP_KEY (do NOT put it in the code or on GitHub).

Usage:
  python collect_hotspots.py --start 2024-01-01
"""
import argparse
import csv
import io
import os
from datetime import date, timedelta

from common import BBOX, get_db, http_get

URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{start}"
DAY_RANGE = 5  # keep requests small; the API caps the range per call
COLS = ["latitude", "longitude", "acq_date", "acq_time", "satellite",
        "bright_ti4", "confidence", "frp", "daynight"]


def fetch(key, source, start, days):
    bbox = ",".join(str(x) for x in BBOX)
    url = URL.format(key=key, source=source, bbox=bbox, days=days, start=start.isoformat())
    text = http_get(url).text.strip()
    if not text.startswith("latitude"):
        # FIRMS returns plain-text errors (bad key, rate limit) with HTTP 200
        raise RuntimeError(f"Unexpected FIRMS response: {text[:200]}")
    return list(csv.DictReader(io.StringIO(text)))


def save(con, rows):
    data = [tuple(r.get(c) for c in COLS) for r in rows]
    con.executemany(
        f"INSERT OR IGNORE INTO hotspots ({', '.join(COLS)}) VALUES ({', '.join('?' * len(COLS))})",
        data)
    con.commit()
    return len(data)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    args = p.parse_args()

    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise SystemExit("Set FIRMS_MAP_KEY first (see top of this file).")

    con = get_db()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    recent_cutoff = date.today() - timedelta(days=60)
    total = 0
    cur = start
    while cur <= end:
        days = min(DAY_RANGE, (end - cur).days + 1)
        # Standard-processing archive for older data, near-real-time for the last ~60 days
        source = "VIIRS_SNPP_SP" if cur < recent_cutoff else "VIIRS_SNPP_NRT"
        n = save(con, fetch(key, source, cur, days))
        total += n
        print(f"{cur} +{days}d ({source}): {n} hotspots")
        cur += timedelta(days=days)
    print("done, rows seen:", total)
    con.close()
