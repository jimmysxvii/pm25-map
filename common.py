"""Shared config + database helpers for the PM2.5 + hotspot project."""
import sqlite3
import time
from pathlib import Path

import requests

DB_PATH = Path(__file__).parent / "data" / "pm25.db"

# name: (latitude, longitude)  -- edit freely
LOCATIONS = {
    "phetchabun": (16.42, 101.15),
    "phitsanulok": (16.82, 100.26),
    "loei": (17.49, 101.73),
    "chaiyaphum": (15.81, 102.03),
    "nakhon_sawan": (15.70, 100.14),
}

# Bounding box for hotspot search: west, south, east, north
BBOX = (99.5, 14.5, 102.8, 17.9)

TIMEZONE = "Asia/Bangkok"

SCHEMA = """
CREATE TABLE IF NOT EXISTS air_quality (
    location TEXT NOT NULL,
    time TEXT NOT NULL,          -- local time (Asia/Bangkok), ISO format
    pm2_5 REAL,
    pm10 REAL,
    PRIMARY KEY (location, time)
);
CREATE TABLE IF NOT EXISTS weather (
    location TEXT NOT NULL,
    time TEXT NOT NULL,
    temperature_2m REAL,
    relative_humidity_2m REAL,
    wind_speed_10m REAL,
    wind_direction_10m REAL,
    precipitation REAL,
    PRIMARY KEY (location, time)
);
CREATE TABLE IF NOT EXISTS hotspots (
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    acq_date TEXT NOT NULL,      -- UTC date from FIRMS
    acq_time TEXT NOT NULL,      -- UTC HHMM
    satellite TEXT NOT NULL,
    bright_ti4 REAL,
    confidence TEXT,
    frp REAL,
    daynight TEXT,
    PRIMARY KEY (latitude, longitude, acq_date, acq_time, satellite)
);
"""


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def http_get(url, params=None, retries=4, timeout=60):
    """GET with simple retry/backoff. Raises on final failure."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code != 200:
                # client error (bad params / bad key): retrying will not help
                raise ValueError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r
        except ValueError:
            raise
        except Exception as e:  # network error, 429, 5xx
            last = e
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"Failed after {retries} tries: {last}")
