"""Sanity-check the collected data. Prints a short report. Read-only."""
import sqlite3
from common import DB_PATH, LOCATIONS

con = sqlite3.connect(DB_PATH)
q = lambda s, *a: con.execute(s, a).fetchall()

print("=== 1) Rows / date range / missing values per province ===")
print(f"{'province':<14}{'rows':>8}  {'from':<17}{'to':<17}{'pm2_5 null':>11}{'wind null':>10}")
for loc in LOCATIONS:
    n, a, b, nul = q("SELECT COUNT(*), MIN(time), MAX(time), SUM(pm2_5 IS NULL) FROM air_quality WHERE location=?", loc)[0]
    wnul = q("SELECT SUM(wind_speed_10m IS NULL) FROM weather WHERE location=?", loc)[0][0]
    print(f"{loc:<14}{n:>8}  {a:<17}{b:<17}{nul or 0:>11}{wnul or 0:>10}")

print("\n=== 2) Phetchabun PM2.5 by month of year (average over all years, ug/m3) ===")
rows = q("""SELECT substr(time,6,2) AS m, ROUND(AVG(pm2_5),1), ROUND(MAX(pm2_5),0)
            FROM air_quality WHERE location='phetchabun' AND pm2_5 IS NOT NULL GROUP BY m ORDER BY m""")
for m, avg, mx in rows:
    print(f"month {m}: avg {avg:>5}  max {mx:>5.0f}  " + "#" * int((avg or 0) / 2))

print("\n=== 3) Hotspots by month of year (all years, whole area) ===")
for m, n in q("SELECT substr(acq_date,6,2) m, COUNT(*) FROM hotspots GROUP BY m ORDER BY m"):
    print(f"month {m}: {n:>6}  " + "#" * (n // 1500))

print("\n=== 4) Hotspots per year ===")
for y, n in q("SELECT substr(acq_date,1,4), COUNT(*) FROM hotspots GROUP BY 1 ORDER BY 1"):
    print(y, n)

print("\n=== 5) Last 3 days of data (do recent rows have nulls?) ===")
for r in q("SELECT time, pm2_5, pm10 FROM air_quality WHERE location='phetchabun' ORDER BY time DESC LIMIT 3"):
    print("air", r)
for r in q("SELECT time, temperature_2m, wind_speed_10m, precipitation FROM weather WHERE location='phetchabun' ORDER BY time DESC LIMIT 3"):
    print("wx ", r)
