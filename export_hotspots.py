"""Export only the hotspot columns the dashboard needs from data/pm25.db to a small CSV.

The Streamlit app reads data/hotspots_app.csv, so the 48 MB SQLite file never has to go on GitHub.
Usage:  python export_hotspots.py
"""
import sqlite3

import pandas as pd

from common import DB_PATH

OUT = DB_PATH.parent / "hotspots_app.csv"


def main():
    con = sqlite3.connect(DB_PATH)
    hs = pd.read_sql("SELECT latitude, longitude, acq_date, acq_time, frp FROM hotspots "
                     "ORDER BY acq_date, acq_time", con)
    con.close()
    hs["acq_time"] = hs["acq_time"].astype(str).str.zfill(4)
    hs["frp"] = hs["frp"].round(1)
    hs.to_csv(OUT, index=False)
    print(f"exported {len(hs)} hotspots ({hs['acq_date'].min()}..{hs['acq_date'].max()}) -> {OUT} "
          f"({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
