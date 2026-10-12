"""Streamlit dashboard: PM2.5 + fire hotspots for Phetchabun and neighbouring provinces.

Run:  streamlit run app.py
Needs data/features.csv, data/hotspots_app.csv, data/predictions.csv
(python build_features.py && python export_hotspots.py && python train_model.py)
"""
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from common import LOCATIONS

DATA = Path(__file__).parent / "data"
THAI = {"phetchabun": "เพชรบูรณ์", "phitsanulok": "พิษณุโลก", "loei": "เลย",
        "chaiyaphum": "ชัยภูมิ", "nakhon_sawan": "นครสวรรค์"}
MODEL_LABEL = {"persistence": "ใช้ค่าวันนี้ (baseline)", "climatology": "ค่าเฉลี่ยรายเดือน",
               "ridge": "Ridge", "hgb_no_fire": "Boosting (ไม่มี hotspot)", "hgb_all": "Boosting (มี hotspot)"}

# Colour bands roughly follow the Thai PM2.5 24h air-quality index (ug/m3).
BANDS = [(15, "#2f6fdf", "ดีมาก (≤15)"), (25, "#2e9e5b", "ดี (15-25)"), (37.5, "#e0b400", "ปานกลาง (25-37.5)"),
         (75, "#e8731a", "เริ่มมีผลต่อสุขภาพ (37.5-75)"), (1e9, "#d12f2f", "มีผลต่อสุขภาพ (>75)")]


def band(v):
    for hi, color, label in BANDS:
        if v <= hi:
            return color, label
    return BANDS[-1][1], BANDS[-1][2]


st.set_page_config(page_title="PM2.5 + Hotspot เพชรบูรณ์", layout="wide")


@st.cache_data
def load_features():
    return pd.read_csv(DATA / "features.csv", parse_dates=["date"])


@st.cache_data
def load_predictions():
    p = DATA / "predictions.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


@st.cache_data
def load_hotspots():
    """data/hotspots_app.csv (made by export_hotspots.py). FIRMS timestamps are UTC -> add local date (UTC+7)."""
    hs = pd.read_csv(DATA / "hotspots_app.csv", dtype={"acq_time": str})
    t = (pd.to_datetime(hs["acq_date"]) + pd.to_timedelta(hs["acq_time"].str.zfill(4).str[:2].astype(int), unit="h")
         + pd.to_timedelta(hs["acq_time"].str.zfill(4).str[2:].astype(int), unit="m") + pd.Timedelta(hours=7))
    hs["local_date"] = t.dt.strftime("%Y-%m-%d")
    return hs


def hotspots_on(day: str):
    """Hotspots on a LOCAL (UTC+7) date."""
    hs = load_hotspots()
    return hs.loc[hs["local_date"] == day, ["latitude", "longitude", "frp"]]


@st.cache_data
def next_day_forecast():
    """Next-day PM2.5 per province from the saved Ridge model, using the newest row in features.csv."""
    mp = DATA / "model_ridge.joblib"
    if not mp.exists():
        return None
    bundle = joblib.load(mp)
    df = load_features()
    df = pd.concat([df, pd.get_dummies(df["location"], prefix="prov", dtype=float)], axis=1)
    last = df.sort_values("date").groupby("location").tail(1).copy()
    last["forecast"] = bundle["model"].predict(last[bundle["features"]])
    return last[["location", "date", "pm25", "forecast"]]


def metrics_table(pred):
    y = pred["target"].values
    burn = pred["date"].dt.month.isin((1, 2, 3, 4)).values
    rows = []
    for k, label in MODEL_LABEL.items():
        if k not in pred:
            continue
        p = pred[k].values.astype(float)
        ok = ~np.isnan(p)

        def mae(m):
            m = m & ok
            return float(np.mean(np.abs(y[m] - p[m])))

        def rmse(m):
            m = m & ok
            return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
        rows.append({"โมเดล": label, "MAE ทั้งปี": mae(np.ones(len(y), bool)), "RMSE ทั้งปี": rmse(np.ones(len(y), bool)),
                     "MAE ช่วงเผา (ม.ค.-เม.ย.)": mae(burn), "RMSE ช่วงเผา": rmse(burn)})
    return pd.DataFrame(rows).set_index("โมเดล").round(2)


# ----------------------------------------------------------------------------- page
st.title("PM2.5 และจุดความร้อน: เพชรบูรณ์และจังหวัดรอบข้าง")
st.caption("ค่า PM2.5 เป็นผลโมเดล CAMS (ผ่าน Open-Meteo) ไม่ใช่ค่าวัดจากสถานีจริง ความละเอียดหยาบ "
           "ยอดฝุ่นช่วงเผาอาจต่ำกว่าสถานีวัดจริง | จุดความร้อนจาก NASA FIRMS (VIIRS)")

if not (DATA / "features.csv").exists() or not (DATA / "hotspots_app.csv").exists():
    st.error("ยังไม่พบข้อมูล: รัน collect_*.py, build_features.py และ export_hotspots.py ก่อน")
    st.stop()

feat = load_features()
pred = load_predictions()
dmin, dmax = feat["date"].min().date(), feat["date"].max().date()

with st.sidebar:
    st.header("ตัวเลือก")
    prov = st.selectbox("จังหวัด", list(LOCATIONS), format_func=lambda k: THAI.get(k, k))
    day = st.date_input("วันที่", value=dmax, min_value=dmin, max_value=dmax)
    st.caption(f"ข้อมูล: {dmin} ถึง {dmax}")

tab_map, tab_pred, tab_season = st.tabs(["แผนที่", "ผลการทำนาย", "ฤดูกาล"])

# ------------------------------------------------------------------------------ map
with tab_map:
    day_rows = feat[feat["date"] == pd.Timestamp(day)].set_index("location")
    hs = hotspots_on(str(day))
    lat0, lon0 = LOCATIONS[prov]

    m = folium.Map(location=[lat0, lon0], zoom_start=8, tiles="OpenStreetMap")
    folium.Circle([lat0, lon0], radius=100_000, color="#555", weight=1, fill=False, dash_array="6",
                  tooltip="รัศมี 100 กม.").add_to(m)
    for r in hs.itertuples():
        folium.CircleMarker([r.latitude, r.longitude], radius=2.5, color="#c0392b", weight=0, fill=True,
                            fill_opacity=0.65, tooltip=f"FRP {r.frp:.1f} MW").add_to(m)
    for loc, (la, lo) in LOCATIONS.items():
        if loc in day_rows.index:
            v = float(day_rows.loc[loc, "pm25"])
            color, label = band(v)
            folium.CircleMarker([la, lo], radius=14, color="#222", weight=2 if loc == prov else 1, fill=True,
                                fill_color=color, fill_opacity=0.9,
                                tooltip=f"{THAI.get(loc, loc)}: PM2.5 {v:.1f} µg/m³ ({label})").add_to(m)
    map_html = m.get_root().render()
    if hasattr(st, "iframe"):  # newer Streamlit; components.html is deprecated there
        st.iframe(map_html, height=560)
    else:
        components.html(map_html, height=560)

    c1, c2, c3 = st.columns(3)
    if prov in day_rows.index:
        v = float(day_rows.loc[prov, "pm25"])
        c1.metric(f"PM2.5 เฉลี่ยวันนั้น ({THAI[prov]})", f"{v:.1f} µg/m³", band(v)[1])
        c2.metric("hotspot ในรัศมี 100 กม.", int(day_rows.loc[prov, "fire_100km"]))
    c3.metric("hotspot ทั้งพื้นที่ในวันนั้น", len(hs))
    st.caption("วงกลมใหญ่ = ค่า PM2.5 เฉลี่ยรายวันของแต่ละจังหวัด (สีตามระดับ) | จุดแดง = hotspot | "
               "วงประ = รัศมี 100 กม. รอบจังหวัดที่เลือก | วันที่นับตามเวลาไทย")
    st.markdown(" ".join(f"<span style='color:{c}'>●</span> {l}" for _, c, l in BANDS), unsafe_allow_html=True)

    fc = next_day_forecast()
    if fc is not None:
        st.subheader("พยากรณ์ PM2.5 วันถัดไป (Ridge)")
        show = fc.assign(จังหวัด=fc["location"].map(THAI), วันที่ข้อมูลล่าสุด=fc["date"].dt.date,
                         ค่าล่าสุด=fc["pm25"].round(1), พยากรณ์วันถัดไป=fc["forecast"].round(1))
        st.dataframe(show[["จังหวัด", "วันที่ข้อมูลล่าสุด", "ค่าล่าสุด", "พยากรณ์วันถัดไป"]], hide_index=True)
        st.caption("คำนวณจากแถวข้อมูลล่าสุดในฐานข้อมูล ถ้าไม่ได้รัน collect_airweather.py --recent ก็จะเป็นของเมื่อวันที่ระบุ")

# ------------------------------------------------------------------------ predictions
with tab_pred:
    if pred is None:
        st.info("ยังไม่มี data/predictions.csv: รัน python train_model.py ก่อน")
    else:
        st.subheader("เปรียบเทียบโมเดล (ชุดทดสอบ)")
        st.dataframe(metrics_table(pred))
        st.caption(f"ชุดทดสอบ {pred['date'].min().date()} ถึง {pred['date'].max().date()} | เทรนด้วยข้อมูลก่อนหน้านั้นเท่านั้น "
                   "(แบ่งตามเวลา ไม่สุ่ม) | ค่าน้อย = ดี")
        sub = pred[pred["location"] == prov].set_index("date")
        chart = sub[["target", "persistence", "ridge", "hgb_all"]].rename(
            columns={"target": "ค่าจริง (วันถัดไป)", "persistence": MODEL_LABEL["persistence"],
                     "ridge": "Ridge", "hgb_all": MODEL_LABEL["hgb_all"]})
        st.subheader(f"ค่าจริงเทียบค่าทำนาย: {THAI[prov]}")
        st.line_chart(chart)

# ------------------------------------------------------------------------------ season
with tab_season:
    s = feat[feat["location"] == prov].copy()
    s["เดือน"] = s["date"].dt.month
    g = s.groupby("เดือน").agg(pm25=("pm25", "mean"), fire=("fire_100km", "mean"))
    st.subheader(f"PM2.5 เฉลี่ยรายเดือน (ทุกปีรวมกัน): {THAI[prov]}")
    st.bar_chart(g["pm25"].rename("PM2.5 เฉลี่ย (µg/m³)"))
    st.subheader("hotspot เฉลี่ยต่อวัน ในรัศมี 100 กม.")
    st.bar_chart(g["fire"].rename("hotspot ต่อวัน"))
    st.caption("ทั้งสองค่าขึ้นสูงพร้อมกันตามฤดู ซึ่งยังไม่พิสูจน์ว่า hotspot เป็นสาเหตุของฝุ่น "
               "(ดูการทดสอบ ablation.py ใน README)")
