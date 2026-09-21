
import io
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


st.set_page_config(
    page_title="RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------
# UNIDADES
# ---------------------------------------------------------------------
UNIDADES = sorted([
    ("UTE Termocamaçari - UTE TCA", -12.666868, -38.314687),
    ("UTE Termobahia - UTE TBA", -12.703235, -38.564904),
    ("UTE Termoceará - UTE TCE", -3.692456, -38.870605),
    ("UTE Vale do Açu - UTE VLA", -5.381685, -36.819754),
    ("Refinaria Abreu e Lima - RNEST", -8.379661, -35.010198),
    ("Unidade de Tratamento de Gás Sul Capixaba - UTGSUL", -20.794464, -40.620910),
    ("Unidade de Tratamento de Gás de Cacimbas - UTGC", -19.463097, -39.760603),
    ("Refinaria Duque de Caxias - REDUC", -22.715100, -43.284008),
    ("UTE Termorio - UTE TRI", -22.714882, -43.254348),
    ("BOAVENTURA, Itaboraí-RJ", -22.660714, -42.853629),
    ("Unidade de Tratamento de Gás de Cabiúnas - UTGCAB", -22.285327, -41.717905),
    ("UTE Termomacaé - UTE TMA", -22.306164, -41.876702),
    ("UTE Seropédica/Baixada Fluminense - UTE SRP/BF", -22.723292, -43.647719),
    ("Refinaria Gabriel Passos - REGAP", -19.964281, -44.095138),
    ("UTE Ibirité - UTE IBT", -19.988579, -44.098207),
    ("UTE Juiz de Fora - UTE JF", -21.690619, -43.456720),
    ("UTE Três Lagoas - UTE TLG", -20.745500, -51.664590),
    ("Unidade de Tratamento de Gás de Caraguatatuba - UTGCA", -23.654191, -45.501209),
    ("Refinaria Presidente Bernardes - RPBC", -23.873326, -46.427572),
    ("UTE Cubatão - UTE CBT", -23.875735, -46.431387),
    ("Refinaria Henrique Lage - REVAP", -23.184796, -45.815806),
    ("Refinaria de Capuava - RECAP", -23.656681, -46.480883),
    ("Refinaria de Paulínia - REPLAN", -22.729589, -47.147712),
    ("UTE Nova Piratininga - UTE NPI", -23.699414, -46.673881),
    ("Refinaria Presidente Getúlio Vargas - REPAR", -25.566140, -49.369423),
    ("Refinaria Alberto Pasqualini - REFAP", -29.869901, -51.178195),
    ("UTE Canoas - UTE CAN", -29.875071, -51.145445),
    ("Armazém Rio de Janeiro", -22.810973, -43.281876),
    ("CILEP - CENPES", -22.854198, -43.233830),
    ("Porto Baia de Guanabara", -22.878942, -43.209330),
    ("ARM Macaé - Armazém Macaé", -22.415317, -41.861353),
    ("Porto de Imbetiba - Macaé", -22.386825, -41.768741),
    ("Porto Açu", -21.864737, -41.016444),
    ("Porto Aratu", -12.780132, -38.496764),
    ("Porto TMIB", -10.824129, -36.946301),
    ("Porto Belém", -1.439901, -48.494920),
    ("Porto Valença", -13.369369, -39.071252),
    ("Porto Guamaré", -5.106692, -36.319592),
    ("Porto Mucuripe", -3.713117, -38.474037),
    ("Porto Paracuru", -3.401146, -39.010896),
], key=lambda u: u[0].casefold())

REGION_HALFSPAN = 2.5
GRID_STEP = 0.5
IDW_POWER = 4.0
HORIZON_MAX = 6
TZ = "America/Sao_Paulo"

WEATHERAPI_FORECAST_URL = "https://api.weatherapi.com/v1/forecast.json"
API_KEY = "5af4a42f4f40420993975651262109"


def weatherapi_forecast(lat, lon):
    params = {
        "key": API_KEY,
        "q": f"{lat:.5f},{lon:.5f}",
        "days": 1,
        "aqi": "no",
        "alerts": "no",
        "lang": "pt",
    }
    resp = requests.get(WEATHERAPI_FORECAST_URL, params=params, timeout=18)
    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", {}).get("message", resp.text[:240])
        except Exception:
            msg = resp.text[:240]
        raise RuntimeError(f"HTTP {resp.status_code}: {msg}")
    return resp.json()


def parse_weatherapi(payload, fallback_lat, fallback_lon):
    rows = []
    for fd in payload.get("forecast", {}).get("forecastday", []):
        for h in fd.get("hour", []):
            c = h.get("condition") or {}
            rows.append({
                "lat": float(payload.get("location", {}).get("lat", fallback_lat)),
                "lon": float(payload.get("location", {}).get("lon", fallback_lon)),
                "tempo": pd.to_datetime(h.get("time"), errors="coerce"),
                "precipitation": float(h.get("precip_mm", 0) or 0),
                "pop": float(h.get("chance_of_rain", 0) or 0),
                "gust": float(h.get("gust_kph", 0) or 0),
                "rh": float(h.get("humidity", np.nan)),
                "clouds": float(h.get("cloud", np.nan)),
                "weather_code": int(c.get("code", 0) or 0),
                "weather_description": str(c.get("text", "")),
                "will_it_rain": int(h.get("will_it_rain", 0) or 0),
                "pressure": float(h.get("pressure_mb", np.nan)),
                "temp_c": float(h.get("temp_c", np.nan)),
                "dewpoint_c": float(h.get("dewpoint_c", np.nan)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # WeatherAPI returns local time for the requested coordinate.
    now_local = pd.Timestamp.now(tz=None).floor("h")
    df = df[df["tempo"] >= now_local].head(HORIZON_MAX + 1).reset_index(drop=True)
    return df


def make_sample_points(ulat, ulon):
    # 5 points: center + four cardinal points. The IDW fills the whole map.
    h = REGION_HALFSPAN
    return [
        (ulat, ulon),
        (ulat + h, ulon),
        (ulat - h, ulon),
        (ulat, ulon + h),
        (ulat, ulon - h),
    ]


@st.cache_data(ttl=1800, show_spinner=False, max_entries=32)
def load_region(ulat, ulon):
    points = make_sample_points(ulat, ulon)
    frames = []
    errors = []

    for lat, lon in points:
        try:
            data = weatherapi_forecast(lat, lon)
            frame = parse_weatherapi(data, lat, lon)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            errors.append(f"{lat:.3f},{lon:.3f}: {exc}")

        # Avoid burst limits on entry-level plans.
        time.sleep(0.45)

    if not frames:
        raise RuntimeError(
            "Nenhum ponto WeatherAPI foi retornado.\n" + "\n".join(errors)
        )
    result = pd.concat(frames, ignore_index=True)
    result = add_lightning_columns(result)
    return result, errors



def add_lightning_columns(df):
    """Adiciona raios_score e raios_classe ao dataframe sem depender de colunas opcionais."""
    df = df.copy()

    def num(name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(dtype=float)
        return np.full(len(df), default, dtype=float)

    p = np.clip(num("precipitation"), 0, None)
    pop = np.clip(num("pop"), 0, 100)
    gust = np.clip(num("gust"), 0, None)
    rh = np.clip(num("rh"), 0, 100)
    clouds = np.clip(num("clouds"), 0, 100)
    weather_code = num("weather_code")
    rain_flag = np.clip(num("will_it_rain"), 0, 1)
    temp = num("temp_c", np.nan)
    dew = num("dewpoint_c", np.nan)

    # WeatherAPI: 1087 = thundery outbreaks nearby;
    # 1273/1276 = rain with thunder; 1279/1282 = snow with thunder.
    thunder = np.isin(weather_code.astype(int), [1087, 1273, 1276, 1279, 1282]).astype(float)

    rain_signal = np.clip(p / 8.0, 0, 1)
    pop_signal = pop / 100.0
    gust_signal = np.clip((gust - 30.0) / 45.0, 0, 1)
    moisture_signal = np.clip((rh - 65.0) / 30.0, 0, 1)
    cloud_signal = np.clip((clouds - 65.0) / 35.0, 0, 1)

    dep = np.where(np.isfinite(temp) & np.isfinite(dew), np.maximum(temp - dew, 0), 10.0)
    dew_signal = np.clip(1.0 - dep / 10.0, 0, 1)

    score = 100.0 * (
        0.50 * thunder
        + 0.16 * rain_signal
        + 0.12 * pop_signal
        + 0.08 * rain_flag
        + 0.06 * gust_signal
        + 0.05 * moisture_signal
        + 0.02 * cloud_signal
        + 0.01 * dew_signal
    )
    # Presença explícita de trovoada nunca fica nas classes BAIXO/MUITO BAIXO.
    score = np.where(thunder > 0, np.maximum(score, 70.0), score)
    score = np.clip(score, 0, 100)

    df["raios_score"] = score.astype(float)
    df["raios_classe"] = np.select(
        [score < 20, score < 40, score < 60, score < 80],
        ["MUITO BAIXO", "BAIXO", "MODERADO", "ALTO"],
        default="MUITO ALTO"
    ).astype(str)
    return df


def idw(values_lat, values_lon, values, grid_lat, grid_lon, power=IDW_POWER):
    plat = np.asarray(values_lat, float)
    plon = np.asarray(values_lon, float)
    vals = np.asarray(values, float)
    valid = np.isfinite(vals)
    plat, plon, vals = plat[valid], plon[valid], vals[valid]
    if vals.size == 0:
        return np.full(grid_lat.shape, np.nan)
    lat0 = np.deg2rad(float(np.nanmean(plat)))
    x = np.deg2rad(plon) * np.cos(lat0) * 6371.0
    y = np.deg2rad(plat) * 6371.0
    gx = np.deg2rad(grid_lon.ravel()) * np.cos(lat0) * 6371.0
    gy = np.deg2rad(grid_lat.ravel()) * 6371.0
    dx = gx[:, None] - x[None, :]
    dy = gy[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    out = np.zeros(len(gx), dtype=float)
    for i in range(len(gx)):
        if np.any(d[i] < 1e-9):
            out[i] = vals[np.argmin(d[i])]
            continue
        w = 1.0 / np.maximum(d[i], 0.001) ** power
        out[i] = np.sum(w * vals) / np.sum(w)
    return out.reshape(grid_lat.shape)


# A WeatherAPI não é usada aqui para desenhar pontos de flashes.
# O índice é exclusivamente heurístico a partir da previsão horária.
def observed_lightning_score(lat, lon):
    return None, None


def map_figure(field, glat, glon, unit_name, ulat, ulon, when, title, label, vmax, cmap, lightning_mode=False):
    fig = plt.figure(figsize=(8.2, 5.0), dpi=135, facecolor="white")
    ax = plt.axes(projection=ccrs.PlateCarree())
    x0, x1 = ulon - REGION_HALFSPAN, ulon + REGION_HALFSPAN
    y0, y1 = ulat - REGION_HALFSPAN, ulat + REGION_HALFSPAN
    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

    if lightning_mode:
        from matplotlib.colors import ListedColormap, BoundaryNorm
        colors = ["#2E7D32", "#8BC34A", "#FDD835", "#FB8C00", "#D32F2F"]
        bounds = [0, 20, 40, 60, 80, 100]
        pm = ax.pcolormesh(
            glon, glat, field, cmap=ListedColormap(colors),
            norm=BoundaryNorm(bounds, 5), shading="nearest",
            transform=ccrs.PlateCarree(), rasterized=True
        )
        cb = fig.colorbar(pm, ax=ax, shrink=.78, pad=.02, ticks=[10,30,50,70,90])
        cb.ax.set_yticklabels(["MUITO BAIXO","BAIXO","MODERADO","ALTO","MUITO ALTO"], fontsize=7)
        cb.set_label("POTENCIAL DE RAIOS", fontsize=9)
    else:
        levels = np.linspace(0, max(vmax, 0.1), 18)
        pm = ax.pcolormesh(
            glon, glat, np.clip(field, 0, vmax),
            cmap=cmap, shading="nearest",
            transform=ccrs.PlateCarree(), rasterized=True
        )
        cb = fig.colorbar(pm, ax=ax, shrink=.78, pad=.02)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#F3F4F6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=.75, edgecolor="#222222", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=.6, edgecolor="#333333", zorder=3)
    try:
        ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=.32, edgecolor="#444444", zorder=3)
    except Exception:
        pass

    lon_ticks = np.arange(np.floor(x0), np.ceil(x1)+1, 1)
    lat_ticks = np.arange(np.floor(y0), np.ceil(y1)+1, 1)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.tick_params(labelsize=8)

    for xx in lon_ticks:
        ax.plot([xx, xx], [y0, y1], color="#888", alpha=.22, linewidth=.25, transform=ccrs.PlateCarree(), zorder=2)
    for yy in lat_ticks:
        ax.plot([x0, x1], [yy, yy], color="#888", alpha=.22, linewidth=.25, transform=ccrs.PlateCarree(), zorder=2)

    ax.scatter([ulon], [ulat], s=125, marker="^", facecolor="white", edgecolor="black", linewidth=1.8,
               transform=ccrs.PlateCarree(), zorder=6)
    ax.set_title(
        f"{unit_name}\n{pd.Timestamp(when):%d/%m/%Y %H:%M} local • {title}",
        fontsize=12, fontweight="bold", color="#181818", pad=8
    )
    fig.subplots_adjust(left=.055, right=.90, bottom=.095, top=.86)
    return fig


def region_grid(ulat, ulon):
    # Centers a 0.5° map over the fixed ±2.5° region.
    lats = np.arange(ulat - REGION_HALFSPAN, ulat + REGION_HALFSPAN + GRID_STEP * .51, GRID_STEP)
    lons = np.arange(ulon - REGION_HALFSPAN, ulon + REGION_HALFSPAN + GRID_STEP * .51, GRID_STEP)
    return np.meshgrid(lats, lons, indexing="ij")


# ---------------------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------------------
st.markdown("""
<style>
.stApp{background:#0d0f14}
section[data-testid="stSidebar"]{background:#11141b}
.block-container{max-width:1400px;padding-top:1.0rem}
h1,h2,h3,p,label,.stMarkdown,.stCaption{color:#f4f4f4!important}
div[data-testid="stMetric"]{background:#151922;border:1px solid #292f3a;border-radius:12px;padding:9px 12px}
.wb-card{background:#151922;border:1px solid #292f3a;border-radius:14px;padding:13px 16px;margin:8px 0}
.wb-green{color:#8BC34A}.wb-yellow{color:#FDD835}.wb-orange{color:#FB8C00}.wb-red{color:#EF4444}
</style>
""", unsafe_allow_html=True)

st.markdown("## ⚡ RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES")
st.caption("WEATHERAPI.COM • PREVISÃO HORÁRIA • 0 A +6 H • IDW 0,5° • POTENCIAL HEURÍSTICO DE RAIOS")

with st.sidebar:
    st.header("CONFIGURAÇÃO")
    q = st.text_input("BUSCAR UNIDADE", "")
    visible = [u for u in UNIDADES if q.strip().casefold() in u[0].casefold()]
    if not visible:
        st.error("Nenhuma unidade encontrada.")
        st.stop()
    names = [u[0] for u in visible]
    selected = st.selectbox("UNIDADE", names)
    unit = next(u for u in visible if u[0] == selected)
    ulat, ulon = unit[1], unit[2]

    horizon = st.slider("HORIZONTE", 1, 6, 6)
    st.markdown("**EXTENSÃO DA REGIÃO: FIXA EM ±2,5°**")
    st.markdown("**RESOLUÇÃO: FIXA EM 0,5°**")
    st.markdown("**IDW: POTÊNCIA FIXA EM 4,0**")

    atualizar = st.button("🔄 ATUALIZAR AGORA", use_container_width=True)
    if atualizar:
        st.cache_data.clear()
        st.rerun()


with st.spinner(f"CONSULTANDO WEATHERAPI.COM • 5 PONTOS DA REGIÃO DE {selected.upper()}..."):
    try:
        df, errors = load_region(float(ulat), float(ulon))
    except Exception as exc:
        st.error(f"ERRO AO CONSULTAR WEATHERAPI.COM: {type(exc).__name__}: {exc}")
        st.stop()

if errors:
    st.warning(f"{len(errors)} ponto(s) da malha não responderam. O campo será calculado com os pontos disponíveis.")

times = sorted(df["tempo"].dropna().unique())
times = times[:horizon + 1]
if not times:
    st.error("A WeatherAPI não retornou horários válidos.")
    st.stop()

# Índice do horário selecionado fica na sessão para os botões ◀ ▶.
if "idx_hora" not in st.session_state:
    st.session_state.idx_hora = 0
st.session_state.idx_hora = min(st.session_state.idx_hora, len(times)-1)

b1, b2, b3 = st.columns([1, 3, 1])
with b1:
    if st.button("◀ HORA ANTERIOR", use_container_width=True, disabled=st.session_state.idx_hora <= 0):
        st.session_state.idx_hora -= 1
        st.rerun()
with b2:
    st.markdown(
        f"<div style='text-align:center;font-size:1.1rem;font-weight:800;padding:8px'>"
        f"{pd.Timestamp(times[st.session_state.idx_hora]):%d/%m/%Y %H:%M} LOCAL"
        f"</div>", unsafe_allow_html=True
    )
with b3:
    if st.button("PRÓXIMA HORA ▶", use_container_width=True, disabled=st.session_state.idx_hora >= len(times)-1):
        st.session_state.idx_hora += 1
        st.rerun()

when = times[st.session_state.idx_hora]
fr = df[df["tempo"] == when].copy()

if "raios_score" not in df.columns or "raios_classe" not in df.columns:
    df = add_lightning_columns(df)
    fr = df[df["tempo"] == when].copy()

g_lat, g_lon = region_grid(float(ulat), float(ulon))
P = idw(fr.lat, fr.lon, fr.precipitation, g_lat, g_lon)
G = idw(fr.lat, fr.lon, fr.gust, g_lat, g_lon)
R = idw(fr.lat, fr.lon, fr.raios_score, g_lat, g_lon)


tabs = st.tabs(["🌧️ PRECIPITAÇÃO", "💨 RAJADA DE VENTO", "⚡ POTENCIAL DE RAIOS"])

with tabs[0]:
    vmax = max(5.0, float(np.nanpercentile(df.precipitation, 98)) if np.isfinite(df.precipitation).any() else 5.0)
    fig = map_figure(P, g_lat, g_lon, selected, ulat, ulon, when, "PRECIPITAÇÃO", "PRECIPITAÇÃO (MM/H)", vmax, "turbo")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with tabs[1]:
    vmax = max(60.0, float(np.nanpercentile(df.gust, 98)) if np.isfinite(df.gust).any() else 60.0)
    fig = map_figure(G, g_lat, g_lon, selected, ulat, ulon, when, "RAJADA DE VENTO", "RAJADA (KM/H)", vmax, "magma")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with tabs[2]:
    fig = map_figure(R, g_lat, g_lon, selected, ulat, ulon, when, "POTENCIAL DE RAIOS", "POTENCIAL", 100, "YlOrRd", lightning_mode=True)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# Condições na unidade mais próxima do ponto central.
distance = (fr["lat"] - ulat).abs() + (fr["lon"] - ulon).abs()
row = fr.loc[distance.idxmin()]
score = float(row["raios_score"])
classe = str(row["raios_classe"])


if score < 20:
    badge = "🟢 MUITO BAIXO"
elif score < 40:
    badge = "🟢 BAIXO"
elif score < 60:
    badge = "🟡 MODERADO"
elif score < 80:
    badge = "🟠 ALTO"
else:
    badge = "🔴 MUITO ALTO"

st.subheader("CONDIÇÕES NA UNIDADE")
c1,c2,c3,c4 = st.columns(4)
c1.metric("PRECIPITAÇÃO", f"{float(row.precipitation):.1f} MM/H")
c2.metric("RAJADA", f"{float(row.gust):.0f} KM/H")
c3.metric("PROB. CHUVA", f"{float(row.pop):.0f}%")
c4.metric("POTENCIAL DE RAIOS", f"{score:.0f}/100")

st.markdown(
    f'<div class="wb-card"><b>⚡ POTENCIAL DE RAIOS: {badge}</b><br>'
    f'<span style="color:#aeb4bf">O indicador combina previsão horária de tempestade/precipitação, probabilidade de chuva, umidade, cobertura de nuvens e rajadas. '
    f'O índice de raios é calculado somente a partir da previsão horária da WeatherAPI.</span></div>',
    unsafe_allow_html=True
)

st.subheader("PREVISÃO HORÁRIA NA UNIDADE")
out = fr.sort_values("tempo").copy()
out["POTENCIAL DE RAIOS"] = np.round(out["raios_score"], 0)
out = out[["tempo","precipitation","gust","pop","POTENCIAL DE RAIOS","raios_classe","weather_description"]]
out.columns = ["HORÁRIO","PRECIPITAÇÃO (MM/H)","RAJADA (KM/H)","PROB. CHUVA (%)","RAIOS (0–100)","CLASSE","CONDIÇÃO"]
out["HORÁRIO"] = pd.to_datetime(out["HORÁRIO"]).dt.strftime("%d/%m %H:%M")
st.dataframe(out, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ BAIXAR CSV DA PREVISÃO",
    out.to_csv(index=False).encode("utf-8-sig"),
    "previsao_weatherapi.csv",
    "text/csv"
)

st.caption("Fonte: WeatherAPI.com. O potencial de raios é um índice heurístico construído a partir da previsão horária; ele não representa detecção de descargas.")
