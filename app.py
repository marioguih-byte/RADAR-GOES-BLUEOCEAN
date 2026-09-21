import re
import unicodedata
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


SITE_TITLE = "RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES"
GRID_STEP = 0.5
IDW_POWER = 4.0
MAX_HOURS = 6
TZ = "America/Sao_Paulo"
CACHE_TTL = 1800
REGION_HALFSPAN = 2.5
SAMPLE_POINTS_PER_SIDE = 3

st.set_page_config(page_title=SITE_TITLE, page_icon="⚡", layout="wide")

UNIDADES = [
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
]

UNIDADES = sorted(
    UNIDADES,
    key=lambda u: unicodedata.normalize("NFKD", u[0]).encode("ascii", "ignore").decode().casefold(),
)


def normalizar_0_1(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((np.asarray(x, dtype=float) - low) / (high - low), 0, 1)


def idw_grid(points_lat, points_lon, values, grid_lat, grid_lon, power=IDW_POWER, k=12):
    la = np.asarray(points_lat, dtype=float)
    lo = np.asarray(points_lon, dtype=float)
    va = np.asarray(values, dtype=float)
    good = np.isfinite(la) & np.isfinite(lo) & np.isfinite(va)
    la, lo, va = la[good], lo[good], va[good]
    if va.size == 0:
        return np.full_like(grid_lat, np.nan, dtype=float)

    lat_ref = np.deg2rad(float(np.nanmean(la)))
    xp = np.deg2rad(lo) * np.cos(lat_ref) * 6371.0
    yp = np.deg2rad(la) * 6371.0
    xg = np.deg2rad(np.asarray(grid_lon).ravel()) * np.cos(lat_ref) * 6371.0
    yg = np.deg2rad(np.asarray(grid_lat).ravel()) * 6371.0

    d = np.hypot(xg[:, None] - xp[None, :], yg[:, None] - yp[None, :])
    n = min(int(k), d.shape[1])
    idx = np.argpartition(d, n - 1, axis=1)[:, :n]
    dd = np.take_along_axis(d, idx, axis=1)
    vv = va[idx]
    exact = dd < 1e-9
    with np.errstate(divide='ignore', invalid='ignore'):
        w = 1.0 / np.maximum(dd, 0.001) ** power
        z = np.sum(w * vv, axis=1) / np.sum(w, axis=1)
    rows = np.where(exact.any(axis=1))[0]
    if rows.size:
        cols = np.argmax(exact[rows], axis=1)
        z[rows] = vv[rows, cols]
    return z.reshape(np.asarray(grid_lat).shape)

def make_plot_grid(lat0, lon0):
    """Pontos realmente consultados no ICON: grade 3x3 cobrindo toda a região."""
    offsets = np.linspace(-REGION_HALFSPAN, REGION_HALFSPAN, SAMPLE_POINTS_PER_SIDE)
    lats, lons = np.meshgrid(lat0 + offsets, lon0 + offsets, indexing="ij")
    return lats, lons


def make_map_grid(lat0, lon0):
    """Grade final visual fixa de 0,5°, usada somente para o IDW."""
    lats = np.arange(lat0 - REGION_HALFSPAN, lat0 + REGION_HALFSPAN + GRID_STEP * 0.51, GRID_STEP)
    lons = np.arange(lon0 - REGION_HALFSPAN, lon0 + REGION_HALFSPAN + GRID_STEP * 0.51, GRID_STEP)
    return np.meshgrid(lats, lons, indexing="ij")


def parse_openmeteo(raw, lats, lons, variable_names):
    items = raw if isinstance(raw, list) else [raw]
    if len(items) != len(lats):
        if len(lats) > 1:
            raise RuntimeError(
                f"Resposta ICON incompatível: esperados {len(lats)} locais, recebidos {len(items)}."
            )
    rows = []
    for i, item in enumerate(items):
        h = item.get("hourly", {})
        times = h.get("time", [])
        for j, t in enumerate(times):
            row = {
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "tempo": pd.Timestamp(t),
            }
            for var in variable_names:
                vals = h.get(var, [])
                row[var] = vals[j] if j < len(vals) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False, max_entries=32)
def consultar_previsao_icon_regiao(ulat, ulon):
    """Consulta ICON somente para 9 pontos da região selecionada.

    Não consulta as 40 unidades. Se o endpoint específico limitar a origem,
    tenta o endpoint genérico explicitando models=icon_global e, por último,
    uma consulta pontual para manter o aplicativo funcional.
    """
    lat_grid, lon_grid = make_plot_grid(float(ulat), float(ulon))
    lats = lat_grid.ravel().tolist()
    lons = lon_grid.ravel().tolist()

    variables = [
        "precipitation", "showers", "wind_gusts_10m", "cape",
        "relative_humidity_2m", "cloud_cover", "weather_code",
        "lightning_potential",
    ]
    base_params = {
        "latitude": ",".join(f"{x:.5f}" for x in lats),
        "longitude": ",".join(f"{x:.5f}" for x in lons),
        "hourly": ",".join(variables),
        "forecast_hours": MAX_HOURS + 1,
        "timezone": TZ,
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
        "cell_selection": "nearest",
    }
    headers = {"User-Agent": "rio-ultra-power-previsoes/5.0"}

    # 1) Endpoint próprio do DWD ICON.
    attempts = [
        ("https://api.open-meteo.com/v1/dwd-icon", {}),
        ("https://api.open-meteo.com/v1/forecast", {"models": "icon_global"}),
    ]
    ultimo_429 = None
    for url, extra in attempts:
        try:
            params = dict(base_params)
            params.update(extra)
            r = requests.get(url, params=params, headers=headers, timeout=22)
            if r.status_code == 429:
                ultimo_429 = r
                continue
            r.raise_for_status()
            df = parse_openmeteo(r.json(), lats, lons, variables)
            if not df.empty:
                return df
        except requests.HTTPError as exc:
            # Só trata 429 como fallback; erros 4xx/5xx reais continuam visíveis.
            if getattr(exc.response, "status_code", None) == 429:
                ultimo_429 = exc.response
                continue
            raise
        except requests.RequestException:
            continue

    # 2) Fallback de emergência: uma única localidade. Isso evita que o site
    # caia completamente em uma origem que esteja temporariamente limitada.
    try:
        single_params = dict(base_params)
        single_params["latitude"] = f"{float(ulat):.5f}"
        single_params["longitude"] = f"{float(ulon):.5f}"
        for url, extra in attempts:
            params = dict(single_params)
            params.update(extra)
            r = requests.get(url, params=params, headers=headers, timeout=22)
            if r.status_code == 429:
                continue
            r.raise_for_status()
            one = parse_openmeteo(r.json(), [float(ulat)], [float(ulon)], variables)
            if not one.empty:
                # Expande o ponto central para uma nuvem 3x3 apenas como
                # modo de contingência; a execução normal continua usando 9 pontos.
                parts = []
                for la, lo in zip(lats, lons):
                    q = one.copy()
                    q["lat"] = la
                    q["lon"] = lo
                    parts.append(q)
                return pd.concat(parts, ignore_index=True)
    except requests.RequestException:
        pass

    retry = ultimo_429.headers.get("Retry-After") if ultimo_429 is not None else None
    detalhe = f" Retry-After={retry}s." if retry else ""
    raise RuntimeError(
        "O Open-Meteo está limitando temporariamente a origem (HTTP 429)."
        + detalhe +
        " A aplicação reduziu a consulta para 9 pontos e tentou o ICON Global e o modo pontual de emergência."
    )

def calcular_indice_holistico(df):
    """Índice heurístico 0-100 baseado exclusivamente no DWD ICON via Open-Meteo."""
    p = np.clip(df['precipitation'].fillna(0).to_numpy(float), 0, None)
    showers = np.clip(df['showers'].fillna(0).to_numpy(float), 0, None)
    cape = np.clip(df['cape'].fillna(0).to_numpy(float), 0, None)
    rh = np.clip(df['relative_humidity_2m'].fillna(0).to_numpy(float), 0, 100)
    cloud = np.clip(df['cloud_cover'].fillna(0).to_numpy(float), 0, 100)
    lpi = np.clip(df['lightning_potential'].fillna(0).to_numpy(float), 0, None)
    wc = df['weather_code'].fillna(-1).to_numpy(float).astype(int)

    def robust_norm(x):
        finite = x[np.isfinite(x)]
        if finite.size < 3:
            hi = max(float(np.nanmax(x)) if finite.size else 1.0, 1.0)
            return np.clip(x / hi, 0, 1)
        lo = float(np.nanpercentile(finite, 50))
        hi = float(np.nanpercentile(finite, 95))
        if hi <= lo + 1e-9:
            hi = lo + 1.0
        return np.clip((x - lo) / (hi - lo), 0, 1)

    s_lpi = robust_norm(lpi)
    s_cape = robust_norm(cape)
    s_showers = robust_norm(showers)
    s_rain = robust_norm(p)
    s_rh = np.clip((rh - 65.0) / 35.0, 0, 1)
    s_cloud = np.clip((cloud - 55.0) / 45.0, 0, 1)

    storm = np.zeros(len(df), dtype=float)
    storm[np.isin(wc, [95, 96, 99])] = 1.0
    storm[np.isin(wc, [80, 81, 82, 83, 84])] = 0.45

    score = (
        0.45 * s_lpi +
        0.22 * storm +
        0.12 * s_showers +
        0.08 * s_cape +
        0.06 * s_rain +
        0.04 * s_rh +
        0.03 * s_cloud
    ) * 100.0
    score += ((showers >= 1.0) & (cape >= 500)).astype(float) * 10.0
    score += ((lpi > 0) & np.isin(wc, [95, 96, 99])).astype(float) * 8.0
    score = np.clip(score, 0, 100)

    classes = np.select(
        [score < 20, score < 40, score < 60, score < 80],
        ['MUITO BAIXO', 'BAIXO', 'MODERADO', 'ALTO'],
        default='MUITO ALTO',
    )
    return df.assign(raios_score=score, raios_classe=classes)

def mapa_cartopy(GLA, GLO, field, unit, ulat, ulon, when, title, label, vmax, cmap, mode):
    fig = plt.figure(figsize=(7.7, 4.85), dpi=140, facecolor="white")
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor("white")

    # Centros em 0,5° -> bordas explícitas, evitando qualquer faixa vazia.
    d = GRID_STEP / 2.0
    lat_centers = GLA[:, 0]
    lon_centers = GLO[0, :]
    lat_edges = np.r_[lat_centers - d, lat_centers[-1] + d]
    lon_edges = np.r_[lon_centers - d, lon_centers[-1] + d]
    x0, x1 = float(lon_edges[0]), float(lon_edges[-1])
    y0, y1 = float(lat_edges[0]), float(lat_edges[-1])
    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")

    if mode == "raios":
        colors = ["#2E7D32", "#8BC34A", "#FDD835", "#FB8C00", "#D32F2F"]
        bounds = np.array([0, 20, 40, 60, 80, 100], float)
        norm = BoundaryNorm(bounds, len(colors))
        pm = ax.pcolormesh(
            lon_edges, lat_edges, np.clip(field, 0, 100),
            cmap=ListedColormap(colors), norm=norm, shading="flat",
            transform=ccrs.PlateCarree(), edgecolors="none", linewidth=0,
            antialiased=False, rasterized=True, zorder=1,
        )
        cb = fig.colorbar(pm, ax=ax, pad=0.018, shrink=0.80, ticks=[10, 30, 50, 70, 90])
        cb.ax.set_yticklabels(["MUITO BAIXO", "BAIXO", "MODERADO", "ALTO", "MUITO ALTO"], fontsize=7.0)
        cb.set_label("POTENCIAL HOLÍSTICO DE RAIOS", fontsize=9)
    else:
        pm = ax.pcolormesh(
            lon_edges, lat_edges, np.clip(field, 0, vmax),
            cmap=cmap, shading="flat", transform=ccrs.PlateCarree(),
            edgecolors="none", linewidth=0, antialiased=False,
            rasterized=True, zorder=1,
        )
        cb = fig.colorbar(pm, ax=ax, pad=0.018, shrink=0.80)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#F1F2F4", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#222", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.65, edgecolor="#333", zorder=3)
    try:
        ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=0.38, edgecolor="#444", zorder=3)
    except Exception:
        pass

    lon_ticks = np.arange(np.floor(x0), np.ceil(x1) + 1, 1)
    lat_ticks = np.arange(np.floor(y0), np.ceil(y1) + 1, 1)
    if len(lon_ticks) > 8:
        lon_ticks = np.linspace(x0, x1, 7)
    if len(lat_ticks) > 8:
        lat_ticks = np.linspace(y0, y1, 7)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.tick_params(axis="both", labelsize=8, colors="#333", width=0.6)

    # Grade sutil, por cima do preenchimento e sem criar linhas entre pixels.
    for x in lon_ticks:
        ax.plot([x, x], [y0, y1], color="#666", linewidth=0.24, alpha=0.20,
                transform=ccrs.PlateCarree(), zorder=2)
    for y in lat_ticks:
        ax.plot([x0, x1], [y, y], color="#666", linewidth=0.24, alpha=0.20,
                transform=ccrs.PlateCarree(), zorder=2)

    ax.scatter([ulon], [ulat], marker="^", s=120, facecolor="white", edgecolor="black",
               linewidth=1.8, transform=ccrs.PlateCarree(), zorder=6)
    ax.scatter([ulon], [ulat], marker="o", s=13, facecolor="black", edgecolor="white",
               linewidth=0.5, transform=ccrs.PlateCarree(), zorder=7)

    ax.set_title(
        f"{unit}\n{pd.Timestamp(when):%d/%m/%Y %H:%M} LOCAL • {title}",
        fontsize=11.5, fontweight="bold", color="#171717", pad=7,
    )
    try:
        ax.spines["geo"].set_edgecolor("#222")
        ax.spines["geo"].set_linewidth(0.8)
    except Exception:
        pass
    fig.subplots_adjust(left=0.035, right=0.915, bottom=0.055, top=0.875)
    return fig


st.markdown(
    """
    <style>
    .stApp{background:#0d0f14}
    .block-container{max-width:1380px;padding-top:1rem;padding-bottom:2rem}
    section[data-testid="stSidebar"]{background:#11141b}
    h1,h2,h3,p,label,.stMarkdown,.stCaption{color:#f2f2f2!important}
    div[data-testid="stMetric"]{background:#151922;border:1px solid #282d37;border-radius:12px;padding:9px 12px}
    div[data-testid="stMetricLabel"]{color:#aeb4bf!important}
    .nav-hour{background:#151922;border:1px solid #2a303b;border-radius:12px;padding:10px 14px;text-align:center}
    .nav-hour .main{font-size:1.35rem;font-weight:800;color:#fff}
    .nav-hour .sub{font-size:.78rem;color:#aeb4bf}
    .lightning-card{background:#151922;border:1px solid #2b303a;border-radius:14px;padding:14px 18px;margin:10px 0}
    .lightning-title{font-size:1.02rem;font-weight:800;color:#fff}
    .lightning-sub{font-size:.86rem;color:#aeb4bf}
    .legend-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
    .legend-chip{font-size:.70rem;padding:5px 9px;border-radius:999px;color:#111;font-weight:800}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(f"⚡ {SITE_TITLE}")
st.caption("OPEN-METEO • DWD ICON • PREVISÃO HORÁRIA • 0 A +6 H • MAPA 0,5° • IDW FIXO • 9 PONTOS REGIONAIS • 1 CONSULTA ICON • DADOS EM CACHE POR 1 H")

with st.sidebar:
    st.header("CONFIGURAÇÃO")
    busca = st.text_input("BUSCAR UNIDADE", "")
    opcoes = [u[0] for u in UNIDADES if busca.casefold() in u[0].casefold()]
    if not opcoes:
        st.warning("Nenhuma unidade encontrada.")
        st.stop()
    unidade_nome = st.selectbox("UNIDADE", opcoes)
    unidade = next(u for u in UNIDADES if u[0] == unidade_nome)
    ulat, ulon = unidade[1], unidade[2]
    st.markdown(f"**EXTENSÃO DA REGIÃO:** FIXA EM ±{REGION_HALFSPAN:.1f}°")
    horas = st.slider("HORIZONTE", 1, 6, 6)
    st.markdown(f"**RESOLUÇÃO OPEN-METEO:** {GRID_STEP:.1f}°")
    st.markdown(f"**POTÊNCIA IDW:** FIXA EM {IDW_POWER:.1f}")
    st.caption("A extensão, a potência e a resolução são fixas para manter a consulta rápida e a comparação espacial consistente.")
    st.info('DADOS ICON ATUALIZADOS AUTOMATICAMENTE A CADA 1 H')

GLA, GLO = make_plot_grid(ulat, ulon)
PGLA, PGLO = make_map_grid(ulat, ulon)

with st.spinner('CONSULTANDO DWD ICON • 9 PONTOS • 1 REQUISIÇÃO...'):
    try:
        df = consultar_previsao_icon_regiao(float(ulat), float(ulon))
        df = calcular_indice_holistico(df)
    except Exception as exc:
        st.error(f"ERRO AO CONSULTAR OPEN-METEO / ICON: {type(exc).__name__}: {exc}")
        st.stop()

# Tempos horários 0h ... +6h.
times = sorted(df["tempo"].drop_duplicates())[:horas + 1]
if not times:
    st.error("A API não retornou horários para a região.")
    st.stop()

if "time_index" not in st.session_state or st.session_state.time_index >= len(times):
    st.session_state.time_index = 0

c_prev, c_hour, c_next = st.columns([1, 2, 1])
with c_prev:
    if st.button("◀ HORA ANTERIOR", use_container_width=True, disabled=st.session_state.time_index == 0):
        st.session_state.time_index -= 1
        st.rerun()
with c_hour:
    st.markdown(
        f'<div class="nav-hour"><div class="main">{pd.Timestamp(times[st.session_state.time_index]):%d/%m/%Y %H:%M}</div>'
        f'<div class="sub">HORA {st.session_state.time_index:+d} • LOCAL</div></div>',
        unsafe_allow_html=True,
    )
with c_next:
    if st.button("PRÓXIMA HORA ▶", use_container_width=True, disabled=st.session_state.time_index == len(times) - 1):
        st.session_state.time_index += 1
        st.rerun()

when = times[st.session_state.time_index]
fr = df[df["tempo"] == when].copy()


vmax_p = max(8.0, float(np.nanpercentile(df["precipitation"], 98)) if np.isfinite(df["precipitation"]).any() else 8.0)
vmax_g = max(60.0, float(np.nanpercentile(df["wind_gusts_10m"], 98)) if np.isfinite(df["wind_gusts_10m"]).any() else 60.0)

T1, T2, T3 = st.tabs(["🌧️ PRECIPITAÇÃO", "💨 RAJADA DE VENTO", "⚡ POTENCIAL DE RAIOS"])

with T1:
    field = idw_grid(fr.lat, fr.lon, fr["precipitation"], PGLA, PGLO)
    fig = mapa_cartopy(PGLA, PGLO, field, unidade_nome, ulat, ulon, when, "PRECIPITAÇÃO • ICON", "PRECIPITAÇÃO (MM/H)", vmax_p, "turbo", "normal")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with T2:
    field = idw_grid(fr.lat, fr.lon, fr["wind_gusts_10m"], PGLA, PGLO)
    fig = mapa_cartopy(PGLA, PGLO, field, unidade_nome, ulat, ulon, when, "RAJADA DE VENTO • ICON", "RAJADA (KM/H)", vmax_g, "magma", "normal")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with T3:
    field = idw_grid(fr.lat, fr.lon, fr["raios_score"], PGLA, PGLO)
    fig = mapa_cartopy(
        PGLA, PGLO, field, unidade_nome, ulat, ulon, when,
        "POTENCIAL HOLÍSTICO DE RAIOS", "", 100, "YlOrRd", "raios"
    )
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# Valor na unidade e contexto ao redor.
nearest_i = ((fr["lat"] - ulat).abs() + (fr["lon"] - ulon).abs()).idxmin()
unit_row = fr.loc[nearest_i]
neighbor = fr[(fr["lat"].sub(ulat).abs() <= 1.0) & (fr["lon"].sub(ulon).abs() <= 1.0)]
local_context = float(np.nanpercentile(neighbor["raios_score"], 80)) if np.isfinite(neighbor["raios_score"]).any() else float(unit_row["raios_score"])
unit_score = 0.75 * float(unit_row["raios_score"]) + 0.25 * local_context
unit_score = float(np.clip(unit_score, 0, 100))
unit_class = str(np.select([unit_score < 20, unit_score < 40, unit_score < 60, unit_score < 80], ["MUITO BAIXO", "BAIXO", "MODERADO", "ALTO"], default="MUITO ALTO"))

st.subheader("CONDIÇÕES NA UNIDADE")
c1, c2, c3, c4 = st.columns(4)
c1.metric("PRECIPITAÇÃO", f"{float(unit_row['precipitation']):.1f} MM/H")
c2.metric("RAJADA", f"{float(unit_row['wind_gusts_10m']):.0f} KM/H")
tp = unit_row["precipitation_probability"]
c3.metric("PROB. CHUVA", "N/D" if pd.isna(tp) else f"{float(tp):.0f}%")
c4.metric("POTENCIAL DE RAIOS", f"{unit_score:.0f}/100")

color = classe_cor(unit_class)
st.markdown(
    f'<div class="lightning-card"><div class="lightning-title">⚡ {unit_class} • POTENCIAL HOLÍSTICO</div>'
    f'<div class="lightning-sub">Índice calculado exclusivamente com o DWD ICON: sinal de trovoada do código meteorológico, probabilidade de chuva, pancadas, precipitação, CAPE, umidade e nebulosidade. O sinal de trovoada do próprio ICON recebe prioridade para não mascarar situações convectivas.</div>'
    f'<div class="legend-row">'
    f'<span class="legend-chip" style="background:#2E7D32">0–19 MUITO BAIXO</span>'
    f'<span class="legend-chip" style="background:#8BC34A">20–39 BAIXO</span>'
    f'<span class="legend-chip" style="background:#FDD835">40–59 MODERADO</span>'
    f'<span class="legend-chip" style="background:#FB8C00">60–79 ALTO</span>'
    f'<span class="legend-chip" style="background:#D32F2F;color:#fff">80–100 MUITO ALTO</span>'
    f'</div>'
    f'<div style="margin-top:10px;font-weight:800;color:{color}">INDICADOR PARA A UNIDADE: {unit_score:.0f}/100 — {unit_class}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.subheader("PREVISÃO HORÁRIA NA UNIDADE")
nearest = ((df["lat"] - ulat).abs() + (df["lon"] - ulon).abs()).groupby(df["tempo"]).idxmin()
serie = df.loc[nearest].sort_values("tempo").head(horas + 1).copy()
serie["TEMPO"] = pd.to_datetime(serie["tempo"]).dt.strftime("%d/%m %H:%M")
serie["HORIZONTE"] = [f"{i:+d} H" for i in range(len(serie))]
serie["PRECIPITAÇÃO (MM/H)"] = serie["precipitation"].round(1)
serie["RAJADA (KM/H)"] = serie["wind_gusts_10m"].round(0)
serie["RAIOS (0–100)"] = serie["raios_score"].round(0)
serie["CLASSE"] = serie["raios_classe"]
tabela = serie[["HORIZONTE", "TEMPO", "PRECIPITAÇÃO (MM/H)", "RAJADA (KM/H)", "RAIOS (0–100)", "CLASSE"]]
st.dataframe(tabela, use_container_width=True, hide_index=True)
st.download_button("⬇️ BAIXAR CSV", tabela.to_csv(index=False).encode("utf-8-sig"), "previsao_openmeteo_holistica.csv", "text/csv")

st.caption("PRECIPITAÇÃO, RAJADAS E POTENCIAL DE RAIOS: DWD ICON VIA OPEN-METEO. O ÍNDICE É HEURÍSTICO E NÃO REPRESENTA UMA PROBABILIDADE ESTATÍSTICA CALIBRADA.")
