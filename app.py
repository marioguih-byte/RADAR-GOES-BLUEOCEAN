import io
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import h5py
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
REGION_HALFSPAN = 2.5
GLM_WINDOW_MIN = 10

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


def idw_grid(points_lat, points_lon, values, grid_lat, grid_lon, power=IDW_POWER, k=24):
    la = np.asarray(points_lat, dtype=float)
    lo = np.asarray(points_lon, dtype=float)
    va = np.asarray(values, dtype=float)
    good = np.isfinite(la) & np.isfinite(lo) & np.isfinite(va)
    la, lo, va = la[good], lo[good], va[good]
    if va.size == 0:
        return np.full_like(grid_lat, np.nan, dtype=float)

    lat_ref = np.deg2rad(np.nanmean(la))
    xp = np.deg2rad(lo) * np.cos(lat_ref) * 6371.0
    yp = np.deg2rad(la) * 6371.0
    xg = np.deg2rad(np.asarray(grid_lon).ravel()) * np.cos(lat_ref) * 6371.0
    yg = np.deg2rad(np.asarray(grid_lat).ravel()) * 6371.0

    out = np.full(xg.size, np.nan, dtype=float)
    for start in range(0, out.size, 3500):
        sl = slice(start, start + 3500)
        d = np.hypot(xg[sl, None] - xp[None, :], yg[sl, None] - yp[None, :])
        n = min(k, d.shape[1])
        idx = np.argpartition(d, n - 1, axis=1)[:, :n]
        dd = np.take_along_axis(d, idx, axis=1)
        vv = va[idx]
        exact = dd < 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = 1.0 / np.maximum(dd, 0.001) ** power
            z = np.sum(weights * vv, axis=1) / np.sum(weights, axis=1)
        rows = np.where(exact.any(axis=1))[0]
        if rows.size:
            cols = np.argmax(exact[rows], axis=1)
            z[rows] = vv[rows, cols]
        out[sl] = z
    return out.reshape(np.asarray(grid_lat).shape)


def make_grid(lat0, lon0):
    """Grade espacial fixa em 0,5° ao redor da unidade."""
    halfspan = REGION_HALFSPAN
    step = GRID_STEP
    lat_min = np.floor((lat0 - halfspan) / step) * step
    lat_max = np.ceil((lat0 + halfspan) / step) * step
    lon_min = np.floor((lon0 - halfspan) / step) * step
    lon_max = np.ceil((lon0 + halfspan) / step) * step
    lats = np.arange(lat_min, lat_max + step * 0.51, step)
    lons = np.arange(lon_min, lon_max + step * 0.51, step)
    return np.meshgrid(lats, lons, indexing="ij")


def batches(seq: Iterable, size: int):
    seq = list(seq)
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def parse_openmeteo(raw, lats, lons, variable_names):
    items = raw if isinstance(raw, list) else [raw]
    rows = []
    for i, item in enumerate(items):
        h = item.get("hourly", {})
        times = h.get("time", [])
        for j, t in enumerate(times):
            row = {"lat": float(lats[i]), "lon": float(lons[i]), "tempo": pd.Timestamp(t)}
            for var in variable_names:
                vals = h.get(var, [])
                row[var] = vals[j] if j < len(vals) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def consultar_previsao_ecmwf(lats_tuple, lons_tuple):
    """Uma única chamada ao Open-Meteo/ECMWF para toda a grade."""
    variables = [
        "precipitation",
        "precipitation_probability",
        "rain",
        "showers",
        "wind_gusts_10m",
        "cape",
        "lifted_index",
        "convective_inhibition",
        "relative_humidity_2m",
        "cloud_cover",
        "weather_code",
        "lightning_density",
    ]
    params = {
        "latitude": ",".join(f"{x:.5f}" for x in lats_tuple),
        "longitude": ",".join(f"{x:.5f}" for x in lons_tuple),
        "hourly": ",".join(variables),
        "forecast_hours": MAX_HOURS + 1,
        "timezone": TZ,
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
        "cell_selection": "land",
    }
    last = None
    for attempt in range(4):
        try:
            r = requests.get("https://api.open-meteo.com/v1/ecmwf", params=params, timeout=75)
            if r.status_code == 429:
                wait = 1.5 * (2 ** attempt)
                import time
                time.sleep(wait)
                last = RuntimeError(f"HTTP 429 após tentativa {attempt + 1}")
                continue
            r.raise_for_status()
            return parse_openmeteo(r.json(), list(lats_tuple), list(lons_tuple), variables)
        except Exception as exc:
            last = exc
            if attempt < 3:
                import time
                time.sleep(1.0 * (2 ** attempt))
    raise last


def calcular_indice_holistico(df):
    p = np.clip(df["precipitation"].fillna(0).to_numpy(float), 0, None)
    pp = np.clip(df["precipitation_probability"].fillna(0).to_numpy(float), 0, 100)
    showers = np.clip(df["showers"].fillna(0).to_numpy(float), 0, None)
    cape = np.clip(df["cape"].fillna(0).to_numpy(float), 0, None)
    li = df["lifted_index"].fillna(0).to_numpy(float)
    cin = df["convective_inhibition"].fillna(0).to_numpy(float)
    rh = np.clip(df["relative_humidity_2m"].fillna(0).to_numpy(float), 0, 100)
    cloud = np.clip(df["cloud_cover"].fillna(0).to_numpy(float), 0, 100)
    ld = np.clip(df["lightning_density"].fillna(0).to_numpy(float), 0, None)
    code = df["weather_code"].fillna(-1).to_numpy(float)
    glm = np.clip(df["glm_score"].fillna(0).to_numpy(float), 0, 100)

    s_showers = normalizar_0_1(showers, 0.2, 8.0) * 100
    s_cape = normalizar_0_1(cape, 100, 1800) * 100
    s_li = normalizar_0_1(-li, 0.0, 5.0) * 100
    s_rh = normalizar_0_1(rh, 65, 100) * 100
    s_cloud = normalizar_0_1(cloud, 55, 100) * 100
    s_cin = normalizar_0_1(-cin, 0, 120) * 100
    s_prec = normalizar_0_1(p, 0.3, 12) * 100
    s_code = np.where(np.isin(code.astype(int), [95, 96, 99]), 100.0, 0.0)

    # Densidade de raios prevista pelo IFS: normalização robusta por horário.
    density_score = np.zeros(len(df), dtype=float)
    if np.isfinite(ld).any() and np.nanmax(ld) > 0:
        density_score = np.empty(len(df), dtype=float)
        for t, idx in df.groupby("tempo").groups.items():
            vals = ld[np.asarray(list(idx), dtype=int)]
            ref = max(float(np.nanpercentile(vals, 90)), 0.05)
            density_score[np.asarray(list(idx), dtype=int)] = np.clip(vals / ref * 100.0, 0, 100)

    # Sem GLM futuro, o modelo domina; o GLM atual entra somente no instante 0.
    model_score = (
        0.38 * density_score +
        0.16 * s_showers +
        0.12 * s_prec +
        0.10 * pp +
        0.08 * s_cape +
        0.05 * s_li +
        0.04 * s_rh +
        0.03 * s_cloud +
        0.02 * s_cin +
        0.02 * s_code
    )
    base_time = pd.Timestamp(df["tempo"].min())
    lead_h = (pd.to_datetime(df["tempo"]) - base_time).dt.total_seconds().to_numpy() / 3600.0

    # Observação GLM recente influencia fortemente a hora inicial e decai rapidamente.
    glm_weight = 0.78 * np.exp(-np.maximum(lead_h, 0) / 1.25)
    score = (1 - glm_weight) * model_score + glm_weight * glm

    counts = df["glm_flashes_10min"].fillna(0).to_numpy(float)
    current = lead_h <= 0.51
    floors = [(30,90),(15,80),(8,65),(4,50),(2,35),(1,20)]
    for threshold, floor in floors:
        mask = current & (counts >= threshold)
        score[mask] = np.maximum(score[mask], floor)
    return df.assign(
        raios_score=np.clip(score, 0, 100),
        raios_classe=np.select(
            [score < 20, score < 40, score < 60, score < 80],
            ["MUITO BAIXO", "BAIXO", "MODERADO", "ALTO"],
            default="MUITO ALTO",
        ),
    )

def _parse_s3_key_time(key):
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    if not m:
        return None
    year, doy, hh, mm, ss = map(int, m.groups())
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy-1, hours=hh, minutes=mm, seconds=ss)


@st.cache_data(ttl=60, show_spinner=False)
def consultar_glm_10min(unit_lat, unit_lon):
    """Atividade GLM recente apenas para a unidade, com downloads concorrentes."""
    now = datetime.now(timezone.utc)
    prefixes = [f"GLM-L2-LCFA/{now:%Y}/{now.timetuple().tm_yday:03d}/{now:%H}/"]
    if (now - timedelta(minutes=10)).hour != now.hour:
        h = now - timedelta(hours=1)
        prefixes.append(f"GLM-L2-LCFA/{h:%Y}/{h.timetuple().tm_yday:03d}/{h:%H}/")

    keys=[]
    for prefix in prefixes:
        try:
            r=requests.get("https://noaa-goes19.s3.amazonaws.com/",params={"list-type":"2","prefix":prefix,"max-keys":1000},timeout=20)
            r.raise_for_status()
            keys += re.findall(r"<Key>([^<]*GLM-L2-LCFA[^<]*)</Key>",r.text)
        except Exception:
            pass
    cutoff=now-timedelta(minutes=GLM_WINDOW_MIN)
    selected=[]
    for key in keys:
        t=_parse_s3_key_time(key)
        if t is not None and cutoff<=t<=now+timedelta(seconds=30):
            selected.append((t,key))
    selected=sorted(selected)[-36:]
    if not selected:
        return 0,0,"GLM SEM DADOS"

    def fetch(item):
        _,key=item
        try:
            rr=requests.get(f"https://noaa-goes19.s3.amazonaws.com/{key}",timeout=30)
            rr.raise_for_status()
            with h5py.File(io.BytesIO(rr.content),"r") as ds:
                if "flash_lat" not in ds or "flash_lon" not in ds:
                    return np.array([]),np.array([]),False
                la=np.asarray(ds["flash_lat"][:],dtype=float)
                lo=np.asarray(ds["flash_lon"][:],dtype=float)
            good=np.isfinite(la)&np.isfinite(lo)
            return la[good],lo[good],True
        except Exception:
            return np.array([]),np.array([]),False

    lats=[]; lons=[]; ok=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs=[ex.submit(fetch,item) for item in selected]
        for fut in as_completed(futs):
            la,lo,good=fut.result()
            if good:
                ok+=1
                if la.size:
                    lats.append(la); lons.append(lo)
    if not lats:
        return 0,ok,"GLM SEM FLASHES"
    fla=np.concatenate(lats); flo=np.concatenate(lons)
    lat0=np.deg2rad(float(unit_lat))
    dx=(flo-float(unit_lon))*111.32*np.cos(lat0)
    dy=(fla-float(unit_lat))*111.32
    count=int(np.count_nonzero(dx*dx+dy*dy <= 75.0**2))
    score=float(100.0*(1.0-np.exp(-count/12.0)))
    return count,ok,f"GLM OK • {ok} arquivos"

def classe_cor(classe):
    return {
        "MUITO BAIXO": "#2E7D32",
        "BAIXO": "#8BC34A",
        "MODERADO": "#FDD835",
        "ALTO": "#FB8C00",
        "MUITO ALTO": "#D32F2F",
    }.get(classe, "#777777")


def mapa_cartopy(GLA, GLO, field, unit, ulat, ulon, when, title, label, vmax, cmap, mode):
    fig = plt.figure(figsize=(7.2, 4.35), dpi=135, facecolor="white")
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor("white")

    dlat = GRID_STEP / 2
    dlon = GRID_STEP / 2
    x0 = float(GLO.min()) - dlon
    x1 = float(GLO.max()) + dlon
    y0 = float(GLA.min()) - dlat
    y1 = float(GLA.max()) + dlat
    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

    if mode == "raios":
        colors = ["#2E7D32", "#8BC34A", "#FDD835", "#FB8C00", "#D32F2F"]
        bounds = np.array([0, 20, 40, 60, 80, 100], float)
        norm = BoundaryNorm(bounds, len(colors))
        pm = ax.pcolormesh(
            GLO, GLA, np.clip(field, 0, 100),
            cmap=ListedColormap(colors), norm=norm,
            shading="nearest", transform=ccrs.PlateCarree(),
            edgecolors="none", linewidth=0, antialiased=False, rasterized=True,
            zorder=1,
        )
        cb = fig.colorbar(pm, ax=ax, pad=0.018, shrink=0.80, ticks=[10, 30, 50, 70, 90])
        cb.ax.set_yticklabels(["MUITO BAIXO", "BAIXO", "MODERADO", "ALTO", "MUITO ALTO"], fontsize=7.2)
        cb.set_label("POTENCIAL HOLÍSTICO DE RAIOS", fontsize=9)
    else:
        pm = ax.pcolormesh(
            GLO, GLA, np.clip(field, 0, vmax),
            cmap=cmap, shading="nearest", transform=ccrs.PlateCarree(),
            edgecolors="none", linewidth=0, antialiased=False, rasterized=True,
            zorder=1,
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
    if len(lon_ticks) > 9:
        lon_ticks = np.linspace(x0, x1, 8)
    if len(lat_ticks) > 9:
        lat_ticks = np.linspace(y0, y1, 7)

    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.tick_params(axis="both", labelsize=8, colors="#333", width=0.6)

    for x in lon_ticks:
        ax.plot([x, x], [y0, y1], color="#888", linewidth=0.25, alpha=0.22, transform=ccrs.PlateCarree(), zorder=2)
    for y in lat_ticks:
        ax.plot([x0, x1], [y, y], color="#888", linewidth=0.25, alpha=0.22, transform=ccrs.PlateCarree(), zorder=2)

    ax.scatter([ulon], [ulat], marker="^", s=112, facecolor="white", edgecolor="black", linewidth=1.8, transform=ccrs.PlateCarree(), zorder=6)
    ax.scatter([ulon], [ulat], marker="o", s=14, facecolor="black", edgecolor="white", linewidth=0.5, transform=ccrs.PlateCarree(), zorder=7)

    ax.set_title(
        f"{unit}\n{pd.Timestamp(when):%d/%m/%Y %H:%M} local • {title}",
        fontsize=12, fontweight="bold", color="#171717", pad=7,
    )
    try:
        ax.spines["geo"].set_edgecolor("#222")
        ax.spines["geo"].set_linewidth(0.8)
    except Exception:
        pass
    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.08, top=0.84)
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
st.caption("OPEN-METEO ECMWF • PREVISÃO HORÁRIA • 0 A +6 H • GRADE 0,5° • IDW FIXO • POTENCIAL HOLÍSTICO DE RAIOS")

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
    if st.button("🔄 ATUALIZAR AGORA", use_container_width=True):
        consultar_previsao_ecmwf.clear()
        consultar_glm_10min.clear()
        st.session_state.time_index = 0
        st.rerun()

GLA, GLO = make_grid(ulat, ulon)
lats = tuple(GLA.ravel().tolist())
lons = tuple(GLO.ravel().tolist())

with st.spinner(f"CONSULTANDO OPEN-METEO/ECMWF E GLM — {len(lats)} PONTOS..."):
    try:
        df = consultar_previsao_ecmwf(lats, lons)
        glm_unit_count, glm_files_ok, glm_status = 0, 0, "GLM INDISPONÍVEL"
        try:
            glm_unit_count, glm_files_ok, glm_status = consultar_glm_10min(ulat, ulon)
        except Exception as exc:
            glm_status = f"GLM INDISPONÍVEL • {type(exc).__name__}"
        df["glm_flashes_10min"] = float(glm_unit_count)
        df["glm_score"] = float(100.0 * (1.0 - np.exp(-glm_unit_count / 12.0)))
        # Mantém o valor observado apenas no horário inicial; nos demais horários é zerado.
        base_t = pd.Timestamp(df["tempo"].min())
        df.loc[df["tempo"] != base_t, "glm_flashes_10min"] = 0.0
        df.loc[df["tempo"] != base_t, "glm_score"] = 0.0
        df = calcular_indice_holistico(df)
    except Exception as exc:
        st.error(f"ERRO AO CONSULTAR OPEN-METEO/GLM: {type(exc).__name__}: {exc}")
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

# Grade de plotagem com 0,5° para preservar o aspecto pixelado.
plot_lat = np.arange(float(GLO.min()) * 0 + float(GLO.min()), float(GLO.max()) + GRID_STEP * 0.51, GRID_STEP)
plot_lon = np.arange(float(GLO.min()), float(GLO.max()) + GRID_STEP * 0.51, GRID_STEP)
# GLA/GLO já formam uma grade regular; usar os próprios centros mantém os pixels originais.
PGLA, PGLO = GLA.copy(), GLO.copy()

vmax_p = max(8.0, float(np.nanpercentile(df["precipitation"], 98)) if np.isfinite(df["precipitation"]).any() else 8.0)
vmax_g = max(60.0, float(np.nanpercentile(df["wind_gusts_10m"], 98)) if np.isfinite(df["wind_gusts_10m"]).any() else 60.0)

T1, T2, T3 = st.tabs(["🌧️ PRECIPITAÇÃO", "💨 RAJADA DE VENTO", "⚡ POTENCIAL DE RAIOS"])

with T1:
    field = idw_grid(fr.lat, fr.lon, fr["precipitation"], PGLA, PGLO)
    fig = mapa_cartopy(PGLA, PGLO, field, unidade_nome, ulat, ulon, when, "PRECIPITAÇÃO", "PRECIPITAÇÃO (MM/H)", vmax_p, "turbo", "normal")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with T2:
    field = idw_grid(fr.lat, fr.lon, fr["wind_gusts_10m"], PGLA, PGLO)
    fig = mapa_cartopy(PGLA, PGLO, field, unidade_nome, ulat, ulon, when, "RAJADA DE VENTO", "RAJADA (KM/H)", vmax_g, "magma", "normal")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with T3:
    field = idw_grid(fr.lat, fr.lon, fr["raios_score"], PGLA, PGLO)
    fig = mapa_cartopy(PGLA, PGLO, field, unidade_nome, ulat, ulon, when, "POTENCIAL HOLÍSTICO DE RAIOS", "", 100, "YlOrRd", "raios")
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
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("PRECIPITAÇÃO", f"{float(unit_row['precipitation']):.1f} MM/H")
c2.metric("RAJADA", f"{float(unit_row['wind_gusts_10m']):.0f} KM/H")
tp = unit_row['thunderstorm_probability']
c3.metric("PROB. TROVOADA", "N/D" if pd.isna(tp) else f"{float(tp):.0f}%")
c4.metric("RAIOS GLM • 10 MIN", f"{int(glm_unit_count)}")
c5.metric("POTENCIAL DE RAIOS", f"{unit_score:.0f}/100")

color = classe_cor(unit_class)

st.caption(f"GLM: {glm_status} • {glm_files_ok} arquivos processados • janela dos últimos 10 min • raio local de 75 km")
st.markdown(
    f'<div class="lightning-card"><div class="lightning-title">⚡ {unit_class} • POTENCIAL HOLÍSTICO</div>'
    f'<div class="lightning-sub">O índice combina densidade de raios prevista pelo ECMWF, chuva/pancadas, CAPE, Lifted Index, umidade, nebulosidade e outros sinais convectivos. A atividade recente do GLM do GOES-19 entra como observação no horário atual para corrigir tempestades já em andamento.</div>'
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

st.caption("Fonte de previsão: Open-Meteo/ECMWF. Observação recente de descargas: GLM/GOES-19. O indicador holístico combina previsão e observação recente; não é uma probabilidade estatística calibrada de raios.")
