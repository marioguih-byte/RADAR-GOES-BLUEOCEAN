import io
import re
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Iterable

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
REGION_HALFSPAN = 2.5
GLM_WINDOW_MIN = 10
GLM_RECENT_MIN = 6
GLM_PREVIOUS_MIN = 6
GLM_BUCKET = "noaa-goes19"
GLM_BASE_URL = f"https://{GLM_BUCKET}.s3.amazonaws.com"
GLM_CACHE_TTL = 90
GLM_UNIT_RADIUS_KM = 75.0

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
    """Grade espacial fixa de 0,5° com 11 x 11 pontos (±2,5°)."""
    vals = np.arange(-REGION_HALFSPAN, REGION_HALFSPAN + GRID_STEP * 0.51, GRID_STEP)
    lats = lat0 + vals
    lons = lon0 + vals
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
def consultar_previsao_icon(lats_tuple, lons_tuple):
    """Consulta única ao DWD ICON Global via Open-Meteo.

    Precipitação e rajadas usadas nos mapas são provenientes do ICON.
    """
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
            r = requests.get("https://api.open-meteo.com/v1/dwd-icon", params=params, timeout=75)
            if r.status_code == 429:
                import time
                time.sleep(1.5 * (2 ** attempt))
                last = RuntimeError(f"HTTP 429 no ICON após tentativa {attempt + 1}")
                continue
            r.raise_for_status()
            return parse_openmeteo(r.json(), list(lats_tuple), list(lons_tuple), variables)
        except Exception as exc:
            last = exc
            if attempt < 3:
                import time
                time.sleep(1.0 * (2 ** attempt))
    raise last


@st.cache_data(ttl=300, show_spinner=False)
def consultar_raios_ecmwf(lats_tuple, lons_tuple):
    """Densidade de raios prevista pelo ECMWF para o índice holístico."""
    params = {
        "latitude": ",".join(f"{x:.5f}" for x in lats_tuple),
        "longitude": ",".join(f"{x:.5f}" for x in lons_tuple),
        "hourly": "lightning_density",
        "forecast_hours": MAX_HOURS + 1,
        "timezone": TZ,
        "cell_selection": "land",
    }
    last = None
    for attempt in range(3):
        try:
            r = requests.get("https://api.open-meteo.com/v1/ecmwf", params=params, timeout=60)
            if r.status_code == 429:
                import time
                time.sleep(1.5 * (2 ** attempt))
                last = RuntimeError(f"HTTP 429 no ECMWF após tentativa {attempt + 1}")
                continue
            r.raise_for_status()
            return parse_openmeteo(r.json(), list(lats_tuple), list(lons_tuple), ["lightning_density"])
        except Exception as exc:
            last = exc
            if attempt < 2:
                import time
                time.sleep(1.0 * (2 ** attempt))
    raise last


@st.cache_data(ttl=300, show_spinner=False)
def consultar_prob_trovoada_gfs(lats_tuple, lons_tuple):
    """Probabilidade de trovoada do GFS para complementar o índice."""
    params = {
        "latitude": ",".join(f"{x:.5f}" for x in lats_tuple),
        "longitude": ",".join(f"{x:.5f}" for x in lons_tuple),
        "hourly": "thunderstorm_probability",
        "forecast_hours": MAX_HOURS + 1,
        "timezone": TZ,
        "cell_selection": "land",
    }
    last = None
    for attempt in range(3):
        try:
            r = requests.get("https://api.open-meteo.com/v1/gfs", params=params, timeout=60)
            if r.status_code == 429:
                import time
                time.sleep(1.5 * (2 ** attempt))
                last = RuntimeError(f"HTTP 429 no GFS após tentativa {attempt + 1}")
                continue
            r.raise_for_status()
            return parse_openmeteo(r.json(), list(lats_tuple), list(lons_tuple), ["thunderstorm_probability"])
        except Exception as exc:
            last = exc
            if attempt < 2:
                import time
                time.sleep(1.0 * (2 ** attempt))
    raise last



def _parse_glm_timestamp(key: str):
    m = re.search(r"_s(\d{13})", key)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%j%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _listar_glm_hora(when_utc: datetime):
    when_utc = when_utc.astimezone(timezone.utc)
    prefix = f"GLM-L2-LCFA/{when_utc:%Y}/{when_utc:%j}/{when_utc:%H}/"
    params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
    r = requests.get(GLM_BASE_URL + "/", params=params, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    keys = []
    for elem in root.iter():
        if elem.tag.rsplit("}", 1)[-1] == "Key" and elem.text:
            keys.append(elem.text)
    return keys


def _achar_dataset_h5(grupo, nome):
    alvo = nome.lower()
    for key in grupo.keys():
        obj = grupo[key]
        if hasattr(obj, "keys"):
            found = _achar_dataset_h5(obj, nome)
            if found is not None:
                return found
        elif getattr(obj, "name", "").rsplit("/", 1)[-1].lower() == alvo:
            return obj
    return None


def _ler_glm_arquivo(key: str, bbox):
    import h5py

    url = f"{GLM_BASE_URL}/{key}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with h5py.File(io.BytesIO(r.content), "r") as h5:
        lat_ds = _achar_dataset_h5(h5, "flash_lat")
        lon_ds = _achar_dataset_h5(h5, "flash_lon")
        energy_ds = _achar_dataset_h5(h5, "flash_energy")
        if lat_ds is None or lon_ds is None:
            return pd.DataFrame(columns=["lat", "lon", "energia", "tempo"])
        lat = np.asarray(lat_ds[...], dtype=float).ravel()
        lon = np.asarray(lon_ds[...], dtype=float).ravel()
        if energy_ds is None:
            energy = np.zeros_like(lat)
        else:
            energy = np.asarray(energy_ds[...], dtype=float).ravel()

    if lon.size:
        lon = np.where(lon > 180.0, lon - 360.0, lon)
    tfile = _parse_glm_timestamp(key)
    if tfile is None:
        return pd.DataFrame(columns=["lat", "lon", "energia", "tempo"])

    lon0, lon1, lat0, lat1 = bbox
    n = min(lat.size, lon.size, energy.size)
    if n == 0:
        return pd.DataFrame(columns=["lat", "lon", "energia", "tempo"])
    lat, lon, energy = lat[:n], lon[:n], energy[:n]
    good = (
        np.isfinite(lat) & np.isfinite(lon) & np.isfinite(energy)
        & (lat >= lat0) & (lat <= lat1)
        & (lon >= lon0) & (lon <= lon1)
    )
    if not good.any():
        return pd.DataFrame(columns=["lat", "lon", "energia", "tempo"])
    return pd.DataFrame({
        "lat": lat[good],
        "lon": lon[good],
        "energia": np.maximum(energy[good], 0.0),
        "tempo": pd.Timestamp(tfile),
    })


@st.cache_data(ttl=GLM_CACHE_TTL, show_spinner=False)
def fetch_glm_recent(ulat: float, ulon: float, lookback_min: int = GLM_RECENT_MIN + GLM_PREVIOUS_MIN):
    """Busca flashes reais do GOES-19 GLM-L2-LCFA no AWS S3 público do NOAA.

    O filtro espacial é aplicado depois da leitura dos arquivos para manter a
    consulta limitada à região da unidade selecionada. Cada arquivo GLM-L2-LCFA
    representa um período de aproximadamente 20 s de detecções de relâmpagos.
    """
    agora = datetime.now(timezone.utc)
    inicio = agora - timedelta(minutes=int(lookback_min))
    bbox = (
        ulon - REGION_HALFSPAN - 0.75,
        ulon + REGION_HALFSPAN + 0.75,
        ulat - REGION_HALFSPAN - 0.75,
        ulat + REGION_HALFSPAN + 0.75,
    )

    horas = {inicio.replace(minute=0, second=0, microsecond=0),
             agora.replace(minute=0, second=0, microsecond=0)}
    keys = []
    falhas = []
    for h in sorted(horas):
        try:
            keys.extend(_listar_glm_hora(h))
        except Exception as exc:
            falhas.append(f"listagem {h:%H:%M}Z: {type(exc).__name__}: {exc}")

    selecionados = []
    vistos = set()
    for key in keys:
        if key in vistos:
            continue
        vistos.add(key)
        t = _parse_glm_timestamp(key)
        if t is not None and inicio <= t <= agora:
            selecionados.append((t, key))
    selecionados.sort(key=lambda x: x[0])

    if not selecionados:
        return pd.DataFrame(columns=["lat", "lon", "energia", "tempo"])

    frames = []
    max_workers = min(8, max(2, len(selecionados)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_ler_glm_arquivo, key, bbox): (t, key) for t, key in selecionados}
        for fut in as_completed(futs):
            t, key = futs[fut]
            try:
                frame = fut.result()
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                falhas.append(f"{t:%H:%M:%S}Z: {type(exc).__name__}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "energia", "tempo"])
    return pd.concat(frames, ignore_index=True).sort_values("tempo").reset_index(drop=True)


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lon1 = np.deg2rad(np.asarray(lon1, dtype=float))
    lat2 = np.deg2rad(float(lat2))
    lon2 = np.deg2rad(float(lon2))
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _glm_score_from_count(count: int) -> float:
    if count <= 0:
        return 0.0
    return float(np.clip(100.0 * (1.0 - np.exp(-count / 3.0)), 0, 100))


def _glm_motion(recent: pd.DataFrame, previous: pd.DataFrame):
    if recent.empty or previous.empty:
        return 0.0, 0.0
    t1 = recent["tempo"].max()
    t0 = previous["tempo"].max()
    dt_hours = max((t1 - t0).total_seconds() / 3600.0, 1.0 / 60.0)
    dlat = float(recent["lat"].mean() - previous["lat"].mean())
    dlon = float(recent["lon"].mean() - previous["lon"].mean())
    return float(np.clip(dlat / dt_hours, -3.0, 3.0)), float(np.clip(dlon / dt_hours, -3.0, 3.0))


def _glm_split_windows(glm_df: pd.DataFrame):
    if glm_df is None or glm_df.empty:
        return (pd.DataFrame(columns=["lat", "lon", "energia", "tempo"]),
                pd.DataFrame(columns=["lat", "lon", "energia", "tempo"]))
    tmax = glm_df["tempo"].max()
    recent_start = tmax - pd.Timedelta(minutes=GLM_RECENT_MIN)
    previous_start = recent_start - pd.Timedelta(minutes=GLM_PREVIOUS_MIN)
    recent = glm_df[glm_df["tempo"] >= recent_start].copy()
    previous = glm_df[(glm_df["tempo"] >= previous_start) & (glm_df["tempo"] < recent_start)].copy()
    return recent, previous


def _glm_field_from_points(points: pd.DataFrame, glat, glon, scale=1.0):
    if points is None or points.empty:
        return np.zeros_like(glat, dtype=float)
    values = np.full(len(points), 100.0 * float(scale), dtype=float)
    field = idw_grid(points["lat"].to_numpy(), points["lon"].to_numpy(), values, glat, glon, power=IDW_POWER, k=min(24, len(values)))
    return np.nan_to_num(np.clip(field, 0, 100), nan=0.0)


def glm_field_nowcast(glm_df: pd.DataFrame, glat, glon, hours_ahead: int):
    """Campo espacial do GLM observado e seu nowcast de deslocamento curto."""
    recent, previous = _glm_split_windows(glm_df)
    if recent.empty:
        return np.zeros_like(glat, dtype=float)
    vlat, vlon = _glm_motion(recent, previous)
    future = recent.copy()
    if hours_ahead:
        future["lat"] = future["lat"] + vlat * float(hours_ahead)
        future["lon"] = future["lon"] + vlon * float(hours_ahead)
    # Janela de 0–6 h: o peso cai progressivamente, mas a posição observada
    # continua sendo a base do nowcast de curto prazo.
    decay = float(np.exp(-float(hours_ahead) / 2.5))
    return _glm_field_from_points(future, glat, glon, decay)


def glm_unit_score(glm_df: pd.DataFrame, ulat: float, ulon: float, hours_ahead: int = 0):
    if glm_df is None or glm_df.empty:
        return 0.0, 0, None
    recent, previous = _glm_split_windows(glm_df)
    if recent.empty:
        return 0.0, 0, None
    vlat, vlon = _glm_motion(recent, previous)
    moved = recent.copy()
    if hours_ahead:
        moved["lat"] = moved["lat"] + vlat * float(hours_ahead)
        moved["lon"] = moved["lon"] + vlon * float(hours_ahead)
    dist = _haversine_km(moved["lat"], moved["lon"], ulat, ulon)
    n = int(np.sum(dist <= GLM_UNIT_RADIUS_KM))
    score = _glm_score_from_count(n)
    if hours_ahead:
        score *= float(np.exp(-float(hours_ahead) / 2.5))
    return float(np.clip(score, 0, 100)), n, recent["tempo"].max()


def calcular_indice_holistico(df):
    """Índice heurístico 0–100 usando exclusivamente variáveis do ICON.

    A ideia é priorizar sinais explícitos de tempestade do ICON e, na ausência
    desses códigos, combinar chuva convectiva, CAPE, estabilidade, umidade,
    nebulosidade e rajada para representar potencial de atividade elétrica.
    """
    p = np.clip(pd.to_numeric(df["precipitation"], errors="coerce").fillna(0).to_numpy(float), 0, None)
    pp = np.clip(pd.to_numeric(df["precipitation_probability"], errors="coerce").fillna(0).to_numpy(float), 0, 100)
    showers = np.clip(pd.to_numeric(df["showers"], errors="coerce").fillna(0).to_numpy(float), 0, None)
    gust = np.clip(pd.to_numeric(df["wind_gusts_10m"], errors="coerce").fillna(0).to_numpy(float), 0, None)
    cape = np.clip(pd.to_numeric(df["cape"], errors="coerce").fillna(0).to_numpy(float), 0, None)
    li = pd.to_numeric(df["lifted_index"], errors="coerce").fillna(0).to_numpy(float)
    cin = np.clip(pd.to_numeric(df["convective_inhibition"], errors="coerce").fillna(0).to_numpy(float), 0, None)
    rh = np.clip(pd.to_numeric(df["relative_humidity_2m"], errors="coerce").fillna(0).to_numpy(float), 0, 100)
    cloud = np.clip(pd.to_numeric(df["cloud_cover"], errors="coerce").fillna(0).to_numpy(float), 0, 100)
    wx = pd.to_numeric(df["weather_code"], errors="coerce").fillna(-1).to_numpy(float).astype(int)

    def s(x, lo, hi):
        return np.clip((x-lo)/(hi-lo), 0, 1) * 100

    # Sinais do ICON, com maior peso para evidência de convecção.
    s_storm = np.select(
        [np.isin(wx, [95]), np.isin(wx, [96, 99])],
        [82.0, 100.0], default=0.0
    )
    s_showers = s(showers, 0.2, 6.0)
    s_prec = s(p, 0.3, 10.0)
    s_cape = s(cape, 100.0, 1800.0)
    s_li = s(-li, 0.0, 6.0)
    s_cin = 100.0 - s(cin, 0.0, 100.0)
    s_rh = s(rh, 68.0, 100.0)
    s_cloud = s(cloud, 60.0, 100.0)
    s_gust = s(gust, 30.0, 80.0)
    s_prob = pp

    # Potencial convectivo sem depender de observação externa.
    base = (
        0.28*s_storm +
        0.18*s_showers +
        0.12*s_cape +
        0.10*s_li +
        0.08*s_prob +
        0.07*s_prec +
        0.06*s_rh +
        0.04*s_cloud +
        0.04*s_cin +
        0.03*s_gust
    )

    # Reforça situações em que o ICON indica explicitamente trovoada.
    explicit = s_storm > 0
    base[explicit] = np.maximum(base[explicit], s_storm[explicit])

    # Reforça convecção intensa mesmo quando o código meteorológico ainda não
    # virou 95/96/99: chuva convectiva + CAPE + umidade elevada.
    conv = (showers >= 1.0) & (cape >= 600.0) & (rh >= 70.0)
    base[conv] = np.maximum(base[conv], np.minimum(95.0, base[conv] + 18.0))

    return df.assign(
        raios_score=np.clip(base, 0, 100),
        raios_classe=np.select(
            [base < 20, base < 40, base < 60, base < 80],
            ["MUITO BAIXO", "BAIXO", "MODERADO", "ALTO"],
            default="MUITO ALTO",
        ),
    )


def classe_cor(classe):
    return {
        "MUITO BAIXO": "#2E7D32",
        "BAIXO": "#8BC34A",
        "MODERADO": "#FDD835",
        "ALTO": "#FB8C00",
        "MUITO ALTO": "#D32F2F",
    }.get(classe, "#777777")


def mapa_cartopy(GLA, GLO, field, unit, ulat, ulon, when, title, label, vmax, cmap, mode):
    fig = plt.figure(figsize=(7.85, 4.90), dpi=145, facecolor="white")
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor("white")

    d = GRID_STEP / 2.0
    lat_centers = np.asarray(GLA[:, 0], float)
    lon_centers = np.asarray(GLO[0, :], float)
    lat_edges = np.r_[lat_centers - d, lat_centers[-1] + d]
    lon_edges = np.r_[lon_centers - d, lon_centers[-1] + d]
    x0, x1 = float(lon_edges[0]), float(lon_edges[-1])
    y0, y1 = float(lat_edges[0]), float(lat_edges[-1])

    # O campo nunca fica transparente: valores ausentes recebem zero apenas
    # na visualização, mantendo a escala e o preenchimento integral do quadro.
    field = np.asarray(field, float)
    field = np.where(np.isfinite(field), field, 0.0)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
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
        cb = fig.colorbar(pm, ax=ax, pad=0.012, shrink=0.79, ticks=[10, 30, 50, 70, 90])
        cb.ax.set_yticklabels(["MUITO BAIXO", "BAIXO", "MODERADO", "ALTO", "MUITO ALTO"], fontsize=7.0)
        cb.set_label("POTENCIAL DE RAIOS • ICON", fontsize=9)
    else:
        pm = ax.pcolormesh(
            lon_edges, lat_edges, np.clip(field, 0, vmax),
            cmap=cmap, shading="flat", transform=ccrs.PlateCarree(),
            edgecolors="none", linewidth=0, antialiased=False,
            rasterized=True, zorder=1,
        )
        cb = fig.colorbar(pm, ax=ax, pad=0.012, shrink=0.79)
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

    lon_ticks = np.arange(np.ceil(x0), np.floor(x1) + 1, 1)
    lat_ticks = np.arange(np.ceil(y0), np.floor(y1) + 1, 1)
    if lon_ticks.size < 2:
        lon_ticks = np.linspace(x0, x1, 6)
    if lat_ticks.size < 2:
        lat_ticks = np.linspace(y0, y1, 6)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.tick_params(axis="both", labelsize=8, colors="#333", width=0.6)

    ax.scatter([ulon], [ulat], marker="^", s=118, facecolor="white", edgecolor="black",
               linewidth=1.8, transform=ccrs.PlateCarree(), zorder=6)
    ax.scatter([ulon], [ulat], marker="o", s=12, facecolor="black", edgecolor="white",
               linewidth=0.5, transform=ccrs.PlateCarree(), zorder=7)

    ax.set_title(
        f"{unit}\n{pd.Timestamp(when):%d/%m/%Y %H:%M} LOCAL • {title}",
        fontsize=11.2, fontweight="bold", color="#171717", pad=7,
    )
    try:
        ax.spines["geo"].set_edgecolor("#222")
        ax.spines["geo"].set_linewidth(0.8)
    except Exception:
        pass
    fig.subplots_adjust(left=0.055, right=0.91, bottom=0.08, top=0.85)
    return fig


@st.cache_data(max_entries=60, show_spinner=False)
def mapa_cartopy_png(field_bytes, field_shape, gla_bytes, glo_bytes, unit, ulat, ulon, when_iso, title, label, vmax, cmap, mode):
    field = np.frombuffer(field_bytes, dtype=np.float32).reshape(tuple(field_shape)).astype(float)
    gla = np.frombuffer(gla_bytes, dtype=np.float32).reshape(tuple(field_shape)).astype(float)
    glo = np.frombuffer(glo_bytes, dtype=np.float32).reshape(tuple(field_shape)).astype(float)
    fig = mapa_cartopy(
        gla, glo, field, unit, ulat, ulon, pd.Timestamp(when_iso), title, label, vmax, cmap, mode
    )
    # Reposiciona o campo diretamente na projeção da grade fixa usada pela função de mapa.
    # O mapa é gerado aqui apenas para criar o PNG; depois o byte array é reutilizado no cache.
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white", bbox_inches=None)
    plt.close(fig)
    return buffer.getvalue()

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
st.caption("OPEN-METEO DWD ICON + GOES-19 GLM • PREVISÃO HORÁRIA • 0 A +6 H • GRADE 0,5° • IDW FIXO • POTENCIAL HOLÍSTICO DE RAIOS • NAVEGAÇÃO OTIMIZADA")

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
        consultar_previsao_icon.clear()
        fetch_glm_recent.clear()
        st.session_state.time_index = 0
        st.rerun()

GLA, GLO = make_grid(ulat, ulon)
lats = tuple(GLA.ravel().tolist())
lons = tuple(GLO.ravel().tolist())

if "forecast_cache_key" not in st.session_state or st.session_state.forecast_cache_key != (ulat, ulon):
    st.session_state.forecast_cache_key = (ulat, ulon)
    st.session_state.time_index = 0

with st.spinner(f"CONSULTANDO OPEN-METEO / ICON — {len(lats)} PONTOS..."):
    try:
        df = consultar_previsao_icon(lats, lons)
        df = calcular_indice_holistico(df)
    except Exception as exc:
        st.error(f"ERRO AO CONSULTAR OPEN-METEO / ICON: {type(exc).__name__}: {exc}")
        st.stop()

times = sorted(df["tempo"].drop_duplicates())[:horas + 1]
if not times:
    st.error("A API não retornou horários para a região.")
    st.stop()

with st.spinner("CONSULTANDO GOES-19 GLM REAL — ÚLTIMOS 12 MINUTOS..."):
    try:
        glm_df = fetch_glm_recent(ulat, ulon, GLM_RECENT_MIN + GLM_PREVIOUS_MIN)
    except Exception:
        glm_df = pd.DataFrame(columns=["lat", "lon", "energia", "tempo"])

if "time_index" not in st.session_state or st.session_state.time_index >= len(times):
    st.session_state.time_index = 0


def _render_map_image(field, when, title, label, vmax, cmap, mode):
    safe = np.nan_to_num(np.asarray(field, dtype=np.float32), nan=0.0, posinf=100.0, neginf=0.0)
    return mapa_cartopy_png(
        safe.tobytes(), safe.shape,
        np.asarray(GLA, dtype=np.float32).tobytes(),
        np.asarray(GLO, dtype=np.float32).tobytes(),
        unidade_nome, ulat, ulon, pd.Timestamp(when).isoformat(),
        title, label, float(vmax), cmap, mode,
    )


@st.fragment
def render_forecast_fragment():
    c_prev, c_hour, c_next = st.columns([1, 2, 1])
    with c_prev:
        if st.button("◀ HORA ANTERIOR", use_container_width=True, disabled=st.session_state.time_index == 0, key="prev_hour"):
            st.session_state.time_index = max(0, int(st.session_state.time_index) - 1)
    with c_next:
        if st.button("PRÓXIMA HORA ▶", use_container_width=True, disabled=st.session_state.time_index == len(times) - 1, key="next_hour"):
            st.session_state.time_index = min(len(times) - 1, int(st.session_state.time_index) + 1)

    # O índice é atualizado pelos botões antes de desenhar o cabeçalho.
    # Isso evita o atraso de uma interação em que o mapa já avança, mas
    # o horário superior ainda mostra a hora anterior.
    current_index = int(st.session_state.time_index)
    when = times[current_index]

    with c_hour:
        st.markdown(
            f'<div class="nav-hour"><div class="main">{pd.Timestamp(when):%d/%m/%Y %H:%M}</div>'
            f'<div class="sub">HORA {current_index:+d} • LOCAL</div></div>',
            unsafe_allow_html=True,
        )
    fr = df[df["tempo"] == when].copy()

    vmax_p = max(8.0, float(np.nanpercentile(df["precipitation"], 98)) if np.isfinite(df["precipitation"]).any() else 8.0)
    vmax_g = max(60.0, float(np.nanpercentile(df["wind_gusts_10m"], 98)) if np.isfinite(df["wind_gusts_10m"]).any() else 60.0)

    try:
        camada = st.segmented_control(
            "CAMADA",
            ["🌧️ PRECIPITAÇÃO", "💨 RAJADA DE VENTO", "⚡ POTENCIAL DE RAIOS"],
            default="🌧️ PRECIPITAÇÃO",
            key="camada_mapa",
            label_visibility="collapsed",
        )
    except AttributeError:
        camada = st.radio(
            "CAMADA",
            ["🌧️ PRECIPITAÇÃO", "💨 RAJADA DE VENTO", "⚡ POTENCIAL DE RAIOS"],
            horizontal=True,
            key="camada_mapa",
            label_visibility="collapsed",
        )

    if camada == "🌧️ PRECIPITAÇÃO":
        field = idw_grid(fr.lat, fr.lon, fr["precipitation"], GLA, GLO)
        png = _render_map_image(field, when, "PRECIPITAÇÃO • ICON", "PRECIPITAÇÃO (MM/H)", vmax_p, "turbo", "normal")
        st.image(png, width="stretch")

    elif camada == "💨 RAJADA DE VENTO":
        field = idw_grid(fr.lat, fr.lon, fr["wind_gusts_10m"], GLA, GLO)
        png = _render_map_image(field, when, "RAJADA DE VENTO • ICON", "RAJADA (KM/H)", vmax_g, "magma", "normal")
        st.image(png, width="stretch")

    else:
        icon_field = idw_grid(fr.lat, fr.lon, fr["raios_score"], GLA, GLO)
        glm_field = glm_field_nowcast(glm_df, GLA, GLO, current_index) if not glm_df.empty else np.zeros_like(icon_field)
        if not glm_df.empty and current_index == 0:
            field = np.maximum(icon_field, 0.95 * glm_field)
        elif not glm_df.empty:
            h = current_index
            w_glm = float(0.65 * np.exp(-h / 3.0))
            field = np.clip((1.0 - w_glm) * icon_field + w_glm * glm_field, 0, 100)
        else:
            field = icon_field
        png = _render_map_image(
            field, when,
            "POTENCIAL HOLÍSTICO DE RAIOS • ICON + GOES-19 GLM", "", 100, "YlOrRd", "raios",
        )
        st.image(png, width="stretch")
        if glm_df.empty:
            st.caption("GOES-19 GLM: nenhuma observação recente disponível; o mapa utiliza o potencial previsto pelo ICON.")
        else:
            glm_recent_count = int(len(_glm_split_windows(glm_df)[0]))
            glm_last = glm_df["tempo"].max()
            age_min = max(0.0, (datetime.now(timezone.utc) - glm_last.to_pydatetime().astimezone(timezone.utc)).total_seconds() / 60.0)
            st.caption(f"GOES-19 GLM real: {glm_recent_count} flashes na janela recente • última observação {glm_last:%d/%m %H:%M:%S}Z • atraso {age_min:.1f} min. Sem pontos individuais no mapa.")

    nearest_i = ((fr["lat"] - ulat).abs() + (fr["lon"] - ulon).abs()).idxmin()
    unit_row = fr.loc[nearest_i]
    neighbor = fr[(fr["lat"].sub(ulat).abs() <= 1.0) & (fr["lon"].sub(ulon).abs() <= 1.0)]
    local_context = float(np.nanpercentile(neighbor["raios_score"], 80)) if np.isfinite(neighbor["raios_score"]).any() else float(unit_row["raios_score"])
    icon_unit_score = float(np.clip(0.75 * float(unit_row["raios_score"]) + 0.25 * local_context, 0, 100))
    glm_now_score, glm_now_count, _ = glm_unit_score(glm_df, ulat, ulon, 0)
    if current_index == 0 and glm_now_count > 0:
        unit_score = max(icon_unit_score, glm_now_score)
    elif not glm_df.empty:
        glm_future_score, _, _ = glm_unit_score(glm_df, ulat, ulon, current_index)
        w_glm = float(0.65 * np.exp(-current_index / 3.0))
        unit_score = (1.0 - w_glm) * icon_unit_score + w_glm * glm_future_score
    else:
        unit_score = icon_unit_score
    unit_score = float(np.clip(unit_score, 0, 100))
    unit_class = str(np.select([unit_score < 20, unit_score < 40, unit_score < 60, unit_score < 80], ["MUITO BAIXO", "BAIXO", "MODERADO", "ALTO"], default="MUITO ALTO"))

    st.subheader("CONDIÇÕES NA UNIDADE")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PRECIPITAÇÃO", f"{float(unit_row['precipitation']):.1f} MM/H")
    c2.metric("RAJADA", f"{float(unit_row['wind_gusts_10m']):.0f} KM/H")
    tp = unit_row["precipitation_probability"]
    c3.metric("PROB. CHUVA", "N/D" if pd.isna(tp) else f"{float(tp):.0f}%")
    c4.metric("CAPE", f"{float(unit_row['cape']):.0f} J/KG")
    c5.metric("POTENCIAL DE RAIOS", f"{unit_score:.0f}/100")

    color = classe_cor(unit_class)
    st.markdown(
        f'<div class="lightning-card"><div class="lightning-title">⚡ {unit_class} • POTENCIAL HOLÍSTICO</div>'
        f'<div class="lightning-sub">A base prevista usa o ICON. No horário corrente, o resultado incorpora flashes reais observados pelo GLM do GOES-19; nas horas seguintes, a observação recente alimenta um nowcast espacial com deslocamento e decaimento de curto prazo.</div>'
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
    raios_tabela = []
    for h, (_, row) in enumerate(serie.iterrows()):
        icon_score = float(row["raios_score"])
        if not glm_df.empty and h == 0:
            gs, gc, _ = glm_unit_score(glm_df, ulat, ulon, 0)
            score = max(icon_score, gs) if gc > 0 else icon_score
        elif not glm_df.empty:
            gs, _, _ = glm_unit_score(glm_df, ulat, ulon, h)
            w = float(0.65 * np.exp(-h / 3.0))
            score = (1.0 - w) * icon_score + w * gs
        else:
            score = icon_score
        raios_tabela.append(float(np.clip(score, 0, 100)))
    serie["RAIOS (0–100)"] = np.round(raios_tabela, 0)
    serie["CLASSE"] = np.select(
        [serie["RAIOS (0–100)"] < 20, serie["RAIOS (0–100)"] < 40, serie["RAIOS (0–100)"] < 60, serie["RAIOS (0–100)"] < 80],
        ["MUITO BAIXO", "BAIXO", "MODERADO", "ALTO"], default="MUITO ALTO"
    )
    tabela = serie[["HORIZONTE", "TEMPO", "PRECIPITAÇÃO (MM/H)", "RAJADA (KM/H)", "RAIOS (0–100)", "CLASSE"]]
    st.dataframe(tabela, width="stretch", hide_index=True)
    st.download_button("⬇️ BAIXAR CSV", tabela.to_csv(index=False).encode("utf-8-sig"), "previsao_openmeteo_holistica.csv", "text/csv")


render_forecast_fragment()

st.caption("PRECIPITAÇÃO E RAJADAS: OPEN-METEO / DWD ICON. RAIOS: potencial previsto do ICON combinado com observação real recente do GOES-19 GLM e nowcast de curto prazo. O índice não representa uma probabilidade estatística calibrada.")
