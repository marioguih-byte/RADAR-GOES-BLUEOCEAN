from __future__ import annotations

import datetime as dt
import re
import sys
import threading
import traceback
import unicodedata
from pathlib import Path
import io
import numpy as np
import pandas as pd
import requests
import xarray as xr
try:
    import h5py  # noqa: F401 — backend HDF5 usado por h5netcdf
except ImportError as exc:
    raise RuntimeError(
        "Dependência ausente: h5py. Instale as dependências de requirements.txt e reinicie o aplicativo."
    ) from exc
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import (gaussian_filter, generic_filter,
                           map_coordinates, uniform_filter)

try:
    from pysteps import motion as pysteps_motion
    from pysteps import nowcasts as pysteps_nowcasts
    TEM_PYSTEPS = True
except ImportError:
    TEM_PYSTEPS = False

# Sem cartopy o mapa ainda sai, só perde costa, fronteiras e divisas estaduais.
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    TEM_CARTOPY = True
except ImportError:
    ccrs = cfeature = LatitudeFormatter = LongitudeFormatter = None
    TEM_CARTOPY = False


# --------------------------------------------------------------------------- #
# Núcleo IMERG (cópia de nowcasting_imerg_petrobras.py)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Parâmetros fixos
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Credencial
# --------------------------------------------------------------------------- #
# Preencha uma vez e não precisa mais passar --email a cada execução.
# Cadastro gratuito e imediato em:
#     https://registration.pps.eosdis.nasa.gov/registration/
# No PPS o usuário E a senha são o próprio e-mail cadastrado.
EMAIL_PPS = "marioguih@gmail.com"
BASE_PPS = "https://jsimpsonhttps.pps.eosdis.nasa.gov"
BASE_GESDISC = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07"
# domínio de trabalho: cobre de Belém (-1,4°) a Canoas (-29,9°) com margem
DOM_LAT = (-34.0, 2.0)
DOM_LON = (-56.0, -31.0)
LIMIARES = [                      # mm/h -> classe
    (0.2, "Sem chuva"),
    (2.5, "Chuva fraca"),
    (10.0, "Chuva moderada"),
    (50.0, "Chuva forte"),
    (np.inf, "Chuva muito forte"),
]
# --------------------------------------------------------------------------- #
# 2. Download do IMERG Early Run
# --------------------------------------------------------------------------- #
# O nome do arquivo não é montado na mão: o sufixo de versão muda (V07A, V07B,
# V07C...) e isso produziria 404 silencioso a cada troca. O código lista o
# diretório do dia e casa o arquivo pelo carimbo de tempo.
_CACHE_LISTAGEM: dict[str, list[str]] = {}
# Unidades Petrobras embutidas (nome, latitude, longitude), conforme a planilha
# LAT_E_LON_ESTAÇÕES_PBR.xlsx. Use --planilha para sobrescrever por um arquivo.
UNIDADES_PADRAO = [
    ("UTE Termocamaçari - UTE TCA",                            -12.666868,  -38.314687),
    ("UTE Termobahia - UTE TBA",                               -12.703235,  -38.564904),
    ("UTE Termoceará - UTE TCE",                                -3.692456,  -38.870605),
    ("UTE Vale do Açu - UTE VLA",                               -5.381685,  -36.819754),
    ("Refinaria Abreu e Lima - RNEST",                          -8.379661,  -35.010198),
    ("Unidade de Tratamento de Gás Sul Capixaba - UTGSUL",     -20.794464,  -40.620910),
    ("Unidade de Tratamento de Gás de Cacimbas - UTGC",        -19.463097,  -39.760603),
    ("Refinaria Duque de Caxias - REDUC",                      -22.715100,  -43.284008),
    ("UTE Termorio - UTE TRI",                                 -22.714882,  -43.254348),
    ("BOAVENTURA, Itaboraí-RJ",                                -22.660714,  -42.853629),
    ("Unidade de Tratamento de Gás de Cabiúnas - UTGCAB",      -22.285327,  -41.717905),
    ("UTE Termomacaé - UTE TMA",                               -22.306164,  -41.876702),
    ("UTE Seropédica/Baixada Fluminense - UTE SRP/BF",         -22.723292,  -43.647719),
    ("Refinaria Gabriel Passos - REGAP",                       -19.964281,  -44.095138),
    ("UTE Ibirité - UTE IBT",                                  -19.988579,  -44.098207),
    ("UTE Juiz de Fora - UTE JF",                              -21.690619,  -43.456720),
    ("UTE Três Lagoas - UTE TLG",                              -20.745500,  -51.664590),
    ("Unidade de Tratamento de Gás de Caraguatatuba - UTGCA",  -23.654191,  -45.501209),
    ("Refinaria Presidente Bernardes - RPBC",                  -23.873326,  -46.427572),
    ("UTE Cubatão - UTE CBT",                                  -23.875735,  -46.431387),
    ("Refinaria Henrique Lage - REVAP",                        -23.184796,  -45.815806),
    ("Refinaria de Capuava - RECAP",                           -23.656681,  -46.480883),
    ("Refinaria de Paulínia - REPLAN",                         -22.729589,  -47.147712),
    ("UTE Nova Piratininga - UTE NPI",                         -23.699414,  -46.673881),
    ("Refinaria Presidente Getúlio Vargas - REPAR",            -25.566140,  -49.369423),
    ("Refinaria Alberto Pasqualini - REFAP",                   -29.869901,  -51.178195),
    ("UTE Canoas - UTE CAN",                                   -29.875071,  -51.145445),
    ("Armazém Rio de Janeiro",                                 -22.810973,  -43.281876),
    ("CILEP - CENPES",                                         -22.854198,  -43.233830),
    ("Porto Baia de Guanabara ",                               -22.878942,  -43.209330),
    ("ARM Macaé - Armazém Macaé",                              -22.415317,  -41.861353),
    ("Porto de Imbetiba - Macaé",                              -22.386825,  -41.768741),
    ("Porto Açu",                                              -21.864737,  -41.016444),
    ("Porto Aratu ",                                           -12.780132,  -38.496764),
    ("Porto TMIB",                                             -10.824129,  -36.946301),
    ("Porto Belém",                                             -1.439901,  -48.494920),
    ("Porto Valença",                                          -13.369369,  -39.071252),
    ("Porto Guamaré",                                           -5.106692,  -36.319592),
    ("Porto Mucuripe ",                                         -3.713117,  -38.474037),
    ("Porto Paracuru",                                          -3.401146,  -39.010896),
]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def abrir_sessao(fonte: str, email: str | None) -> requests.Session:
    s = requests.Session()
    if fonte == "PPS_NRT":
        email = email or EMAIL_PPS
        if not email or "@" not in email:
            raise ValueError(
                "e-mail do PPS não definido.\n"
                "  1. cadastre-se em https://registration.pps.eosdis.nasa.gov/registration/\n"
                "  2. preencha a constante EMAIL_PPS no topo do script,\n"
                "     ou passe --email seu.email@dominio.com\n"
                "  (no PPS, usuário e senha são o próprio e-mail cadastrado)")
        s.auth = (email, email)           # PPS: usuário = senha = e-mail
    else:
        s.trust_env = True                # usa ~/.netrc para urs.earthdata.nasa.gov
    s.headers.update({"User-Agent": "imerg-nowcast-petrobras/1.0"})
    return s


def url_diretorio(t: dt.datetime, fonte: str) -> str:
    if fonte == "PPS_NRT":
        # NRT: diretório único por mês (YYYYMM) e listagem em texto puro ("/text/").
        return f"{BASE_PPS}/text/imerg/early/{t:%Y%m}/"
    return f"{BASE_GESDISC}/{t:%Y}/{t.timetuple().tm_yday:03d}/"


def listar_diretorio(sessao: requests.Session, t: dt.datetime,
                     fonte: str) -> list[tuple[str, str]]:
    """Lista o diretório do dia como pares (nome_do_arquivo, url_completa)."""
    url = url_diretorio(t, fonte)
    if url in _CACHE_LISTAGEM:
        return _CACHE_LISTAGEM[url]

    r = sessao.get(url, timeout=120)
    r.raise_for_status()

    if fonte == "PPS_NRT":
        # resposta em texto: um caminho absoluto por linha, p.ex.
        # /imerg/early/202609/3B-HHR-E.MS.MRG.3IMERG.20260920-S143000-...V07B.RT-H5
        itens = [ln.strip() for ln in r.text.split() if "3IMERG" in ln]
        pares = [(i.split("/")[-1], f"{BASE_PPS}/text{i}"
                  if i.startswith("/") else url + i) for i in itens]
    else:
        itens = re.findall(r'href="([^"]*3B-HHR-E[^"]*)"', r.text)
        pares = [(i.split("/")[-1], i if i.startswith("http") else url + i.split("/")[-1])
                 for i in itens]

    # NRT usa .RT-H5; o arquivo do GES DISC usa .HDF5. Ambos são HDF5 por dentro.
    pares = [(n, u) for n, u in pares if n.endswith((".RT-H5", ".HDF5"))]
    _CACHE_LISTAGEM[url] = pares
    return pares


def baixar_frame(sessao: requests.Session, t: dt.datetime,
                 fonte: str, dir_dados: Path) -> Path:
    """Baixa (ou reaproveita do cache local) o granulo IMERG do slot t."""
    carimbo = f"{t:%Y%m%d}-S{t:%H%M%S}"
    locais = [p for p in sorted(dir_dados.glob(f"*{carimbo}*"))
              if p.suffix in (".RT-H5", ".HDF5")]
    if locais:
        return locais[0]

    alvo = [(n, u) for n, u in listar_diretorio(sessao, t, fonte) if carimbo in n]
    if not alvo:
        raise FileNotFoundError(
            f"granulo {carimbo} ainda não disponível em {url_diretorio(t, fonte)}")

    nome, url = alvo[0]
    destino = dir_dados / nome
    tmp = destino.with_suffix(destino.suffix + ".part")
    with sessao.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(destino)
    return destino


# --------------------------------------------------------------------------- #
# 3. Montagem do cubo espaço-temporal
# --------------------------------------------------------------------------- #
# No IMERG V07 o HDF5 traz os dados no grupo /Grid, com 'precipitation' em
# (time, lon, lat) e _FillValue = -9999.9.
def ler_frame(caminho: Path) -> xr.DataArray:
    ds = xr.open_dataset(caminho, group="Grid", engine="h5netcdf", decode_times=False)
    nome = "precipitation" if "precipitation" in ds else "precipitationCal"
    da = ds[nome].isel(time=0)
    if "lat" in da.dims and "lon" in da.dims:
        da = da.transpose("lat", "lon")
    da = da.sel(lat=slice(*DOM_LAT), lon=slice(*DOM_LON))
    return da.where(da >= 0).load()          # descarta -9999.9


def montar_cubo(arquivos: list[tuple[dt.datetime, Path]], fonte: str) -> xr.DataArray:
    campos = []
    for t, p in arquivos:
        da = ler_frame(p)
        campos.append(da.assign_coords(time=np.datetime64(t.replace(tzinfo=None), "ns")))

    cubo = xr.concat(campos, dim="time").sortby("time")
    cubo.name = "precipitacao"
    cubo.attrs.update(units="mm/h", fonte=f"GPM IMERG Early Run ({fonte})")

    log(f"cubo: {dict(cubo.sizes)} | resolução "
        f"{float(abs(cubo.lat.diff('lat').mean())):.2f}° x "
        f"{float(abs(cubo.lon.diff('lon').mean())):.2f}°")
    log(f"taxa máxima na janela: {float(cubo.max()):.1f} mm/h | "
        f"cobertura com chuva: {float((cubo > 0.2).mean()) * 100:.1f}%")
    return cubo


# --------------------------------------------------------------------------- #
# 4. Campo de movimento
# --------------------------------------------------------------------------- #
# Duas implementações, na ordem de preferência:
#   1. pysteps (se instalado) — Lucas-Kanade denso, padrão consolidado da área.
#   2. Fallback próprio — correlação de fase por blocos sobrepostos de 6,4°,
#      com refinamento subpixel por parábola, filtro de mediana 3x3 nos vetores
#      e interpolação para a grade cheia.
#
# Um vetor único para todo o Brasil seria fisicamente errado — Belém e Canoas
# não compartilham escoamento —, daí a estimativa por blocos.
#
# ONDE O FALLBACK FALHA: quando duas estruturas com movimentos distintos se
# sobrepõem no mesmo bloco (convecção embebida num campo estratiforme com outra
# direção), o pico da correlação pode travar em zero. Em campo sintético com
# translação coerente o erro fica abaixo de 0,01 px, mas colapsa para ~0 nesse
# caso. É a razão para preferir o pysteps e para rodar a verificação.
def _correlacao_fase(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Deslocamento (dy, dx) em pixels que leva o bloco 'a' em 'b', com subpixel."""
    a = a - a.mean()
    b = b - b.mean()
    janela = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    R = np.fft.fft2(a * janela).conj() * np.fft.fft2(b * janela)
    mag = np.abs(R)
    mag[mag == 0] = 1e-12
    r = np.fft.ifft2(R / mag).real

    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    pico = r[iy, ix]
    dy = iy - a.shape[0] if iy > a.shape[0] // 2 else iy
    dx = ix - a.shape[1] if ix > a.shape[1] // 2 else ix

    def parabola(c, m, p):
        d = m - 2 * c + p
        return 0.0 if d == 0 else 0.5 * (m - p) / d

    dy += parabola(pico, r[(iy - 1) % r.shape[0], ix], r[(iy + 1) % r.shape[0], ix])
    dx += parabola(pico, r[iy, (ix - 1) % r.shape[1]], r[iy, (ix + 1) % r.shape[1]])
    return float(dy), float(dx), float(pico)


def _preparar(campo: np.ndarray, limiar: float = 0.2, sigma: float = 1.0) -> np.ndarray:
    """log-transforma, corta chuva fraca e suaviza antes de correlacionar."""
    c = np.nan_to_num(campo)
    return gaussian_filter(np.where(c > limiar, np.log1p(c), 0.0), sigma)


def campo_movimento_fallback(f0, f1, bloco=64, passo=16, frac_min=0.03,
                             desloc_max=30, pico_min=0.02):
    ny, nx = f0.shape
    a, b = _preparar(f0), _preparar(f1)
    ys = np.arange(0, ny - bloco + 1, passo)
    xs = np.arange(0, nx - bloco + 1, passo)
    VY = np.full((len(ys), len(xs)), np.nan)
    VX = VY.copy()

    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            ba = a[y0:y0 + bloco, x0:x0 + bloco]
            bb = b[y0:y0 + bloco, x0:x0 + bloco]
            if (ba > 0).mean() < frac_min or (bb > 0).mean() < frac_min:
                continue
            dy, dx, pico = _correlacao_fase(ba, bb)
            if pico < pico_min or abs(dy) > desloc_max or abs(dx) > desloc_max:
                continue
            VY[i, j], VX[i, j] = dy, dx

    n_validos = int(np.isfinite(VY).sum())
    if n_validos < 3:
        return np.zeros((ny, nx)), np.zeros((ny, nx)), 0

    def nanmed(v):
        return np.nan if np.all(np.isnan(v)) else np.nanmedian(v)

    VY = generic_filter(VY, nanmed, size=3, mode="nearest")
    VX = generic_filter(VX, nanmed, size=3, mode="nearest")

    cy, cx = ys + bloco / 2.0, xs + bloco / 2.0
    gy, gx = np.mgrid[0:ny, 0:nx].astype(float)
    saida = []
    for V in (VY, VX):
        preenche = np.nanmean(V) if np.isfinite(V).any() else 0.0
        Vf = np.where(np.isnan(V), preenche, V)
        interp = RegularGridInterpolator((cy, cx), Vf, bounds_error=False, fill_value=None)
        saida.append(interp(np.stack([gy.ravel(), gx.ravel()], -1)).reshape(ny, nx))
    return saida[0], saida[1], n_validos


def estimar_movimento(cubo: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (vy, vx) em pixels por passo de 30 min."""
    arr = np.nan_to_num(cubo.values)
    if TEM_PYSTEPS:
        campo = np.where(arr > 0.2, 10 * np.log10(np.maximum(arr, 0.2)), -15.0)
        V = pysteps_motion.get_method("lucaskanade")(campo)
        log("movimento: pysteps / Lucas-Kanade")
        return V[1], V[0]                 # pysteps devolve (vx, vy) -> (linha, coluna)

    vy, vx, n = campo_movimento_fallback(arr[-2], arr[-1])
    log(f"movimento: correlação de fase ({n} blocos válidos)")
    return vy, vx


# --------------------------------------------------------------------------- #
# 5. Extrapolação semi-lagrangiana
# --------------------------------------------------------------------------- #
# Para cada prazo o campo é reamostrado nas posições de origem
# (y - vy*dt, x - vx*dt) por interpolação bilinear — advecção retroativa, com o
# campo de movimento congelado no tempo.
def advectar(campo: np.ndarray, vy: np.ndarray, vx: np.ndarray, passos: int = 1):
    ny, nx = campo.shape
    gy, gx = np.mgrid[0:ny, 0:nx].astype(float)
    return map_coordinates(np.nan_to_num(campo),
                           [gy - vy * passos, gx - vx * passos],
                           order=1, mode="constant", cval=0.0)


def gerar_nowcast(cubo, vy, vx, leads) -> xr.DataArray:
    base = cubo.isel(time=-1)
    t0 = pd.to_datetime(base.time.values)
    # o passo do nowcast é o espaçamento real da base (30 min no IMERG, o que
    # a cadência escolhida der no GOES) — não um valor fixo.
    if cubo.sizes["time"] >= 2:
        passo = pd.to_datetime(cubo.time.values[-1]) - pd.to_datetime(cubo.time.values[-2])
    else:
        passo = pd.Timedelta(minutes=30)
    campos, tempos = [], []
    for k in leads:
        if TEM_PYSTEPS:
            prev = pysteps_nowcasts.get_method("extrapolation")(
                np.nan_to_num(base.values), np.stack([vx, vy]), k)[-1]
        else:
            prev = advectar(base.values, vy, vx, k)
        campos.append(prev)
        tempos.append(t0 + passo * k)

    return xr.DataArray(
        np.stack(campos), dims=("time", "lat", "lon"),
        coords={"time": tempos, "lat": base.lat, "lon": base.lon},
        name="precipitacao_prevista",
        attrs={"units": "mm/h", "metodo": "extrapolação lagrangiana IMERG"})


# --------------------------------------------------------------------------- #
# 7. Extração por unidade
# --------------------------------------------------------------------------- #
# Para cada unidade o valor é a estatística (max por padrão) das células da
# grade dentro de RAIO_KM. O máximo num raio é deliberadamente conservador:
# numa grade de 0,1° (~11 km) o pixel exato da unidade pode perder uma célula
# convectiva que está passando ao lado. Distâncias por Haversine.
def indices_no_raio(cubo, lat0, lon0, raio_km) -> np.ndarray:
    LA, LO = np.meshgrid(cubo.lat.values, cubo.lon.values, indexing="ij")
    R = 6371.0
    p1, p2 = np.deg2rad(lat0), np.deg2rad(LA)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(LO - lon0) / 2) ** 2)
    dist = 2 * R * np.arcsin(np.sqrt(a))
    m = dist <= raio_km
    return m if m.any() else (dist == dist.min())      # raio < célula: pixel mais próximo


def classificar(v) -> str:
    if not np.isfinite(v):
        return "Sem dado"
    for lim, rot in LIMIARES:
        if v < lim:
            return rot
    return LIMIARES[-1][1]


# --------------------------------------------------------------------------- #
# O Brasil não adota mais horário de verão desde 2019, então o deslocamento é
# fixo. Todas as 40 unidades estão em UTC-3.
TZ_LOCAL = dt.timedelta(hours=-3)
ROTULO_TZ = "UTC-3"


# --------------------------------------------------------------------------- #
# Lógica de dados — sem Tkinter, testável isoladamente
# --------------------------------------------------------------------------- #
def ultimo_disponivel(sessao, agora_utc: dt.datetime,
                      fonte: str = "PPS_NRT") -> dt.datetime | None:
    """Carimbo do granulo mais recente que já está no servidor.

    Melhor do que descontar uma latência fixa: a do Early Run varia, e um valor
    fixo ou descarta granulo que já existe, ou pede um que ainda não subiu.
    """
    padrao = re.compile(r"3IMERG\.(\d{8})-S(\d{6})")
    candidatos = []
    # o dia local cruza dois diretórios mensais na virada do mês
    for quando in (agora_utc, agora_utc - dt.timedelta(days=1)):
        try:
            for nome, _ in listar_diretorio(sessao, quando, fonte):
                m = padrao.search(nome)
                if m:
                    t = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
                    if t <= agora_utc:
                        candidatos.append(t)
        except Exception:
            continue
        if candidatos:
            break
    return max(candidatos) if candidatos else None


def slots_do_dia(agora_utc: dt.datetime, latencia_h: float,
                 desde_meia_noite: bool = True,
                 horas: float = 6.0,
                 limite: dt.datetime | None = None) -> list[dt.datetime]:
    """Carimbos de 30 min (em UTC) do dia LOCAL corrente, dada a latência.

    O dia é ancorado na data local, não na UTC: meia-noite local são 03:00 UTC,
    e meia-noite UTC seriam 21:00 do dia anterior aqui. Ancorar em UTC faria a
    série começar no dia errado para quem lê o gráfico em horário local.

    desde_meia_noite=True  -> de 00:00 local de hoje até agora-latência
    desde_meia_noite=False -> apenas as últimas 'horas' antes desse limite
    """
    # 'limite' vindo da listagem do servidor é o carimbo real mais recente;
    # sem ele, cai na latência informada.
    if limite is None:
        limite = agora_utc - dt.timedelta(hours=latencia_h)
    limite = limite.replace(minute=(limite.minute // 30) * 30,
                            second=0, microsecond=0)

    # meia-noite local de hoje, expressa em UTC
    meia_noite_local = (agora_utc + TZ_LOCAL).replace(
        hour=0, minute=0, second=0, microsecond=0)
    inicio_dia = meia_noite_local - TZ_LOCAL

    if limite < inicio_dia:
        # Nas primeiras horas do dia local, tudo ainda está dentro da latência.
        return []

    inicio = inicio_dia if desde_meia_noite else max(
        inicio_dia, limite - dt.timedelta(hours=horas))

    n = int((limite - inicio).total_seconds() // 1800) + 1
    return [inicio + dt.timedelta(minutes=30 * i) for i in range(n)]


def serie_no_ponto(da, mascara: np.ndarray, estatistica: str = "max") -> pd.DataFrame:
    """Série temporal de um DataArray (time, lat, lon) sob uma máscara."""
    reg = []
    for i in range(da.sizes["time"]):
        v = da.isel(time=i).values[mascara]
        v = v[np.isfinite(v)]
        val = np.nan if v.size == 0 else (v.max() if estatistica == "max" else v.mean())
        reg.append({"tempo_utc": pd.to_datetime(da.time.values[i]),
                    "mm_h": float(val)})
    df = pd.DataFrame(reg)
    df["tempo_local"] = df["tempo_utc"] + TZ_LOCAL
    return df


def resumir(df_obs: pd.DataFrame) -> dict:
    """Resumo do dia. Cada granulo cobre 30 min, daí o fator 0,5 h."""
    mm = df_obs["mm_h"].to_numpy()
    mm = mm[np.isfinite(mm)]
    if mm.size == 0:
        return {"acumulado_mm": np.nan, "pico_mm_h": np.nan,
                "horarios": 0, "classe_pico": "Sem dado"}
    pico = float(mm.max())
    return {"acumulado_mm": float(mm.sum() * 0.5),
            "pico_mm_h": pico,
            "horarios": int(mm.size),
            "classe_pico": classificar(pico)}


# --------------------------------------------------------------------------- #
# GOES-19 / ABI — taxa de chuva (RRQPE) em tempo quase real
# --------------------------------------------------------------------------- #
# Produto ABI-L2-RRQPEF: Rainfall Rate / QPE, disco completo, a cada 10 min, em
# mm/h — mesma unidade do IMERG, por isso entra no mesmo cubo. Vem do bucket
# público da NOAA na AWS, sem credencial.
#
# Serve para fechar o vão do Early Run: enquanto o IMERG mais recente tem ~5 h,
# o GOES tem minutos. O nowcast passa a partir do quadro mais novo disponível.
BASE_GOES = "https://noaa-goes19.s3.amazonaws.com"
PRODUTO_GOES = "ABI-L2-RRQPEF"

# Constantes da projeção geoestacionária do ABI (GOES-East).
GOES_REQ = 6378137.0          # semieixo maior (m)
GOES_RPOL = 6356752.31414     # semieixo menor (m)
GOES_H = 42164160.0           # altura de perspectiva + semieixo maior (m)
GOES_LON0 = -75.0             # longitude subsatélite


def latlon_para_xy(lat, lon, lon0=GOES_LON0, H=GOES_H,
                   req=GOES_REQ, rpol=GOES_RPOL):
    """(lat, lon) em graus -> ângulos de varredura (x, y) em rad.

    NaN onde o ponto está atrás do limbo. Fórmulas do PUG vol.3, §5.1.2.8;
    validadas por ida e volta contra a inversa oficial (erro < 1e-9 m).
    """
    lat = np.deg2rad(np.asarray(lat, float))
    lon = np.deg2rad(np.asarray(lon, float))
    lam0 = np.deg2rad(lon0)
    e2 = (req ** 2 - rpol ** 2) / req ** 2

    phi_c = np.arctan((rpol ** 2 / req ** 2) * np.tan(lat))
    rc = rpol / np.sqrt(1.0 - e2 * np.cos(phi_c) ** 2)

    sx = H - rc * np.cos(phi_c) * np.cos(lon - lam0)
    sy = -rc * np.cos(phi_c) * np.sin(lon - lam0)
    sz = rc * np.sin(phi_c)

    visivel = H * (H - sx) > sy ** 2 + (req ** 2 / rpol ** 2) * sz ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.arctan(sz / sx)
        x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    return np.where(visivel, x, np.nan), np.where(visivel, y, np.nan)


def sessao_goes() -> requests.Session:
    """Sessão SEM autenticação: o bucket da NOAA é público e as credenciais
    do PPS não devem vazar para ele."""
    s = requests.Session()
    s.headers.update({"User-Agent": "imerg-nowcast-petrobras/1.0"})
    return s


def listar_goes(sessao, hora: dt.datetime) -> list[tuple[dt.datetime, str]]:
    """Chaves do RRQPE na hora indicada, como (instante inicial, chave S3)."""
    prefixo = f"{PRODUTO_GOES}/{hora:%Y}/{hora.timetuple().tm_yday:03d}/{hora:%H}/"
    r = sessao.get(BASE_GOES, params={"list-type": "2", "prefix": prefixo},
                   timeout=90)
    r.raise_for_status()
    saida = []
    for chave in re.findall(r"<Key>([^<]+)</Key>", r.text):
        m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", chave)
        if not m:
            continue
        ano, doy, hh, mm, ss = (int(g) for g in m.groups())
        t = (dt.datetime(ano, 1, 1) +
             dt.timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss))
        saida.append((t, chave))
    return sorted(saida)


def granulos_goes(sessao, de: dt.datetime, ate: dt.datetime,
                  cadencia_min: int = 30) -> list[tuple[dt.datetime, str]]:
    """Um granulo a cada 'cadencia_min' no intervalo, o mais próximo do alvo."""
    todos = []
    hora = de.replace(minute=0, second=0, microsecond=0)
    while hora <= ate:
        try:
            todos += listar_goes(sessao, hora)
        except Exception:
            pass
        hora += dt.timedelta(hours=1)
    todos = [(t, k) for t, k in todos if de <= t <= ate]
    if not todos:
        return []

    escolhidos, alvo = [], de
    while alvo <= ate:
        cand = min(todos, key=lambda tk: abs((tk[0] - alvo).total_seconds()))
        if abs((cand[0] - alvo).total_seconds()) <= cadencia_min * 60:
            escolhidos.append(cand)
        alvo += dt.timedelta(minutes=cadencia_min)
    # remove repetições preservando a ordem
    vistos, unicos = set(), []
    for t, k in escolhidos:
        if t not in vistos:
            vistos.add(t); unicos.append((t, k))
    return unicos


def baixar_goes(sessao, chave: str, dir_dados: Path) -> Path:
    destino = dir_dados / chave.split("/")[-1]
    if destino.exists():
        return destino
    tmp = destino.with_suffix(destino.suffix + ".part")
    with sessao.get(f"{BASE_GOES}/{chave}", stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(destino)
    return destino


def ler_goes(caminho: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Lê o RRQPE e reamostra na grade (lats, lons) do IMERG.

    O ABI tem 2 km e a grade alvo tem 0,1° (~11 km). Pegar o pixel mais próximo
    aliasaria: uma célula de 11 km viraria um ponto de 2 km escolhido ao acaso.
    Por isso aplica-se uma média de bloco antes de amostrar, aproximando a
    área que o IMERG integra.
    """
    ds = xr.open_dataset(caminho, engine="h5netcdf", decode_times=False)
    campo = ds["RRQPE"]

    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    X, Y = latlon_para_xy(LA, LO)

    xs = ds["x"].values
    ys = ds["y"].values
    # recorta o disco na região de interesse antes de filtrar (o disco inteiro
    # tem ~29 milhões de pixels)
    margem = 40
    ix = np.searchsorted(xs, [np.nanmin(X), np.nanmax(X)])
    iy = np.searchsorted(-ys, [-np.nanmax(Y), -np.nanmin(Y)])
    i0, i1 = max(int(ix[0]) - margem, 0), min(int(ix[1]) + margem, len(xs))
    j0, j1 = max(int(iy[0]) - margem, 0), min(int(iy[1]) + margem, len(ys))

    bloco = campo.values[j0:j1, i0:i1].astype("float32")
    bloco = np.where(np.isfinite(bloco) & (bloco >= 0), bloco, np.nan)

    # média de bloco ~ resolução do IMERG (0,1° ≈ 5 pixels de 2 km)
    passo_x = float(abs(np.diff(xs[:2])[0]))
    n_suav = max(int(round(np.deg2rad(0.1) / passo_x)), 1)
    if n_suav > 1:
        preenchido = np.nan_to_num(bloco)
        valido = np.isfinite(bloco).astype("float32")
        soma = uniform_filter(preenchido, n_suav, mode="nearest")
        peso = uniform_filter(valido, n_suav, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            bloco = np.where(peso > 0.2, soma / peso, np.nan)

    jj = np.searchsorted(-ys[j0:j1], -Y) 
    ii = np.searchsorted(xs[i0:i1], X)
    jj = np.clip(jj, 0, bloco.shape[0] - 1)
    ii = np.clip(ii, 0, bloco.shape[1] - 1)
    saida = bloco[jj, ii]
    saida[~np.isfinite(X)] = np.nan
    ds.close()
    return saida


def cubo_goes(sessao, granulos, lats, lons, dir_dados,
              progresso=None, cancelar=None):
    """Monta um cubo (time, lat, lon) de RRQPE na grade do IMERG."""
    campos, tempos, falhas = [], [], []
    for i, (t, chave) in enumerate(granulos, 1):
        if cancelar is not None and cancelar.is_set():
            raise KeyboardInterrupt("cancelado pelo usuário")
        if progresso:
            progresso(i - 1, len(granulos), f"GOES {t:%H:%M}Z")
        try:
            p = baixar_goes(sessao, chave, dir_dados)
            campos.append(ler_goes(p, lats, lons))
            tempos.append(np.datetime64(t.replace(second=0, microsecond=0), "ns"))
        except Exception as e:
            falhas.append((t, f"{type(e).__name__}: {e}"))
    if not campos:
        return None, falhas
    da = xr.DataArray(np.stack(campos), dims=("time", "lat", "lon"),
                      coords={"time": tempos, "lat": lats, "lon": lons},
                      name="precipitacao_goes",
                      attrs={"units": "mm/h", "fonte": "GOES-19 ABI-L2-RRQPEF"})
    return da.sortby("time"), falhas


def previsao_modelo(lat: float, lon: float, horas: int = 24) -> pd.DataFrame:
    """Previsão horária de precipitação por modelo numérico (Open-Meteo).

    NÃO é IMERG e NÃO é extrapolação: vem de modelo de previsão numérica do
    tempo. É a única das três fontes que tem sentido físico em horizonte de
    horas a dias. Em troca, é previsão PONTUAL — não existe campo espacial
    aqui, por isso ela aparece na tabela e não no mapa.
    """
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon,
                "hourly": "precipitation,precipitation_probability",
                "timezone": "auto", "forecast_days": 3},
        timeout=60)
    r.raise_for_status()
    dados = r.json()
    if "hourly" not in dados:
        raise RuntimeError(f"resposta sem bloco 'hourly': {str(dados)[:200]}")

    h = dados["hourly"]
    df = pd.DataFrame({
        "tempo_local": pd.to_datetime(h["time"]),
        "mm_h": pd.to_numeric(h.get("precipitation"), errors="coerce"),
        "prob": pd.to_numeric(h.get("precipitation_probability"), errors="coerce"),
    })
    # A API devolve a partir de 00:00 do dia corrente; corta da hora atual.
    agora = pd.Timestamp.now().floor("h")
    df = df[(df.tempo_local >= agora) &
            (df.tempo_local <= agora + pd.Timedelta(hours=horas))]
    df = df.reset_index(drop=True)
    df["tempo_utc"] = df["tempo_local"] - TZ_LOCAL
    return df


def coletar_dia(unidade: str, lat: float, lon: float, *, latencia_h: float,
                desde_meia_noite: bool, horas: float, raio_km: float,
                estatistica: str, com_nowcast: bool, horizonte_h: float = 2.0,
                com_modelo: bool = False, horas_modelo: int = 24,
                com_goes: bool = False, cadencia_goes: int = 30,
                progresso=None, cancelar: threading.Event | None = None,
                email_pps: str | None = None):
    """Baixa os granulos do dia, extrai a série no ponto e devolve tudo.

    progresso(feito, total, texto) é chamado a cada granulo, para a barra.
    """
    sessao = abrir_sessao("PPS_NRT", email_pps)
    agora = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    recente = ultimo_disponivel(sessao, agora)
    if recente is not None:
        log(f"granulo mais recente no servidor: {recente:%d/%m %H:%M}Z "
            f"(latência real {(agora - recente).total_seconds() / 3600:.1f} h)")
    else:
        log(f"listagem indisponível — usando latência fixa de {latencia_h:.0f} h")
    slots = slots_do_dia(agora, latencia_h, desde_meia_noite, horas, limite=recente)
    if not slots:
        raise RuntimeError(
            "Nenhum granulo de hoje disponível ainda — o dia corrente ainda está "
            f"inteiro dentro da latência de {latencia_h:.0f} h.")

    dir_dados = Path("./dados_imerg")
    dir_dados.mkdir(exist_ok=True)

    arquivos, falhas = [], []
    for i, t in enumerate(slots, 1):
        if cancelar is not None and cancelar.is_set():
            raise KeyboardInterrupt("cancelado pelo usuário")
        if progresso:
            progresso(i - 1, len(slots), f"{t:%H:%M}Z")
        try:
            arquivos.append((t, baixar_frame(sessao, t, "PPS_NRT", dir_dados)))
        except Exception as e:
            falhas.append((t, f"{type(e).__name__}: {e}"))
    if progresso:
        progresso(len(slots), len(slots), "montando o cubo")

    if len(arquivos) < 1:
        raise RuntimeError("nenhum granulo baixado — veja as falhas no log")

    cubo = montar_cubo(arquivos, "PPS_NRT")
    mascara = indices_no_raio(cubo, lat, lon, raio_km)
    obs = serie_no_ponto(cubo, mascara, estatistica)

    # ---- GOES-19: fecha o vão entre o último IMERG e agora ----
    goes, serie_goes, falhas_goes = None, None, []
    if com_goes:
        t_ult = pd.to_datetime(cubo.time.values[-1]).to_pydatetime()
        sg = sessao_goes()
        try:
            gran = granulos_goes(sg, t_ult + dt.timedelta(minutes=cadencia_goes),
                                 agora, cadencia_goes)
            log(f"GOES-19: {len(gran)} granulos entre {t_ult:%H:%M}Z e {agora:%H:%M}Z")
            if gran:
                goes, falhas_goes = cubo_goes(
                    sg, gran, cubo.lat.values, cubo.lon.values, dir_dados,
                    progresso, cancelar)
                if goes is not None:
                    serie_goes = serie_no_ponto(goes, mascara, estatistica)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            falhas_goes.append((agora, f"{type(e).__name__}: {e}"))

    # ---- nowcast: parte do quadro mais recente que existir ----
    prev = None
    nc = None
    base_nowcast = None
    if goes is not None and goes.sizes["time"] >= 2:
        base_nowcast, rotulo_base = goes, "GOES-19"
    elif cubo.sizes["time"] >= 2:
        base_nowcast, rotulo_base = cubo, "IMERG"

    if com_nowcast and base_nowcast is not None:
        log(f"nowcast a partir de {rotulo_base} "
            f"({pd.to_datetime(base_nowcast.time.values[-1]):%H:%M}Z)")
        vy, vx = estimar_movimento(base_nowcast)
        leads = list(range(1, max(int(round(horizonte_h * 2)), 1) + 1))
        nc = gerar_nowcast(base_nowcast, vy, vx, leads)
        prev = serie_no_ponto(nc, mascara, estatistica)

    modelo, erro_modelo = None, None
    if com_modelo:
        try:
            modelo = previsao_modelo(lat, lon, horas_modelo)
        except Exception as e:
            erro_modelo = f"{type(e).__name__}: {e}"

    return {"unidade": unidade, "lat": lat, "lon": lon, "obs": obs, "prev": prev,
            "nowcast_cubo": nc, "modelo": modelo, "erro_modelo": erro_modelo,
            "goes": goes, "serie_goes": serie_goes, "falhas_goes": falhas_goes,
            "falhas": falhas, "resumo": resumir(obs), "cubo": cubo,
            "raio_km": raio_km, "estatistica": estatistica,
            "n_celulas": int(mascara.sum())}


# --------------------------------------------------------------------------- #


# ============================================================================
# INTERFACE WEB — STREAMLIT
# ============================================================================

import io
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS leve e responsivo.
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 14px;
        padding: 12px 14px;
        background: rgba(128,128,128,.035);
    }
    .source-note {
        font-size: .82rem;
        opacity: .75;
        margin-top: -.35rem;
        margin-bottom: .8rem;
    }
    .status-ok {
        border-left: 4px solid #2e7d32;
        padding: .55rem .8rem;
        background: rgba(46,125,50,.07);
        border-radius: 8px;
    }
    .status-warn {
        border-left: 4px solid #ef6c00;
        padding: .55rem .8rem;
        background: rgba(239,108,0,.07);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def _unidades():
    def chave(u):
        return unicodedata.normalize("NFKD", u[0]).encode(
            "ascii", "ignore"
        ).decode().lower()
    return sorted(UNIDADES_PADRAO, key=chave)

def _local_str(ts):
    return pd.to_datetime(ts).strftime("%d/%m %H:%M")

def _build_frames(r):
    quadros = []
    cubo = r["cubo"]

    for i in range(cubo.sizes["time"]):
        t = pd.to_datetime(cubo.time.values[i])
        quadros.append(
            {"campo": cubo.isel(time=i), "tempo_utc": t, "tipo": "observado"}
        )

    g = r.get("goes")
    if g is not None:
        for i in range(g.sizes["time"]):
            t = pd.to_datetime(g.time.values[i])
            quadros.append(
                {
                    "campo": g.isel(time=i),
                    "tempo_utc": t,
                    "tipo": "observado GOES",
                }
            )

    nc = r.get("nowcast_cubo")
    if nc is not None:
        for i in range(nc.sizes["time"]):
            t = pd.to_datetime(nc.time.values[i])
            quadros.append(
                {"campo": nc.isel(time=i), "tempo_utc": t, "tipo": "previsto"}
            )
    return quadros

def _map_fig(r, q, zoom_deg):
    campo = q["campo"]
    lat0, lon0 = float(r["lat"]), float(r["lon"])

    recorte = campo.sel(
        lat=slice(lat0 - zoom_deg, lat0 + zoom_deg),
        lon=slice(lon0 - zoom_deg, lon0 + zoom_deg),
    )

    z = np.asarray(recorte.values, dtype=float)
    z = np.where(np.isfinite(z), z, np.nan)
    lats = np.asarray(recorte.lat.values, dtype=float)
    lons = np.asarray(recorte.lon.values, dtype=float)

    if not TEM_CARTOPY:
        raise RuntimeError(
            "Cartopy não está instalado. Adicione cartopy ao requirements.txt "
            "e reinicie o aplicativo."
        )

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12.8, 7.0), dpi=130, facecolor="white")
    ax = fig.add_axes([0.055, 0.075, 0.79, 0.82], projection=proj)
    ax.set_facecolor("white")
    ax.set_extent(
        [lon0 - zoom_deg, lon0 + zoom_deg, lat0 - zoom_deg, lat0 + zoom_deg],
        crs=proj,
    )

    # Base cartográfica: limpa, branca e com referências geográficas sutis.
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#fafafa", edgecolor="none", zorder=0.1)
    ax.add_feature(
        cfeature.LAKES, facecolor="white", edgecolor="#b8b8b8", linewidth=0.35, zorder=0.2
    )
    ax.add_feature(
        cfeature.BORDERS, edgecolor="#8d8d8d", linewidth=0.65, zorder=4
    )
    ax.add_feature(
        cfeature.STATES, edgecolor="#b7b7b7", linewidth=0.45, zorder=4
    )
    ax.coastlines(resolution="50m", color="#444444", linewidth=0.8, zorder=4.2)

    # Campo de precipitação.
    pcm = ax.pcolormesh(
        lons,
        lats,
        z,
        transform=proj,
        cmap="turbo",
        vmin=0,
        vmax=25,
        shading="auto",
        alpha=0.92,
        zorder=2,
    )

    # Unidade Petrobras.
    ax.scatter(
        lon0,
        lat0,
        transform=proj,
        marker="^",
        s=85,
        facecolor="white",
        edgecolor="black",
        linewidth=1.45,
        zorder=7,
    )

    # Raio de influência.
    ang = np.linspace(0, 2 * np.pi, 240)
    dlat = float(r["raio_km"]) / 111.0
    dlon = float(r["raio_km"]) / (111.0 * max(np.cos(np.deg2rad(lat0)), 1e-6))
    ax.plot(
        lon0 + dlon * np.cos(ang),
        lat0 + dlat * np.sin(ang),
        transform=proj,
        color="black",
        linewidth=1.5,
        linestyle="-",
        alpha=0.9,
        zorder=6,
    )

    # Grade geográfica com rótulos limpos.
    gl = ax.gridlines(
        crs=proj,
        draw_labels=True,
        linewidth=0.45,
        color="#9d9d9d",
        alpha=0.65,
        linestyle="--",
        x_inline=False,
        y_inline=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9, "color": "#3a3a3a"}
    gl.ylabel_style = {"size": 9, "color": "#3a3a3a"}
    gl.xformatter = LongitudeFormatter(number_format=".0f", degree_symbol="°")
    gl.yformatter = LatitudeFormatter(number_format=".0f", degree_symbol="°")

    # Título e identificação.
    local = q["tempo_utc"] + TZ_LOCAL
    tipo = q["tipo"].upper()
    titulo = f"{r['unidade']}"
    subtitulo = f"{local:%d/%m/%Y %H:%M} local  •  {tipo}"
    fig.text(0.45, 0.962, titulo, ha="center", va="top", fontsize=15.5, fontweight="bold", color="#1f1f1f")
    fig.text(0.45, 0.932, subtitulo, ha="center", va="top", fontsize=9.5, color="#5b5b5b")

    # Identificação do ponto e do raio.
    ax.text(
        0.015,
        0.025,
        f"Unidade  •  {lat0:.4f}, {lon0:.4f}\nRaio de análise: {r['raio_km']:.0f} km",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#333333",
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#bdbdbd", "alpha": 0.92},
        zorder=8,
    )

    cax = fig.add_axes([0.865, 0.16, 0.022, 0.66])
    cbar = fig.colorbar(pcm, cax=cax, extend="max")
    cbar.set_label("Precipitação (mm/h)", fontsize=9.5, color="#2f2f2f")
    cbar.ax.tick_params(labelsize=8.5, colors="#3f3f3f", length=3)
    cbar.outline.set_edgecolor("#999999")
    cbar.outline.set_linewidth(0.6)

    # Moldura discreta para dar acabamento ao mapa.
    for spine in ax.spines.values():
        spine.set_edgecolor("#666666")
        spine.set_linewidth(0.8)

    return fig

def _table_df(r):
    blocos = []

    obs = r.get("obs")
    if obs is not None and len(obs):
        d = obs.copy()
        d["tipo"] = "observado (IMERG)"

        blocos.append(d)

    goes = r.get("serie_goes")
    if goes is not None and len(goes):
        d = goes.copy()
        d["tipo"] = "observado (GOES-19)"
        blocos.append(d)

    prev = r.get("prev")
    if prev is not None and len(prev):
        d = prev.copy()
        d["tipo"] = "nowcast (extrapolação)"
        blocos.append(d)

    modelo = r.get("modelo")
    if modelo is not None and len(modelo):
        d = modelo.copy()
        d["tipo"] = "previsão (modelo)"
        d["probabilidade_%"] = d.get("prob", np.nan)
        blocos.append(d)

    if not blocos:
        return pd.DataFrame()

    cols = [
        "tempo_local",
        "tempo_utc",
        "mm_h",
        "tipo",
        "probabilidade_%",
    ]
    out = pd.concat(
        [b.reindex(columns=cols) for b in blocos],
        ignore_index=True,
    )
    out["classe"] = out["mm_h"].apply(classificar)
    out["tempo_local"] = pd.to_datetime(out["tempo_local"]).dt.strftime("%d/%m %H:%M")
    out["tempo_utc"] = pd.to_datetime(out["tempo_utc"]).dt.strftime("%d/%m %H:%M")
    out["mm_h"] = pd.to_numeric(out["mm_h"], errors="coerce").round(2)
    out["probabilidade_%"] = pd.to_numeric(
        out["probabilidade_%"], errors="coerce"
    ).round(0)
    return out

def _xlsx_bytes(r):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        r["obs"].to_excel(w, sheet_name="observado", index=False)
        if r.get("prev") is not None:
            r["prev"].to_excel(w, sheet_name="nowcast", index=False)
        if r.get("serie_goes") is not None:
            r["serie_goes"].to_excel(
                w, sheet_name="observado_goes", index=False
            )
        if r.get("modelo") is not None:
            r["modelo"].to_excel(
                w, sheet_name="previsao_modelo", index=False
            )

        resumo = {
            "unidade": r["unidade"],
            "lat": r["lat"],
            "lon": r["lon"],
            "raio_km": r["raio_km"],
            "estatistica": r["estatistica"],
            **r["resumo"],
        }
        pd.DataFrame([resumo]).to_excel(w, sheet_name="resumo", index=False)
    bio.seek(0)
    return bio.getvalue()

def _log_text(r):
    out = [
        f"{r['unidade']}  ({r['lat']:.4f}, {r['lon']:.4f})",
        f"células no raio de {r['raio_km']:.0f} km: {r['n_celulas']}",
    ]

    for t, err in r.get("falhas", []):
        out.append(f"falhou {t:%H:%M}Z  {err}")

    if r.get("falhas"):
        out.append(
            f"granulos IMERG indisponíveis: {len(r['falhas'])}"
        )

    for t, err in r.get("falhas_goes", []):
        out.append(f"GOES falhou {t:%H:%M}Z  {err}")

    if r.get("goes") is not None:
        g = r["goes"]
        out.append(
            f"GOES-19: {g.sizes['time']} quadros até "
            f"{(pd.to_datetime(g.time.values[-1]) + TZ_LOCAL):%H:%M} local"
        )

    if r.get("erro_modelo"):
        out.append(f"previsão de modelo falhou: {r['erro_modelo']}")
    elif r.get("modelo") is not None:
        m = r["modelo"]
        out.append(
            f"previsão de modelo: {len(m)} horas, "
            f"acumulado {m.mm_h.sum():.1f} mm, "
            f"pico {m.mm_h.max():.1f} mm/h "
            f"(Open-Meteo, pontual)"
        )

    out.append(
        f"método do movimento: {'PySTEPS / Lucas-Kanade' if TEM_PYSTEPS else 'fallback por correlação de fase'}"
    )
    return "\n".join(out)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.title("🌧️ RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES")
st.caption(
    "IMERG Early Run + GOES-19 RRQPEF + nowcast lagrangiano + previsão numérica pontual."
)

with st.sidebar:
    st.header("Consulta")

    unidades = _unidades()
    nomes = [u[0] for u in unidades]
    busca = st.text_input("Buscar unidade", placeholder="Digite parte do nome...")

    filtradas = [
        u for u in unidades
        if busca.strip().lower() in u[0].lower()
    ]
    if not filtradas:
        st.warning("Nenhuma unidade encontrada.")
        st.stop()

    nome_sel = st.selectbox(
        "Unidade",
        [u[0] for u in filtradas],
        index=0,
    )
    _, lat, lon = next(u for u in filtradas if u[0] == nome_sel)

    st.divider()
    modo = st.radio(
        "Janela do IMERG",
        ["Dia todo", "Últimas horas"],
        index=0,
    )

    if modo == "Últimas horas":
        horas = st.number_input(
            "Horas", min_value=1.0, max_value=24.0,
            value=6.0, step=1.0
        )
    else:
        horas = 6.0

    raio = st.number_input(
        "Raio de extração (km)",
        min_value=5.0, max_value=100.0,
        value=15.0, step=5.0
    )

    estatistica = st.selectbox(
        "Estatística espacial",
        ["max", "mean"],
        index=0,
    )

    latencia = st.number_input(
        "Latência de fallback (h)",
        min_value=4.0, max_value=12.0,
        value=5.0, step=1.0
    )

    com_goes = st.checkbox(
        "Completar com GOES-19",
        value=True,
    )
    cadencia_goes = st.select_slider(
        "Cadência GOES",
        options=[10, 20, 30, 40, 50, 60],
        value=30,
        format_func=lambda x: f"{x} min",
    )

    com_nowcast = st.checkbox(
        "Gerar nowcast",
        value=True,
    )
    horizonte = st.number_input(
        "Horizonte do nowcast (h)",
        min_value=0.5, max_value=6.0,
        value=2.0, step=0.5,
        disabled=not com_nowcast,
    )

    com_modelo = st.checkbox(
        "Previsão de modelo (Open-Meteo)",
        value=True,
    )
    horas_modelo = st.number_input(
        "Horizonte do modelo (h)",
        min_value=1, max_value=72,
        value=24, step=1,
        disabled=not com_modelo,
    )

    consultar = st.button(
        "🚀 Consultar",
        type="primary",
        use_container_width=True,
    )

# ----------------------------------------------------------------------------
# Execução
# ----------------------------------------------------------------------------
if consultar:
    progress = st.progress(0, text="Iniciando...")
    status_box = st.empty()

    def progresso_web(feito, total, texto):
        frac = 0.0 if total <= 0 else min(max(feito / total, 0.0), 1.0)
        progress.progress(frac, text=f"{feito}/{total} — {texto}")
        status_box.caption(texto)

    try:
        with st.spinner("Baixando e processando os dados..."):
            resultado = coletar_dia(
                nome_sel,
                lat,
                lon,
                latencia_h=float(latencia),
                desde_meia_noite=(modo == "Dia todo"),
                horas=float(horas),
                raio_km=float(raio),
                estatistica=estatistica,
                com_nowcast=bool(com_nowcast),
                horizonte_h=float(horizonte),
                com_modelo=bool(com_modelo),
                horas_modelo=int(horas_modelo),
                com_goes=bool(com_goes),
                cadencia_goes=int(cadencia_goes),
                progresso=progresso_web,
                cancelar=None,
                email_pps=EMAIL_PPS,
            )

        st.session_state["resultado"] = resultado
        st.session_state["quadros"] = _build_frames(resultado)
        st.session_state["indice_quadro"] = 0
        st.session_state["zoom"] = 4.0
        progress.progress(1.0, text="Concluído")
        status_box.success("Consulta concluída.")
    except Exception as e:
        progress.empty()
        status_box.empty()
        st.error(f"{type(e).__name__}: {e}")
        st.exception(e)
        st.stop()

# ----------------------------------------------------------------------------
# Resultado
# ----------------------------------------------------------------------------
r = st.session_state.get("resultado")

if r is None:
    st.info(
        "Selecione uma unidade e clique em **Consultar**. "
        "A análise começa no IMERG e pode ser completada pelo GOES-19 e pelo nowcast."
    )
    st.markdown(
        """
        **Fontes e papéis**

        **IMERG Early Run:** estimativa observada quase em tempo real.  
        **GOES-19 RRQPEF:** cobertura intermediária em maior frequência.  
        **Nowcast:** extrapolação do campo mais recente.  
        **Open-Meteo:** previsão pontual por modelo numérico.
        """
    )
    st.stop()

s = r["resumo"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Acumulado observado", f"{s['acumulado_mm']:.1f} mm")
c2.metric("Pico", f"{s['pico_mm_h']:.1f} mm/h")
c3.metric("Classe do pico", s["classe_pico"])
c4.metric("Horários IMERG", f"{s['horarios']}")

st.caption(
    f"{r['unidade']} · {r['lat']:.4f}, {r['lon']:.4f} · "
    f"raio {r['raio_km']:.0f} km · estatística {r['estatistica']} · {ROTULO_TZ}"
)

quadros = st.session_state.get("quadros", _build_frames(r))
if quadros:
    n = len(quadros)
    idx = int(
        st.slider(
            "Linha do tempo",
            min_value=0,
            max_value=max(n - 1, 0),
            value=min(st.session_state.get("indice_quadro", 0), max(n - 1, 0)),
            step=1,
            format="%d",
        )
    )
    st.session_state["indice_quadro"] = idx

    nav1, nav2, nav3 = st.columns([1, 1, 8])
    with nav1:
        if st.button("◀", use_container_width=True):
            st.session_state["indice_quadro"] = max(idx - 1, 0)
            st.rerun()
    with nav2:
        if st.button("▶", use_container_width=True):
            st.session_state["indice_quadro"] = min(idx + 1, n - 1)
            st.rerun()

    zoom = st.number_input(
        "Zoom do mapa (°)",
        min_value=1.0, max_value=20.0,
        value=float(st.session_state.get("zoom", 4.0)),
        step=1.0,
    )
    st.session_state["zoom"] = zoom

    q = quadros[st.session_state["indice_quadro"]]
    fig = _map_fig(r, q, float(zoom))
else:
    st.warning("Nenhum quadro espacial foi produzido.")
    fig = None

tab_mapa, tab_tabela, tab_log = st.tabs(["🗺️ Mapa", "📋 Tabela", "🧾 Log"])

with tab_mapa:
    if fig is not None:
        st.pyplot(fig, clear_figure=False, use_container_width=True)

        q = quadros[st.session_state["indice_quadro"]]
        local = q["tempo_utc"] + TZ_LOCAL
        st.caption(
            f"Quadro {st.session_state['indice_quadro'] + 1}/{len(quadros)} · "
            f"{local:%d/%m/%Y %H:%M} local · {q['tipo']}"
        )

with tab_tabela:
    tabela = _table_df(r)
    if tabela.empty:
        st.info("Nenhum dado tabulável.")
    else:
        st.dataframe(
            tabela,
            use_container_width=True,
            hide_index=True,
            column_config={
                "mm_h": st.column_config.NumberColumn("mm/h", format="%.2f"),
                "probabilidade_%": st.column_config.NumberColumn(
                    "Prob. chuva (%)", format="%.0f"
                ),
            },
        )

with tab_log:
    st.code(_log_text(r), language="text")

st.divider()

download1, download2 = st.columns(2)
with download1:
    st.download_button(
        "⬇️ Baixar XLSX",
        data=_xlsx_bytes(r),
        file_name=f"{re.sub(r'[^A-Za-z0-9_-]+', '_', r['unidade'])}.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )

with download2:
    if fig is not None:
        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format="png", dpi=220, facecolor="white", bbox_inches="tight")
        st.download_button(
            "⬇️ Baixar PNG do mapa",
            data=png_buffer.getvalue(),
            file_name="precipitacao_mapa.png",
            mime="image/png",
            use_container_width=True,
        )

st.caption(
    "IMERG é estimativa observada; o nowcast é extrapolação do campo e não substitui "
    "previsão numérica em horizontes longos."
)
