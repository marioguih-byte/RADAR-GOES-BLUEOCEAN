import unicodedata
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

st.set_page_config(page_title='RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES', page_icon='🌩️', layout='wide')

UNIDADES = sorted([
('UTE Termocamaçari - UTE TCA',-12.666868,-38.314687),('UTE Termobahia - UTE TBA',-12.703235,-38.564904),('UTE Termoceará - UTE TCE',-3.692456,-38.870605),('UTE Vale do Açu - UTE VLA',-5.381685,-36.819754),('Refinaria Abreu e Lima - RNEST',-8.379661,-35.010198),('Unidade de Tratamento de Gás Sul Capixaba - UTGSUL',-20.794464,-40.620910),('Unidade de Tratamento de Gás de Cacimbas - UTGC',-19.463097,-39.760603),('Refinaria Duque de Caxias - REDUC',-22.715100,-43.284008),('UTE Termorio - UTE TRI',-22.714882,-43.254348),('BOAVENTURA, Itaboraí-RJ',-22.660714,-42.853629),('Unidade de Tratamento de Gás de Cabiúnas - UTGCAB',-22.285327,-41.717905),('UTE Termomacaé - UTE TMA',-22.306164,-41.876702),('UTE Seropédica/Baixada Fluminense - UTE SRP/BF',-22.723292,-43.647719),('Refinaria Gabriel Passos - REGAP',-19.964281,-44.095138),('UTE Ibirité - UTE IBT',-19.988579,-44.098207),('UTE Juiz de Fora - UTE JF',-21.690619,-43.456720),('UTE Três Lagoas - UTE TLG',-20.745500,-51.664590),('Unidade de Tratamento de Gás de Caraguatatuba - UTGCA',-23.654191,-45.501209),('Refinaria Presidente Bernardes - RPBC',-23.873326,-46.427572),('UTE Cubatão - UTE CBT',-23.875735,-46.431387),('Refinaria Henrique Lage - REVAP',-23.184796,-45.815806),('Refinaria de Capuava - RECAP',-23.656681,-46.480883),('Refinaria de Paulínia - REPLAN',-22.729589,-47.147712),('UTE Nova Piratininga - UTE NPI',-23.699414,-46.673881),('Refinaria Presidente Getúlio Vargas - REPAR',-25.566140,-49.369423),('Refinaria Alberto Pasqualini - REFAP',-29.869901,-51.178195),('UTE Canoas - UTE CAN',-29.875071,-51.145445),('Armazém Rio de Janeiro',-22.810973,-43.281876),('CILEP - CENPES',-22.854198,-43.233830),('Porto Baia de Guanabara',-22.878942,-43.209330),('ARM Macaé - Armazém Macaé',-22.415317,-41.861353),('Porto de Imbetiba - Macaé',-22.386825,-41.768741),('Porto Açu',-21.864737,-41.016444),('Porto Aratu',-12.780132,-38.496764),('Porto TMIB',-10.824129,-36.946301),('Porto Belém',-1.439901,-48.494920),('Porto Valença',-13.369369,-39.071252),('Porto Guamaré',-5.106692,-36.319592),('Porto Mucuripe',-3.713117,-38.474037),('Porto Paracuru',-3.401146,-39.010896)
], key=lambda u: unicodedata.normalize('NFKD', u[0]).encode('ascii', 'ignore').decode().casefold())

@st.cache_data(ttl=600, show_spinner=False)
def consultar_open_meteo(lats, lons):
    url='https://api.open-meteo.com/v1/forecast'
    params={
        'latitude':','.join(f'{x:.5f}' for x in lats),
        'longitude':','.join(f'{x:.5f}' for x in lons),
        'hourly': 'precipitation,precipitation_probability,wind_gusts_10m,cape,lifted_index,cloud_cover',
        'forecast_hours': 7,
        'timezone':'America/Sao_Paulo',
        'wind_speed_unit':'kmh', 'precipitation_unit':'mm', 'temperature_unit':'celsius'
    }
    r=requests.get(url,params=params,timeout=90)
    r.raise_for_status()
    raw=r.json()
    items=raw if isinstance(raw,list) else [raw]
    rows=[]
    for i,item in enumerate(items):
        h=item['hourly']
        for j,t in enumerate(h['time']):
            rows.append({
                'lat':lats[i], 'lon':lons[i], 'tempo':pd.Timestamp(t),
                'precipitation':h['precipitation'][j], 'precipitation_probability':h['precipitation_probability'][j],
                'wind_gusts_10m':h['wind_gusts_10m'][j], 'cape':h['cape'][j], 'lifted_index':h['lifted_index'][j], 'cloud_cover':h['cloud_cover'][j]
            })
    df=pd.DataFrame(rows)
    p=np.clip(pd.to_numeric(df.precipitation),0,None)
    g=np.clip(pd.to_numeric(df.wind_gusts_10m),0,None)
    c=np.clip(pd.to_numeric(df.cape),0,None)
    pp=np.clip(pd.to_numeric(df.precipitation_probability),0,100)
    li=pd.to_numeric(df.lifted_index).fillna(0)
    score=(0.42*np.clip((p-0.2)/(12-0.2),0,1)+0.28*np.clip((c-50)/(1800-50),0,1)+0.15*np.clip((g-25)/(80-25),0,1)+0.10*(pp/100)+0.05*np.clip((-li+1)/(8+1),0,1))*100
    score=np.where(p<0.15,score*0.30,score)
    df['raios_score']=score
    df['raios_classe']=np.select([score<20,score<40,score<60,score<80],['MUITO BAIXO','BAIXO','MODERADO','ALTO'],default='MUITO ALTO')
    return df

def pontos_grade(lat0, lon0, halfspan):
    step = 0.5
    lats = np.arange(
        np.floor((lat0-halfspan)/step)*step,
        np.ceil((lat0+halfspan)/step)*step + step*0.51,
        step
    )
    lons = np.arange(
        np.floor((lon0-halfspan)/step)*step,
        np.ceil((lon0+halfspan)/step)*step + step*0.51,
        step
    )
    glat, glon = np.meshgrid(lats, lons, indexing='ij')
    return glat.ravel(), glon.ravel()

def idw(values_lat,values_lon,values,grid_lat,grid_lon,power=2,k=30):
    la=np.asarray(values_lat,float); lo=np.asarray(values_lon,float); v=np.asarray(values,float)
    good=np.isfinite(la)&np.isfinite(lo)&np.isfinite(v); la,lo,v=la[good],lo[good],v[good]
    lat0=np.deg2rad(np.mean(la)); xp=np.deg2rad(lo)*np.cos(lat0)*6371; yp=np.deg2rad(la)*6371
    xg=np.deg2rad(grid_lon.ravel())*np.cos(lat0)*6371; yg=np.deg2rad(grid_lat.ravel())*6371
    out=np.full(xg.size,np.nan)
    for a in range(0,len(out),3000):
        sl=slice(a,a+3000); dx=xg[sl,None]-xp; dy=yg[sl,None]-yp; d=np.hypot(dx,dy)
        n=min(k,d.shape[1]); idx=np.argpartition(d,max(n-1,0),axis=1)[:,:n]; dd=np.take_along_axis(d,idx,axis=1); vv=v[idx]
        exact=dd<1e-9
        with np.errstate(divide='ignore',invalid='ignore'):
            w=1/np.maximum(dd,0.001)**power
            z=np.sum(w*vv,axis=1)/np.sum(w,axis=1)
        if exact.any(): z[exact.any(axis=1)]=vv[np.arange(len(vv))[exact.any(axis=1)],np.argmax(exact[exact.any(axis=1)],axis=1)]
        out[sl]=z
    return out.reshape(grid_lat.shape)

def plotar(grid_lat, grid_lon, field, unit, ulat, ulon, when, title, label, vmax, cmap, points, modo="normal"):
    fig = plt.figure(figsize=(7.15, 4.35), dpi=135, facecolor="white")
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor("white")

    x0, x1 = float(grid_lon.min()) - 0.25, float(grid_lon.max()) + 0.25
    y0, y1 = float(grid_lat.min()) - 0.25, float(grid_lat.max()) + 0.25
    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

    if modo == "raios":
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cores = ["#2E7D32", "#8BC34A", "#FDD835", "#FB8C00", "#D32F2F"]
        bounds = [0, 20, 40, 60, 80, 100]
        cmap_obj = ListedColormap(cores)
        norm = BoundaryNorm(bounds, cmap_obj.N)
        pm = ax.pcolormesh(
            grid_lon, grid_lat, np.clip(field, 0, 100),
            cmap=cmap_obj, norm=norm,
            shading="nearest",
            transform=ccrs.PlateCarree(),
            rasterized=True,
            zorder=1
        )
        cb = plt.colorbar(pm, ax=ax, pad=.018, shrink=.79, ticks=[10, 30, 50, 70, 90])
        cb.ax.set_yticklabels(
            ["MUITO BAIXO", "BAIXO", "MODERADO", "ALTO", "MUITO ALTO"],
            fontsize=7.4
        )
        cb.set_label("POTENCIAL DE RAIOS", fontsize=9)
    else:
        pm = ax.pcolormesh(
            grid_lon, grid_lat, np.clip(field, 0, vmax),
            cmap=cmap,
            shading="nearest",
            transform=ccrs.PlateCarree(),
            rasterized=True,
            zorder=1
        )
        cb = plt.colorbar(pm, ax=ax, pad=.018, shrink=.79)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#F3F4F6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=.8, edgecolor="#222222", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=.65, edgecolor="#333333", zorder=3)
    try:
        ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=.38, edgecolor="#444444", zorder=3)
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
    ax.tick_params(axis="both", labelsize=8, colors="#333333", width=.7)

    for x in lon_ticks:
        ax.plot([x, x], [y0, y1], transform=ccrs.PlateCarree(),
                color="#888888", linewidth=.28, alpha=.24, zorder=2)
    for y in lat_ticks:
        ax.plot([x0, x1], [y, y], transform=ccrs.PlateCarree(),
                color="#888888", linewidth=.28, alpha=.24, zorder=2)

    ax.scatter([ulon], [ulat], s=116, marker="^", facecolor="white",
               edgecolor="black", linewidth=1.8, zorder=6, transform=ccrs.PlateCarree())
    ax.scatter([ulon], [ulat], s=14, marker="o", facecolor="#111111",
               edgecolor="white", linewidth=.6, zorder=7, transform=ccrs.PlateCarree())

    if st.session_state.get("show_points", False):
        ax.scatter(
            points["lon"], points["lat"], s=12, c="#444444", alpha=.20,
            zorder=5, transform=ccrs.PlateCarree()
        )

    ax.set_title(
        f"{unit}\n{pd.Timestamp(when):%d/%m/%Y %H:%M} local • {title}",
        fontsize=12.0, fontweight="bold", color="#171717", pad=8
    )
    fig.subplots_adjust(left=.07, right=.91, bottom=.10, top=.86)
    return fig

st.markdown('''<style>
    .stApp{background:#0d0f14}
    .block-container{max-width:1380px;padding-top:1rem;padding-bottom:2rem}
    section[data-testid="stSidebar"]{background:#11141b}
    h1,h2,h3,p,label,.stMarkdown,.stCaption{color:#f2f2f2!important}
    div[data-testid="stMetric"]{background:#151922;border:1px solid #282d37;border-radius:12px;padding:10px 12px}
    div[data-testid="stMetricLabel"]{color:#aeb4bf!important}
    .lightning-card{background:#151922;border:1px solid #2b303a;border-radius:14px;padding:14px 18px;margin:0 0 12px}
    .lightning-title{font-size:1.05rem;font-weight:800;color:#fff}
    .lightning-sub{font-size:.86rem;color:#aeb4bf}
    .legend-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
    .legend-chip{font-size:.72rem;padding:5px 9px;border-radius:999px;color:#111;font-weight:700}
    </style>''',unsafe_allow_html=True)
st.title('🌩️ RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES')
st.caption('OPEN-METEO • PREVISÃO HORÁRIA • ATÉ +6 H • IDW 0,5° • ÍNDICE HEURÍSTICO DE RAIOS')

with st.sidebar:
    st.header('CONFIGURAÇÃO')
    q=st.text_input('BUSCAR UNIDADE','')
    opts=[u[0] for u in UNIDADES if q.lower() in u[0].lower()]
    if not opts: st.stop()
    name=st.selectbox('UNIDADE',opts)
    unit=next(u for u in UNIDADES if u[0]==name); ulat,ulon=unit[1],unit[2]
    hours=st.slider('HORAS À FRENTE',1,6,6)
    halfspan=st.slider('EXTENSÃO DA REGIÃO (°)',1.0,5.0,2.5,.5)
    power=st.slider('POTÊNCIA IDW',1.0,4.0,2.0,.5)
    st.session_state['show_points']=st.checkbox('MOSTRAR PONTOS DE AMOSTRAGEM',False)

lats,lons=pontos_grade(ulat,ulon,halfspan)
with st.spinner(f'CONSULTANDO OPEN-METEO PARA {len(lats)} PONTOS...'):
    try: df=consultar_open_meteo(lats.tolist(),lons.tolist())
    except Exception as e: st.error(f'ERRO OPEN-METEO: {type(e).__name__}: {e}'); st.stop()

times=sorted(df.tempo.unique())[:hours+1]
when=st.select_slider('HORÁRIO',options=times,value=times[0],format_func=lambda x:pd.Timestamp(x).strftime('%d/%m %H:%M'))
fr=df[df.tempo==when].copy()
glat=np.linspace(ulat-halfspan,ulat+halfspan,105); glon=np.linspace(ulon-halfspan/max(np.cos(np.deg2rad(ulat)),.55),ulon+halfspan/max(np.cos(np.deg2rad(ulat)),.55),125); GLA,GLO=np.meshgrid(glat,glon,indexing='ij')

maps=[
    ('PRECIPITAÇÃO','precipitation','PRECIPITAÇÃO (MM/H)',
     max(10,float(np.nanpercentile(df.precipitation,99.5) or 10)),'turbo','normal'),
    ('RAJADA DE VENTO','wind_gusts_10m','RAJADA (KM/H)',
     max(60,float(np.nanpercentile(df.wind_gusts_10m,99.5) or 60)),'magma','normal'),
    ('POTENCIAL DE RAIOS','raios_score','',
     100,'YlOrRd','raios')
]
T1,T2,T3=st.tabs(['🌧️ PRECIPITAÇÃO','💨 RAJADA DE VENTO','⚡ POTENCIAL DE RAIOS'])
for tab,(ttl,var,lab,vmax,cmap,modo) in zip([T1,T2,T3],maps):
    with tab:
        field=idw(fr.lat,fr.lon,fr[var],GLA,GLO,power)
        fig=plotar(GLA,GLO,field,name,ulat,ulon,when,ttl,lab,vmax,cmap,fr,modo=modo)
        st.pyplot(fig,use_container_width=False)
        plt.close(fig)
        if modo == "raios":
            st.markdown('''
            <div class="lightning-card">
              <div class="lightning-title">⚡ COMO INTERPRETAR</div>
              <div class="lightning-sub">O índice varia de 0 a 100. Quanto maior o valor, maior o potencial convectivo estimado pelo método heurístico.</div>
              <div class="legend-row">
                <span class="legend-chip" style="background:#2E7D32">0–19 MUITO BAIXO</span>
                <span class="legend-chip" style="background:#8BC34A">20–39 BAIXO</span>
                <span class="legend-chip" style="background:#FDD835">40–59 MODERADO</span>
                <span class="legend-chip" style="background:#FB8C00">60–79 ALTO</span>
                <span class="legend-chip" style="background:#D32F2F;color:#fff">80–100 MUITO ALTO</span>
              </div>
            </div>
            ''',unsafe_allow_html=True)

nearest=((df.lat-ulat).abs()+(df.lon-ulon).abs()).groupby(df.tempo).idxmin(); serie=df.loc[nearest].sort_values('tempo').head(hours+1).copy()
tab=serie[['tempo','precipitation','wind_gusts_10m','cape','raios_score','raios_classe']].copy(); tab.columns=['HORÁRIO','PRECIPITAÇÃO (MM/H)','RAJADA (KM/H)','CAPE (J/KG)','ÍNDICE RAIOS','CLASSE RAIOS']; tab['HORÁRIO']=pd.to_datetime(tab.HORÁRIO).dt.strftime('%d/%m %H:%M')
# Resumo do horário selecionado na unidade.
unit_row = fr.iloc[((fr['lat']-ulat).abs() + (fr['lon']-ulon).abs()).argsort()[:1]].iloc[0]
score_u = float(unit_row['raios_score'])
classe_u = str(unit_row['raios_classe'])

st.subheader('CONDIÇÕES NA UNIDADE')
c1,c2,c3,c4 = st.columns(4)
c1.metric('PRECIPITAÇÃO', f"{float(unit_row['precipitation']):.1f} mm/h")
c2.metric('RAJADA', f"{float(unit_row['wind_gusts_10m']):.0f} km/h")
c3.metric('PROB. DE CHUVA', f"{float(unit_row['precipitation_probability']):.0f}%")
c4.metric('POTENCIAL DE RAIOS', f"{score_u:.0f}/100")

badge_colors = {
    'MUITO BAIXO':'#2E7D32',
    'BAIXO':'#8BC34A',
    'MODERADO':'#FDD835',
    'ALTO':'#FB8C00',
    'MUITO ALTO':'#D32F2F'
}
badge = badge_colors.get(classe_u,'#777777')
st.markdown(
    f'<div class="lightning-card"><span class="lightning-title">⚡ POTENCIAL DE RAIOS: '
    f'<span style="background:{badge};color:#111;padding:5px 10px;border-radius:999px">{classe_u}</span></span>'
    f'<div class="lightning-sub" style="margin-top:7px">Índice heurístico baseado nas condições meteorológicas previstas pelo Open-Meteo. '
    f'Não representa probabilidade observacional de descargas.</div></div>',
    unsafe_allow_html=True
)

st.subheader('PREVISÃO HORÁRIA NA UNIDADE')
st.dataframe(tab,use_container_width=True,hide_index=True)
st.download_button('⬇️ BAIXAR CSV',tab.to_csv(index=False).encode('utf-8-sig'),'previsao_openmeteo.csv','text/csv')

st.caption('Fonte dos dados meteorológicos: Open-Meteo. O campo espacial é interpolado por IDW. O índice de raios é heurístico e serve como indicador de potencial convectivo, não como detecção de descargas.')
