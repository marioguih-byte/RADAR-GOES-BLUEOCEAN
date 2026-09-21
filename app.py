import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

st.set_page_config(page_title='RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES', page_icon='🌩️', layout='wide')

UNIDADES = [
('UTE Termocamaçari - UTE TCA',-12.666868,-38.314687),('UTE Termobahia - UTE TBA',-12.703235,-38.564904),('UTE Termoceará - UTE TCE',-3.692456,-38.870605),('UTE Vale do Açu - UTE VLA',-5.381685,-36.819754),('Refinaria Abreu e Lima - RNEST',-8.379661,-35.010198),('Unidade de Tratamento de Gás Sul Capixaba - UTGSUL',-20.794464,-40.620910),('Unidade de Tratamento de Gás de Cacimbas - UTGC',-19.463097,-39.760603),('Refinaria Duque de Caxias - REDUC',-22.715100,-43.284008),('UTE Termorio - UTE TRI',-22.714882,-43.254348),('BOAVENTURA, Itaboraí-RJ',-22.660714,-42.853629),('Unidade de Tratamento de Gás de Cabiúnas - UTGCAB',-22.285327,-41.717905),('UTE Termomacaé - UTE TMA',-22.306164,-41.876702),('UTE Seropédica/Baixada Fluminense - UTE SRP/BF',-22.723292,-43.647719),('Refinaria Gabriel Passos - REGAP',-19.964281,-44.095138),('UTE Ibirité - UTE IBT',-19.988579,-44.098207),('UTE Juiz de Fora - UTE JF',-21.690619,-43.456720),('UTE Três Lagoas - UTE TLG',-20.745500,-51.664590),('Unidade de Tratamento de Gás de Caraguatatuba - UTGCA',-23.654191,-45.501209),('Refinaria Presidente Bernardes - RPBC',-23.873326,-46.427572),('UTE Cubatão - UTE CBT',-23.875735,-46.431387),('Refinaria Henrique Lage - REVAP',-23.184796,-45.815806),('Refinaria de Capuava - RECAP',-23.656681,-46.480883),('Refinaria de Paulínia - REPLAN',-22.729589,-47.147712),('UTE Nova Piratininga - UTE NPI',-23.699414,-46.673881),('Refinaria Presidente Getúlio Vargas - REPAR',-25.566140,-49.369423),('Refinaria Alberto Pasqualini - REFAP',-29.869901,-51.178195),('UTE Canoas - UTE CAN',-29.875071,-51.145445),('Armazém Rio de Janeiro',-22.810973,-43.281876),('CILEP - CENPES',-22.854198,-43.233830),('Porto Baia de Guanabara',-22.878942,-43.209330),('ARM Macaé - Armazém Macaé',-22.415317,-41.861353),('Porto de Imbetiba - Macaé',-22.386825,-41.768741),('Porto Açu',-21.864737,-41.016444),('Porto Aratu',-12.780132,-38.496764),('Porto TMIB',-10.824129,-36.946301),('Porto Belém',-1.439901,-48.494920),('Porto Valença',-13.369369,-39.071252),('Porto Guamaré',-5.106692,-36.319592),('Porto Mucuripe',-3.713117,-38.474037),('Porto Paracuru',-3.401146,-39.010896)]

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

def pontos_grade(lat0,lon0,halfspan,step):
    lats=np.arange(lat0-halfspan,lat0+halfspan+step*0.51,step)
    dlon=step/max(np.cos(np.deg2rad(lat0)),0.55)
    lons=np.arange(lon0-halfspan/max(np.cos(np.deg2rad(lat0)),0.55),lon0+halfspan/max(np.cos(np.deg2rad(lat0)),0.55)+dlon*0.51,dlon)
    glat,glon=np.meshgrid(lats,lons,indexing='ij')
    return glat.ravel(),glon.ravel()

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

def plotar(grid_lat,grid_lon,field,unit,ulat,ulon,when,title,label,vmax,cmap,points):
    fig=plt.figure(figsize=(8.0,4.9),dpi=125); ax=plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([grid_lon.min()-0.25,grid_lon.max()+0.25,grid_lat.min()-0.25,grid_lat.max()+0.25])
    lev=np.linspace(0,vmax,24)
    cf=ax.contourf(grid_lon,grid_lat,np.clip(field,0,vmax),levels=lev,cmap=cmap,transform=ccrs.PlateCarree(),extend='max')
    ax.add_feature(cfeature.OCEAN,facecolor='white',zorder=0); ax.add_feature(cfeature.LAND,facecolor='#f2f2f2',zorder=0)
    ax.add_feature(cfeature.COASTLINE,linewidth=.75,zorder=3); ax.add_feature(cfeature.BORDERS,linewidth=.6,zorder=3)
    try: ax.add_feature(cfeature.STATES.with_scale('10m'),linewidth=.35,zorder=3)
    except Exception: pass
    # Não usar GeoAxes.gridlines aqui: algumas combinações Cartopy/Shapely
    # podem falhar na construção do polígono da moldura durante o draw.
    # A grade é desenhada com ticks normais do GeoAxes, mantendo Cartopy.
    xmin, xmax = float(grid_lon.min()), float(grid_lon.max())
    ymin, ymax = float(grid_lat.min()), float(grid_lat.max())
    xt = np.arange(np.floor(xmin), np.ceil(xmax) + 1, 1.0)
    yt = np.arange(np.floor(ymin), np.ceil(ymax) + 1, 1.0)
    ax.set_xticks(xt, crs=ccrs.PlateCarree())
    ax.set_yticks(yt, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f', degree_symbol='°'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f', degree_symbol='°'))
    ax.tick_params(axis='both', labelsize=8.5, length=3, width=.6)
    ax.grid(True, linewidth=.35, alpha=.4, linestyle='--', zorder=2)

    ax.scatter([ulon],[ulat],s=90,marker='^',facecolor='white',edgecolor='black',linewidth=1.4,zorder=5,transform=ccrs.PlateCarree())
    if st.session_state.get('show_points',True): ax.scatter(points['lon'],points['lat'],s=7,c='black',alpha=.25,zorder=4,transform=ccrs.PlateCarree())
    ax.set_title(f'{unit}\n{pd.Timestamp(when):%d/%m/%Y %H:%M} local • {title}',fontsize=12.5,fontweight='bold',pad=10)
    cb=plt.colorbar(cf,ax=ax,pad=.02,shrink=.80); cb.set_label(label)
    fig.tight_layout(); return fig

st.markdown('''<style>.stApp{background:#0d0f14}.block-container{max-width:1400px;padding-top:1rem}.stSidebar{background:#11141b}h1,h2,h3,p,label,.stMarkdown,.stCaption{color:#f2f2f2!important}</style>''',unsafe_allow_html=True)
st.title('🌩️ RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES')
st.caption('OPEN-METEO • PREVISÃO HORÁRIA • ATÉ +6 H • IDW • ÍNDICE HEURÍSTICO DE RAIOS')

with st.sidebar:
    st.header('CONFIGURAÇÃO')
    q=st.text_input('BUSCAR UNIDADE','')
    opts=[u[0] for u in UNIDADES if q.lower() in u[0].lower()]
    if not opts: st.stop()
    name=st.selectbox('UNIDADE',opts)
    unit=next(u for u in UNIDADES if u[0]==name); ulat,ulon=unit[1],unit[2]
    hours=st.slider('HORAS À FRENTE',1,6,6)
    halfspan=st.slider('EXTENSÃO DA REGIÃO (°)',1.0,5.0,2.5,.5)
    step=st.select_slider('PONTOS OPEN-METEO (°)',options=[.25,.5,.75,1.0],value=.5)
    power=st.slider('POTÊNCIA IDW',1.0,4.0,2.0,.5)
    st.session_state['show_points']=st.checkbox('MOSTRAR PONTOS',True)

lats,lons=pontos_grade(ulat,ulon,halfspan,step)
with st.spinner(f'CONSULTANDO OPEN-METEO PARA {len(lats)} PONTOS...'):
    try: df=consultar_open_meteo(lats.tolist(),lons.tolist())
    except Exception as e: st.error(f'ERRO OPEN-METEO: {type(e).__name__}: {e}'); st.stop()

times=sorted(df.tempo.unique())[:hours+1]
when=st.select_slider('HORÁRIO',options=times,value=times[0],format_func=lambda x:pd.Timestamp(x).strftime('%d/%m %H:%M'))
fr=df[df.tempo==when].copy()
glat=np.linspace(ulat-halfspan,ulat+halfspan,105); glon=np.linspace(ulon-halfspan/max(np.cos(np.deg2rad(ulat)),.55),ulon+halfspan/max(np.cos(np.deg2rad(ulat)),.55),125); GLA,GLO=np.meshgrid(glat,glon,indexing='ij')

maps=[('PRECIPITAÇÃO','precipitation','PRECIPITAÇÃO (MM/H)',max(10,float(np.nanpercentile(df.precipitation,99.5) or 10)),'turbo'),('RAJADA DE VENTO','wind_gusts_10m','RAJADA (KM/H)',max(60,float(np.nanpercentile(df.wind_gusts_10m,99.5) or 60)),'magma'),('ÍNDICE HEURÍSTICO DE RAIOS','raios_score','ÍNDICE 0–100',100,'YlOrRd')]
T1,T2,T3=st.tabs(['PRECIPITAÇÃO','RAJADA DE VENTO','PREVISÃO DE RAIOS'])
for tab,(ttl,var,lab,vmax,cmap) in zip([T1,T2,T3],maps):
    with tab:
        field=idw(fr.lat,fr.lon,fr[var],GLA,GLO,power)
        fig=plotar(GLA,GLO,field,name,ulat,ulon,when,ttl,lab,vmax,cmap,fr); st.pyplot(fig,use_container_width=False); plt.close(fig)

nearest=((df.lat-ulat).abs()+(df.lon-ulon).abs()).groupby(df.tempo).idxmin(); serie=df.loc[nearest].sort_values('tempo').head(hours+1).copy()
tab=serie[['tempo','precipitation','wind_gusts_10m','cape','raios_score','raios_classe']].copy(); tab.columns=['HORÁRIO','PRECIPITAÇÃO (MM/H)','RAJADA (KM/H)','CAPE (J/KG)','ÍNDICE RAIOS','CLASSE RAIOS']; tab['HORÁRIO']=pd.to_datetime(tab.HORÁRIO).dt.strftime('%d/%m %H:%M')
st.subheader('PREVISÃO HORÁRIA NA UNIDADE'); st.dataframe(tab,use_container_width=True,hide_index=True)
st.download_button('⬇️ BAIXAR CSV',tab.to_csv(index=False).encode('utf-8-sig'),'previsao_openmeteo.csv','text/csv')
st.caption('Fonte dos dados meteorológicos: Open-Meteo. O índice de raios é uma heurística baseada nas variáveis retornadas pela API; não é observação nem detecção de descargas atmosféricas.')
