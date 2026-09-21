# RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES

Dashboard Streamlit usando exclusivamente a API Open-Meteo.

## Recursos
- Precipitação horária.
- Rajada de vento a 10 m.
- Potencial holístico de raios.
- Horizonte de 0 a +6 horas.
- Grade de consulta Open-Meteo fixa em 0,5°.
- Potência IDW fixa em 4,0.
- Campo espacial pixelado com `pcolormesh` sem linhas entre as células.
- Cartopy para costa, fronteiras e estados.
- Setas para navegar entre as horas.
- 40 unidades em ordem alfabética.
- Download CSV da previsão horária.

## Potencial holístico de raios
O índice combina a probabilidade de trovoada do Open-Meteo com a densidade de raios modelada pelo ECMWF disponibilizada pelo Open-Meteo, além de precipitação convectiva, CAPE, Lifted Index, probabilidade de precipitação, umidade e cobertura de nuvens.

Ele é um indicador de potencial previsto, não uma observação/detecção de descargas em tempo real.

## Executar
```bash
pip install -r requirements.txt
streamlit run app.py
```


## Raios

A versão atual combina previsão do Open-Meteo (GFS/ECMWF) com observação recente do GLM do GOES-19. O GLM-L2-LCFA fornece flashes individuais com centroides geográficos; a aplicação conta os flashes dos últimos 10 minutos em um raio de 75 km da unidade e usa essa observação para corrigir o horário corrente e alimentar um decaimento de curto prazo.


## GLM GOES-19
A camada GLM usa o produto GLM-L2-LCFA do GOES-19, lido no bucket público `noaa-goes19` da NOAA/AWS. Os arquivos contêm `flash_lat`, `flash_lon` e `flash_energy`. Os flashes não são desenhados individualmente.

No horário atual, os flashes dos últimos 10 minutos são agregados por célula e usados para reforçar o campo de raios. Para +1 a +6 h, o padrão espacial observado é deslocado usando o movimento estimado entre as duas últimas janelas de 10 minutos e recebe decaimento temporal; esse nowcast é combinado ao componente previsto pelo ICON.
