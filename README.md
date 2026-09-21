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
