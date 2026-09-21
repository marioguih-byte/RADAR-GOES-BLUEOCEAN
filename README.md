# RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES

Dashboard Streamlit com dados meteorológicos exclusivamente da API Open-Meteo.

## Inclui
- Precipitação horária
- Rajada de vento a 10 m
- Índice heurístico de previsão de raios
- Horizonte de 1 a 6 horas
- Interpolação IDW espacial
- Cartopy para costa, fronteiras e estados
- 40 unidades Petrobras
- Tabela e download CSV

## Execução
```bash
pip install -r requirements.txt
streamlit run app.py
```

O índice de raios é heurístico e combina precipitação, probabilidade de precipitação, CAPE, índice de levantamento e rajadas. Não representa detecção de descargas.
