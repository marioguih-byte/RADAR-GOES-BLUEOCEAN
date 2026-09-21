# RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES

Dashboard Streamlit usando Open-Meteo/DWD ICON para previsão e GLM do GOES-19 para observação recente de raios.

## Recursos
- Precipitação horária.
- Rajada de vento.
- Potencial holístico de raios.
- GLM do GOES-19 nos últimos 10 minutos para reforçar o diagnóstico do horário atual.
- Horizonte de 0 a +6 horas.
- Grade fixa de 0,5°.
- IDW fixo em potência 4,0.
- Campo espacial pixelado com pcolormesh.
- Cartopy para costa, fronteiras e estados.
- Setas de navegação entre horas.
- 40 unidades em ordem alfabética.
- Download CSV.

## GLM
Os flashes GLM-L2-LCFA do GOES-19 são lidos diretamente do bucket público `noaa-goes19` da NOAA/AWS. Os flashes não são desenhados como pontos; são agregados espacialmente e usados para reforçar o campo de potencial elétrico do horário atual.

## Execução
```bash
pip install -r requirements.txt
streamlit run app.py
```
