# RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES

Aplicação Streamlit de previsão de curtíssimo prazo usando uma única consulta DWD ICON via Open-Meteo.

- Precipitação: ICON
- Rajada de vento: ICON
- Potencial heurístico de raios: ICON (weather_code, CAPE, showers, precipitation probability, umidade e nebulosidade)
- Grade fixa: 0,5°
- Região fixa: ±2,5°
- IDW fixo: potência 4,0
- Horizonte: até +6 h
- Um único request por atualização, com cache de 30 minutos
- Fallback para `/v1/forecast?models=icon_global` somente se o endpoint DWD ICON retornar erro temporário

## Executar
```bash
pip install -r requirements.txt
streamlit run app.py
```

O índice de raios é um indicador heurístico e não deve ser interpretado como probabilidade estatística calibrada de descarga.


## Otimização contra HTTP 429
A consulta espacial usa 9 pontos (3x3) por região, cache de 1 hora e fallback automático para ICON Global/ponto único quando o servidor limita temporariamente a origem.
