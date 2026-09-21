# RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES

Aplicação Streamlit usando WeatherAPI.com para previsão horária.

## Recursos
- 40 unidades em ordem alfabética.
- Região fixa de ±2,5°.
- Grade visual de 0,5°.
- IDW fixo em potência 4.
- Precipitação e rajadas horárias.
- Índice heurístico de potencial de raios.
- Horizonte de 0 a +6 h.
- Navegação horária.
- Cache de 30 minutos.

## API
A chave WeatherAPI informada pelo usuário está configurada no backend do `app.py` e não é exibida na interface.

A WeatherAPI fornece previsão horária no endpoint `/forecast.json`; `q` aceita latitude/longitude e `days=1` é suficiente para o horizonte de até 6 horas. A opção de bulk é restrita a planos Pro+, por isso esta versão usa cinco pontos regionais e cache para reduzir o número de chamadas.
