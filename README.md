# RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES

Versão Weatherbit.

## O que precisa
Crie uma conta Weatherbit e obtenha uma API key.
No Streamlit Cloud, coloque:

WEATHERBIT_API_KEY = "SUA_CHAVE"

em **Manage app → Settings → Secrets**.

## Recursos
- 40 unidades em ordem alfabética.
- Região fixa em ±2,5°.
- Grade visual fixa em 0,5°.
- IDW fixo, potência 4.
- Precipitação, rajada e potencial de raios.
- Previsão horária de 0 a +6 h.
- Navegação por botões de hora.
- Weatherbit Current Lightning, quando disponível no plano.
- Download CSV.

## Limitação importante
A API Hourly é por ponto. Esta implementação consulta 9 pontos da região e faz IDW. No plano gratuito da Weatherbit, a documentação informa 50 requests/dia e 1 request/s; por isso a aplicação usa consultas sequenciais com pequena espera e cache. A Current Lightning API consome quota adicional e pode depender do plano.
