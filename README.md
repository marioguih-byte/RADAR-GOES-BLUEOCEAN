# RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES

Dashboard Streamlit baseado na versão estável do aplicativo, mantendo a estrutura do Open-Meteo/DWD ICON e acrescentando observação real do GLM do GOES-19 para raios.

## Recursos
- Precipitação horária do DWD ICON via Open-Meteo.
- Rajada de vento a 10 m.
- Potencial holístico de raios.
- Horizonte de 0 a +6 horas.
- Grade espacial fixa de 0,5°.
- Potência IDW fixa em 4,0.
- Campo espacial pixelado com `pcolormesh` sem linhas entre as células.
- Cartopy para costa, fronteiras e estados.
- Navegação entre as horas.
- 40 unidades em ordem alfabética.
- Download CSV da previsão horária.

## GLM do GOES-19
A aplicação consulta os arquivos públicos NOAA GOES-19 GLM-L2-LCFA no bucket `noaa-goes19`, lê os centroides geográficos dos flashes (`flash_lat` e `flash_lon`) e restringe a leitura à região da unidade selecionada.

O horário corrente utiliza a observação real recente do GLM. Para as horas seguintes, os flashes observados alimentam um nowcast espacial simples por deslocamento do centro de atividade e decaimento temporal, combinado com o potencial previsto pelo ICON.

Os flashes individuais não são desenhados como pontos no mapa. O GLM é convertido em um campo espacial e visualizado na mesma grade de 0,5° do aplicativo.

## Observação importante
O GLM é uma observação, não uma previsão. O nowcast futuro é uma extrapolação de curto prazo baseada na atividade observada recentemente; ele não deve ser interpretado como previsão estatística calibrada de raios.

## Executar
```bash
pip install -r requirements.txt
streamlit run app.py
```
