# RIO ULTRA POWER ULTIMATE ARNOLD SCHWARZENEGGER EDITION PREVISÕES

Aplicação Streamlit para monitoramento de precipitação com IMERG Early Run, GOES-19 RRQPEF, nowcast lagrangiano e previsão numérica pontual.

## Interface
- Tema geral do Streamlit em preto.
- Mapa meteorológico com Cartopy e fundo branco.
- Mapa reduzido e centralizado na tela.
- Visualização padrão em GIF animado.
- Alternativa de visualização quadro a quadro com linha do tempo e botões de navegação.

## Dependências
O `requirements.txt` inclui h5py, cftime, Cartopy e Pillow para leitura dos arquivos HDF5/netCDF e geração do GIF.

## Execução
```bash
pip install -r requirements.txt
streamlit run app.py
```
