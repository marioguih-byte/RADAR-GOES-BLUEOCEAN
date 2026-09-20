

### Correção do erro `No module named h5py`

O IMERG/GOES é lido como HDF5 por meio do `h5netcdf`, que utiliza o `h5py`. O `requirements.txt` já inclui `h5py>=3.15,<4`, compatível com Python 3.14. Em serviços como o Streamlit Cloud, depois de atualizar os arquivos, faça um novo deploy/reinício do app para que as dependências sejam reinstaladas.

### Correção para arquivos IMERG/GOES com calendário juliano

A leitura dos HDF5 é feita com `decode_times=False`, pois os carimbos temporais dos produtos não são usados para extrair o campo espacial. Isso evita que o xarray tente interpretar o calendário `julian` durante a abertura do arquivo. `cftime` também permanece nas dependências como suporte adicional para outros arquivos com calendários não padrão.
