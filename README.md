# Site de precipitação Petrobras

Aplicação web em **Streamlit** convertida a partir do seletor desktop original.

## O que a versão web mantém

- 40 unidades Petrobras embutidas no código-fonte original.
- IMERG Early Run para precipitação observada quase em tempo real.
- GOES-19 ABI-L2-RRQPEF como complemento de alta frequência.
- Nowcast por extrapolação lagrangiana do campo mais recente.
- Previsão pontual via Open-Meteo.
- Busca e seleção de unidade.
- Raio de extração e escolha de `max` ou `mean`.
- Linha do tempo para navegar entre observado, GOES e previsto.
- Tabela unificada.
- Log de falhas e etapas.
- Download dos resultados em XLSX e do mapa em PNG.

## Rodar localmente

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
streamlit run app.py
```

Abra o endereço mostrado pelo Streamlit.

## NASA PPS

O IMERG Early Run do código original usa autenticação do NASA PPS.

Cadastre o e-mail em:

https://registration.pps.eosdis.nasa.gov/registration/

Na aplicação, informe esse e-mail no campo lateral **E-mail NASA PPS (NRT)**.

### Para publicar sem deixar o e-mail no código

No Streamlit Community Cloud, em **Settings > Secrets**, coloque:

```toml
EMAIL_PPS = "seu-email-cadastrado-no-pps@dominio.com"
```

O campo já aparecerá preenchido quando o site iniciar.

## Deploy no Streamlit Community Cloud

1. Crie um repositório no GitHub.
2. Envie `app.py`, `requirements.txt` e este `README.md`.
3. No Streamlit Community Cloud, selecione o repositório e o arquivo `app.py`.
4. Configure `EMAIL_PPS` em **Secrets**.
5. Publique.

## Observações técnicas

O código científico principal foi preservado. A camada Tkinter foi substituída por Streamlit e o mapa passou a ser Plotly.

O **PySTEPS** continua sendo detectado opcionalmente. Caso esteja instalado no ambiente, o movimento usa Lucas-Kanade; caso contrário, entra o fallback por correlação de fase já presente no código original.

A pasta `dados_imerg/` é criada automaticamente e funciona como cache local. Em hospedagens efêmeras, o cache pode ser perdido quando a aplicação é reiniciada.

O nowcast segue sendo extrapolação do campo existente e não é uma previsão numérica de longo alcance.
