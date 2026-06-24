

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import time

st.set_page_config(page_title="CS2 Skin Analyzer", page_icon="🔫", layout="wide")

# autenticação steam
COOKIES = {
    'sessionid': '497e73e3a400dd775628e45e',
    'steamLoginSecure': '76561198176540254%7C%7CeyAidHlwIjogIkpXVCIsICJhbGciOiAiRWREU0EiIH0.eyAiaXNzIjogInI6MDAxNF8yN0UzNzc2OF9CQzk1MiIsICJzdWIiOiAiNzY1NjExOTgxNzY1NDAyNTQiLCAiYXVkIjogWyAid2ViOmNvbW11bml0eSIgXSwgImV4cCI6IDE3ODIyNDk0ODcsICJuYmYiOiAxNzczNTIxMzY5LCAiaWF0IjogMTc4MjE2MTM2OSwgImp0aSI6ICIwMDAyXzI4NUI3MTlFXzAwMkI0IiwgIm9hdCI6IDE3NzQ0OTQ5OTcsICJydF9leHAiOiAxNzkyNjk1OTIxLCAicGVyIjogMCwgImlwX3N1YmplY3QiOiAiMTg3LjUzLjc2LjY2IiwgImlwX2NvbmZpcm1lciI6ICIxODcuNTMuNzYuNjYiIH0.Pz-TdPaiDUVN3UEw5R2obo4oj79WSGRmrZ5KhInHtHvyYmNsNh9S39MmmlR9V1-_G4lTTOiIO0D4Mr11u17GAQ'
}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

@st.cache_data(ttl=3600)
def buscar_historico(skin_name):
    url = "https://steamcommunity.com/market/pricehistory/"
    params = {"appid": 730, "market_hash_name": skin_name}
    resposta = requests.get(url, params=params, headers=HEADERS, cookies=COOKIES)
    if resposta.status_code != 200 or 'prices' not in resposta.json():
        return None
    dados = resposta.json()['prices']
    df = pd.DataFrame(dados, columns=['data', 'preco', 'volume'])
    df['data'] = pd.to_datetime(df['data'].str[:11], format='%b %d %Y')
    df['preco'] = df['preco'].astype(float)
    df['volume'] = df['volume'].astype(int)
    return df

@st.cache_data(ttl=3600)
def buscar_skins_arma(arma):
    skins_arma = {
        'AK-47': [
            'AK-47 | Redline (Field-Tested)',
            'AK-47 | Fire Serpent (Field-Tested)',
            'AK-47 | Asiimov (Field-Tested)',
            'AK-47 | Vulcan (Field-Tested)',
            'AK-47 | Neon Rider (Field-Tested)'
        ],
        'AWP': [
            'AWP | Asiimov (Field-Tested)',
            'AWP | Dragon Lore (Field-Tested)',
            'AWP | Medusa (Field-Tested)',
            'AWP | Fade (Factory New)',
            'AWP | Hyper Beast (Field-Tested)'
        ],
        'M4A4': [
            'M4A4 | Howl (Field-Tested)',
            'M4A4 | Asiimov (Field-Tested)',
            'M4A4 | Neo-Noir (Field-Tested)',
            'M4A4 | The Emperor (Field-Tested)',
            'M4A4 | Radiation Hazard (Field-Tested)'
        ]
    }
    return skins_arma.get(arma, [])

def calcular_tendencia(df, dias=90):
    recente = df.tail(dias).copy()
    recente['dias'] = range(len(recente))
    modelo = LinearRegression()
    modelo.fit(recente[['dias']], recente['preco'])
    return modelo.coef_[0]

def classificar_volume(volume_skin, volumes_comparacao):
    p33 = np.percentile(volumes_comparacao, 33)
    p66 = np.percentile(volumes_comparacao, 66)
    if volume_skin <= p33:
        return "Baixa", "#e05c5c"
    elif volume_skin <= p66:
        return "Média", "#e0a05c"
    else:
        return "Alta", "#5ce0a8"

def prever_preco(df, dias=30):
    df_prophet = df[['data', 'preco']].copy()
    df_prophet.columns = ['ds', 'y']
    modelo = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, changepoint_prior_scale=0.05)
    modelo.fit(df_prophet)
    futuro = modelo.make_future_dataframe(periods=dias)
    return modelo.predict(futuro)

# interface
st.title("CS2 Skin Market Analyzer")
st.markdown("Análise completa de preço, tendência, volume e previsão para qualquer skin do CS2.")

col1, col2 = st.columns([3, 1])
with col1:
    skin_input = st.text_input("Nome da skin (exatamente como no Steam Market)",
                                placeholder="Ex: AK-47 | Redline (Field-Tested)")
with col2:
    arma_input = st.selectbox("Arma para comparação de volume",
                               ['AK-47', 'AWP', 'M4A4'])

analisar = st.button("Analisar", type="primary")

if analisar and skin_input:
    with st.spinner("coletando dados do Steam Market..."):
        df = buscar_historico(skin_input)

    if df is None:
        st.error("Skin não encontrada. Verifique o nome exato no Steam Market.")
    else:
        # métricas principais
        tendencia = calcular_tendencia(df)
        preco_atual = df['preco'].iloc[-1]
        media_historica = df['preco'].mean()
        media_90d = df.tail(90)['preco'].mean()
        volume_medio = df.tail(90)['volume'].mean()
        volatilidade = df.tail(90)['preco'].std()
        maior_queda = df.tail(30)['preco'].pct_change().min() * 100

        if tendencia > 0.5:
            status_tend = "Subindo"
            cor_tend = "#5ce0a8"
        elif tendencia < -0.5:
            status_tend = "Caindo"
            cor_tend = "#e05c5c"
        else:
            status_tend = "Estável"
            cor_tend = "#e0a05c"

        vs_historico = ((preco_atual - media_historica) / media_historica) * 100

        # busca volumes de comparação
        skins_comparacao = buscar_skins_arma(arma_input)
        volumes_comparacao = []
        for s in skins_comparacao:
            df_temp = buscar_historico(s)
            if df_temp is not None:
                volumes_comparacao.append(df_temp.tail(90)['volume'].mean())
            time.sleep(1)

        if volumes_comparacao:
            class_volume, cor_volume = classificar_volume(volume_medio, volumes_comparacao)
        else:
            class_volume, cor_volume = "N/A", "#5c9ee0"

        st.divider()

        # métricas
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Preço Atual", f"R${preco_atual:.2f}")
        col2.metric("Tendência", status_tend, f"R${tendencia:.2f}/dia")
        col3.metric("vs Média Histórica", f"{vs_historico:+.1f}%")
        col4.metric("Volatilidade 90d", f"R${volatilidade:.2f}")
        col5.metric("Pior queda 30d", f"{maior_queda:.1f}%")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            # histórico completo
            st.subheader("Histórico de preço")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['data'], y=df['preco'],
                                     name='preço', line=dict(color='#5c9ee0', width=1)))
            ma30 = df['preco'].rolling(30).mean()
            fig.add_trace(go.Scatter(x=df['data'], y=ma30,
                                     name='média 30d', line=dict(color='orange', dash='dash')))
            fig.add_hline(y=media_historica, line_dash="dot",
                         annotation_text="média histórica", line_color="gray")
            fig.update_layout(xaxis_title="data", yaxis_title="preço (R$)", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # previsão
            st.subheader("Previsão — próximos 30 dias")
            with st.spinner("calculando previsão..."):
                previsao = prever_preco(df)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.tail(90)['data'], y=df.tail(90)['preco'],
                                      name='real', line=dict(color='#5c9ee0')))
            futuro = previsao.tail(30)
            fig2.add_trace(go.Scatter(x=futuro['ds'], y=futuro['yhat'],
                                      name='previsão', line=dict(color='#e05c5c', dash='dash')))
            fig2.add_trace(go.Scatter(x=pd.concat([futuro['ds'], futuro['ds'][::-1]]),
                                      y=pd.concat([futuro['yhat_upper'], futuro['yhat_lower'][::-1]]),
                                      fill='toself', fillcolor='rgba(224,92,92,0.1)',
                                      line=dict(color='rgba(255,255,255,0)'),
                                      name='intervalo de confiança'))
            fig2.update_layout(xaxis_title="data", yaxis_title="preço (R$)", height=350)
            st.plotly_chart(fig2, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # volume de vendas
            st.subheader("Volume de vendas — últimos 90 dias")
            df_90 = df.tail(90)
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=df_90['data'], y=df_90['volume'],
                                  marker_color='#5ce0a8', name='vendas/dia'))
            fig3.update_layout(xaxis_title="data", yaxis_title="vendas por dia", height=300)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(f"Liquidez comparada a outras skins de **{arma_input}**: "
                       f"<span style='color:{cor_volume}'>**{class_volume}**</span> "
                       f"({volume_medio:.1f} vendas/dia)", unsafe_allow_html=True)

        with col2:
            # análise de risco
            st.subheader("Análise de risco")

            df_90 = df.tail(90).copy()
            df_90['retorno'] = df_90['preco'].pct_change() * 100

            fig4 = px.histogram(df_90, x='retorno', nbins=30,
                               color_discrete_sequence=['#a05ce0'])
            fig4.add_vline(x=0, line_dash="dash", line_color="white")
            fig4.update_layout(xaxis_title="variação diária (%)",
                              yaxis_title="frequência", height=300)
            st.plotly_chart(fig4, use_container_width=True)

            prob_subir = (df_90['retorno'] > 0).mean() * 100
            st.markdown(f"Nos últimos 90 dias a skin subiu em **{prob_subir:.0f}%** dos dias")
            st.markdown(f"Maior queda em 30 dias: **{maior_queda:.1f}%**")
            st.markdown(f"Volatilidade: **R${volatilidade:.2f}**")

        # sazonalidade
        st.divider()
        st.subheader("Sazonalidade — preço médio por mês")
        df['mes'] = df['data'].dt.month
        sazon = df.groupby('mes')['preco'].mean()
        meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        fig5 = px.bar(x=[meses[m-1] for m in sazon.index], y=sazon.values,
                     color_discrete_sequence=['#5c9ee0'])
        fig5.update_layout(xaxis_title="mês", yaxis_title="preço médio (R$)", height=300)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("Picos de preço em certos meses podem indicar sazonalidade ligada a Majors e atualizações do jogo.")
