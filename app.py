
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

COOKIES = {
    "sessionid": st.secrets.get("sessionid", "497e73e3a400dd775628e45e"),
    "steamLoginSecure": st.secrets.get("steamLoginSecure", "76561198176540254%7C%7CeyAidHlwIjogIkpXVCIsICJhbGciOiAiRWREU0EiIH0.eyAiaXNzIjogInI6MDAxNF8yN0UzNzc2OF9CQzk1MiIsICJzdWIiOiAiNzY1NjExOTgxNzY1NDAyNTQiLCAiYXVkIjogWyAid2ViOmNvbW11bml0eSIgXSwgImV4cCI6IDE3ODIzNTg1OTcsICJuYmYiOiAxNzczNjMwNjMxLCAiaWF0IjogMTc4MjI3MDYzMSwgImp0aSI6ICIwMDAyXzI4NjRBRTlGXzU2NTE4IiwgIm9hdCI6IDE3NzQ0OTQ5OTcsICJydF9leHAiOiAxNzkyNjk1OTIxLCAicGVyIjogMCwgImlwX3N1YmplY3QiOiAiMTg3LjUzLjc2LjY2IiwgImlwX2NvbmZpcm1lciI6ICIxODcuNTMuNzYuNjYiIH0.SzY6NcyNBxxWCuNqbBwVpLxG0GvoBR8k8WTNJLH2ArkhQxsaaXVfprCpk7I1sHtGDnzGU3T8M1FKIzGjW_6BBg")
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

DESGASTES = {
    "Nova de Fábrica": "Factory New",
    "Pouco Usada": "Minimal Wear",
    "Testada em Campo": "Field-Tested",
    "Bastante Usada": "Well-Worn",
    "Veterana de Guerra": "Battle-Scarred"
}

@st.cache_data(ttl=3600)
def buscar_historico(skin_name):
    url = "https://steamcommunity.com/market/pricehistory/"
    params = {"appid": 730, "market_hash_name": skin_name}
    resposta = requests.get(url, params=params, headers=HEADERS, cookies=COOKIES)
    if resposta.status_code != 200 or "prices" not in resposta.json():
        return None
    dados = resposta.json()["prices"]
    df = pd.DataFrame(dados, columns=["data", "preco", "volume"])
    df["data"] = pd.to_datetime(df["data"].str[:11], format="%b %d %Y")
    df["preco"] = df["preco"].astype(float)
    df["volume"] = df["volume"].astype(int)
    return df

@st.cache_data(ttl=3600)
def buscar_skins_comparacao(arma):
    url = "https://steamcommunity.com/market/search/render/"
    params = {
        "appid": 730,
        "q": arma,
        "count": 10,
        "search_descriptions": 0,
        "sort_column": "popular",
        "sort_dir": "desc"
    }
    resposta = requests.get(url, params=params, headers=HEADERS, cookies=COOKIES)
    if not resposta.json().get("success"):
        return []
    skins = []
    for item in resposta.json().get("results", []):
        nome = item.get("name", "")
        if arma in nome:
            skins.append(nome)
    return skins[:5]

def calcular_tendencia(df, dias=90):
    recente = df.tail(dias).copy()
    recente["dias"] = range(len(recente))
    modelo = LinearRegression()
    modelo.fit(recente[["dias"]], recente["preco"])
    return modelo.coef_[0]

def classificar_volume(volume_skin, volumes_comparacao):
    if not volumes_comparacao:
        return "N/A", "#5c9ee0"
    p33 = np.percentile(volumes_comparacao, 33)
    p66 = np.percentile(volumes_comparacao, 66)
    if volume_skin <= p33:
        return "Baixa", "#e05c5c"
    elif volume_skin <= p66:
        return "Média", "#e0a05c"
    else:
        return "Alta", "#5ce0a8"

def prever_preco(df, dias=30):
    df_prophet = df[["data", "preco"]].copy()
    df_prophet.columns = ["ds", "y"]
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
    arma_skin = st.text_input(
        "Nome da skin sem desgaste",
        placeholder="Ex: AWP | Asiimov"
    )
with col2:
    desgaste_pt = st.selectbox("Desgaste", list(DESGASTES.keys()))

skin_input = f"{arma_skin} ({DESGASTES[desgaste_pt]})" if arma_skin else ""

if skin_input:
    st.caption(f"Nome enviado para a API: `{skin_input}`")

analisar = st.button("Analisar", type="primary")

if analisar and skin_input:
    with st.spinner("coletando dados do Steam Market..."):
        df = buscar_historico(skin_input)

    if df is None:
        st.error("Skin não encontrada. Verifique o nome exato no Steam Market.")
    else:
        tendencia = calcular_tendencia(df)
        preco_atual = df["preco"].iloc[-1]
        media_historica = df["preco"].mean()
        volume_medio = df.tail(90)["volume"].mean()
        volatilidade = df.tail(90)["preco"].std()
        maior_queda = df.tail(30)["preco"].pct_change().min() * 100
        vs_historico = ((preco_atual - media_historica) / media_historica) * 100

        if tendencia > 0.5:
            status_tend = "Subindo"
        elif tendencia < -0.5:
            status_tend = "Caindo"
        else:
            status_tend = "Estável"

        # detecta a arma pelo nome
        arma = arma_skin.split("|")[0].strip() if "|" in arma_skin else arma_skin

        with st.spinner("buscando skins similares para comparação..."):
            skins_comp = buscar_skins_comparacao(arma)
            volumes_comp = []
            for s in skins_comp:
                if s != skin_input:
                    df_temp = buscar_historico(s)
                    if df_temp is not None:
                        volumes_comp.append(df_temp.tail(90)["volume"].mean())
                    time.sleep(0.5)

        class_volume, cor_volume = classificar_volume(volume_medio, volumes_comp)

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
            st.subheader("Histórico de preço")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["data"], y=df["preco"],
                                     name="preço", line=dict(color="#5c9ee0", width=1)))
            ma30 = df["preco"].rolling(30).mean()
            fig.add_trace(go.Scatter(x=df["data"], y=ma30,
                                     name="média 30d", line=dict(color="orange", dash="dash")))
            fig.add_hline(y=media_historica, line_dash="dot",
                          annotation_text="média histórica", line_color="gray")
            fig.update_layout(xaxis_title="data", yaxis_title="preço (R$)", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Previsão — próximos 30 dias")
            with st.spinner("calculando previsão..."):
                previsao = prever_preco(df)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.tail(90)["data"], y=df.tail(90)["preco"],
                                      name="real", line=dict(color="#5c9ee0")))
            futuro = previsao.tail(30)
            fig2.add_trace(go.Scatter(x=futuro["ds"], y=futuro["yhat"],
                                      name="previsão", line=dict(color="#e05c5c", dash="dash")))
            fig2.add_trace(go.Scatter(
                x=pd.concat([futuro["ds"], futuro["ds"][::-1]]),
                y=pd.concat([futuro["yhat_upper"], futuro["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(224,92,92,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name="intervalo de confiança"
            ))
            fig2.update_layout(xaxis_title="data", yaxis_title="preço (R$)", height=350)
            st.plotly_chart(fig2, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Volume de vendas — últimos 90 dias")
            df_90 = df.tail(90)
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=df_90["data"], y=df_90["volume"],
                                  marker_color="#5ce0a8", name="vendas/dia"))
            fig3.update_layout(xaxis_title="data", yaxis_title="vendas por dia", height=300)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(
                f"Liquidez comparada a outras skins de **{arma}**: "
                f"<span style='color:{cor_volume}'>**{class_volume}**</span> "
                f"({volume_medio:.1f} vendas/dia)",
                unsafe_allow_html=True
            )

        with col2:
            st.subheader("Análise de risco")
            df_90 = df.tail(90).copy()
            df_90["retorno"] = df_90["preco"].pct_change() * 100
            fig4 = px.histogram(df_90, x="retorno", nbins=30,
                                color_discrete_sequence=["#a05ce0"])
            fig4.add_vline(x=0, line_dash="dash", line_color="white")
            fig4.update_layout(xaxis_title="variação diária (%)",
                               yaxis_title="frequência", height=300)
            st.plotly_chart(fig4, use_container_width=True)
            prob_subir = (df_90["retorno"] > 0).mean() * 100
            st.markdown(f"Nos últimos 90 dias a skin subiu em **{prob_subir:.0f}%** dos dias")
            st.markdown(f"Maior queda em 30 dias: **{maior_queda:.1f}%**")
            st.markdown(f"Volatilidade: **R${volatilidade:.2f}**")

        st.divider()
        st.subheader("Sazonalidade — preço médio por mês")
        df["mes"] = df["data"].dt.month
        sazon = df.groupby("mes")["preco"].mean()
        meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                 "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
        fig5 = px.bar(x=[meses[m-1] for m in sazon.index], y=sazon.values,
                      color_discrete_sequence=["#5c9ee0"])
        fig5.update_layout(xaxis_title="mês", yaxis_title="preço médio (R$)", height=300)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("Picos em certos meses podem indicar sazonalidade ligada a Majors e atualizações.")
