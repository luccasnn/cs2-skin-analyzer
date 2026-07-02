
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import time
import json

st.set_page_config(page_title="CS2 Skin Analyzer", page_icon="🔫", layout="wide")

COOKIES = {
    "sessionid": st.secrets.get("sessionid", ""),
    "steamLoginSecure": st.secrets.get("steamLoginSecure", "")
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

DESGASTES = {
    "Nova de Fábrica": "Factory New",
    "Pouco Usada": "Minimal Wear",
    "Testada em Campo": "Field-Tested",
    "Bastante Usada": "Well-Worn",
    "Veterana de Guerra": "Battle-Scarred"
}

# inicializa watchlist e alertas no session state
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "alertas" not in st.session_state:
    st.session_state.alertas = []

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

# navegação
pagina = st.sidebar.radio("Navegação", ["Analisar Skin", "Watchlist", "Alertas", "Scanner de Mercado"])

# ============================================================
# PÁGINA: ANALISAR SKIN
# ============================================================
if pagina == "Analisar Skin":
    st.title("CS2 Skin Market Analyzer")
    st.markdown("Análise completa de preço, tendência, volume e previsão para qualquer skin do CS2.")

    col1, col2 = st.columns([3, 1])
    with col1:
        arma_skin = st.text_input("Nome da skin sem desgaste", placeholder="Ex: AWP | Asiimov")
    with col2:
        desgaste_pt = st.selectbox("Desgaste", list(DESGASTES.keys()))

    skin_input = f"{arma_skin.strip()} ({DESGASTES[desgaste_pt]})" if arma_skin else ""

    if skin_input:
        st.caption(f"Nome enviado para a API: `{skin_input}`")

    col_btn1, col_btn2 = st.columns([1, 1])
    analisar = col_btn1.button("Analisar", type="primary")
    adicionar_watchlist = col_btn2.button("Adicionar à Watchlist")

    if adicionar_watchlist and skin_input:
        if skin_input not in st.session_state.watchlist:
            st.session_state.watchlist.append(skin_input)
            st.success(f"{skin_input} adicionada à Watchlist!")
        else:
            st.info("Skin já está na Watchlist.")

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

            arma = arma_skin.split("|")[0].strip() if "|" in arma_skin else arma_skin

            with st.spinner("buscando skins similares..."):
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
                    f"<span style='color:{cor_volume}'><b>{class_volume}</b></span> "
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

# ============================================================
# PÁGINA: WATCHLIST
# ============================================================
elif pagina == "Watchlist":
    st.title("Watchlist")
    st.markdown("Acompanhe todas as suas skins favoritas em um só lugar.")

    if not st.session_state.watchlist:
        st.info("Sua Watchlist está vazia. Adicione skins na página Analisar Skin.")
    else:
        remover = st.selectbox("Remover skin da Watchlist", [""] + st.session_state.watchlist)
        if st.button("Remover") and remover:
            st.session_state.watchlist.remove(remover)
            st.rerun()

        st.divider()

        for skin in st.session_state.watchlist:
            with st.spinner(f"carregando {skin}..."):
                df = buscar_historico(skin)
            if df is None:
                st.error(f"Erro ao carregar {skin}")
                continue

            preco_atual = df["preco"].iloc[-1]
            tendencia = calcular_tendencia(df)
            vs_historico = ((preco_atual - df["preco"].mean()) / df["preco"].mean()) * 100

            if tendencia > 0.5:
                status = "⬆ Subindo"
                cor = "green"
            elif tendencia < -0.5:
                status = "⬇ Caindo"
                cor = "red"
            else:
                status = "➡ Estável"
                cor = "orange"

            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                col1.markdown(f"**{skin}**")
                col2.metric("Preço", f"R${preco_atual:.2f}")
                col3.markdown(f"<span style='color:{cor}'>{status}</span>", unsafe_allow_html=True)
                col4.metric("vs histórico", f"{vs_historico:+.1f}%")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.tail(30)["data"], y=df.tail(30)["preco"],
                                         line=dict(color="#5c9ee0", width=1.5)))
                fig.update_layout(height=120, margin=dict(l=0, r=0, t=0, b=0),
                                  xaxis=dict(showgrid=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, showticklabels=False),
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                st.divider()
            time.sleep(0.5)

# ============================================================
# PÁGINA: ALERTAS
# ============================================================
elif pagina == "Alertas":
    st.title("Alertas de Preço")
    st.markdown("Defina um preço alvo e veja se a skin já atingiu o valor.")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        alerta_skin = st.text_input("Nome completo da skin", placeholder="Ex: AWP | Asiimov (Field-Tested)")
    with col2:
        alerta_tipo = st.selectbox("Alertar quando", ["Preço abaixo de", "Preço acima de"])
    with col3:
        alerta_preco = st.number_input("Preço alvo (R$)", min_value=0.0, value=100.0, step=10.0)

    if st.button("Adicionar Alerta", type="primary"):
        if alerta_skin:
            st.session_state.alertas.append({
                "skin": alerta_skin,
                "tipo": alerta_tipo,
                "preco": alerta_preco
            })
            st.success("Alerta adicionado!")

    st.divider()

    if not st.session_state.alertas:
        st.info("Nenhum alerta configurado.")
    else:
        st.subheader("Verificando alertas...")
        for i, alerta in enumerate(st.session_state.alertas):
            df = buscar_historico(alerta["skin"])
            if df is None:
                continue

            preco_atual = df["preco"].iloc[-1]
            disparado = False

            if alerta["tipo"] == "Preço abaixo de" and preco_atual < alerta["preco"]:
                disparado = True
            elif alerta["tipo"] == "Preço acima de" and preco_atual > alerta["preco"]:
                disparado = True

            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            col1.markdown(f"**{alerta['skin']}**")
            col2.markdown(f"R${preco_atual:.2f} atual")
            col3.markdown(f"{alerta['tipo']} R${alerta['preco']:.2f}")

            if disparado:
                col4.success("DISPARADO!")
            else:
                col4.info("Aguardando")

            if st.button(f"Remover alerta {i+1}", key=f"rem_{i}"):
                st.session_state.alertas.pop(i)
                st.rerun()

            time.sleep(0.3)

# ============================================================
# PÁGINA: SCANNER DE MERCADO
# ============================================================
elif pagina == "Scanner de Mercado":
    st.title("Scanner de Mercado")
    st.markdown("Detecta skins com volume anormalmente alto nos últimos dias — sinal de movimentação no mercado.")

    arma_scan = st.text_input("Arma para escanear", placeholder="Ex: AK-47")
    escanear = st.button("Escanear", type="primary")

    if escanear and arma_scan:
        with st.spinner(f"buscando skins de {arma_scan}..."):
            skins_encontradas = buscar_skins_comparacao(arma_scan)

        if not skins_encontradas:
            st.error("Nenhuma skin encontrada para essa arma.")
        else:
            resultados = []
            for skin in skins_encontradas:
                df = buscar_historico(skin)
                if df is None:
                    continue

                volume_recente = df.tail(7)["volume"].mean()
                volume_historico = df.tail(90)["volume"].mean()
                variacao_volume = ((volume_recente - volume_historico) / volume_historico * 100) if volume_historico > 0 else 0
                tendencia = calcular_tendencia(df)
                preco_atual = df["preco"].iloc[-1]

                resultados.append({
                    "Skin": skin,
                    "Preço Atual": f"R${preco_atual:.2f}",
                    "Volume 7d": f"{volume_recente:.1f}/dia",
                    "Volume 90d": f"{volume_historico:.1f}/dia",
                    "Variação Volume": f"{variacao_volume:+.0f}%",
                    "Tendência R$/dia": f"{tendencia:+.2f}"
                })
                time.sleep(0.5)

            if resultados:
                df_result = pd.DataFrame(resultados)
                st.dataframe(df_result, use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("Skins com maior variação de volume")
                df_result["_var"] = df_result["Variação Volume"].str.replace("%","").str.replace("+","").astype(float)
                df_sorted = df_result.sort_values("_var", ascending=False)

                fig = px.bar(df_sorted, x="Skin", y="_var",
                             color="_var", color_continuous_scale=["#e05c5c", "#e0a05c", "#5ce0a8"],
                             labels={"_var": "Variação de Volume (%)", "Skin": ""},)
                fig.update_layout(height=350, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
