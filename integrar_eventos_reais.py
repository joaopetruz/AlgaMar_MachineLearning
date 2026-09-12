import pandas as pd
import numpy as np

# ===== 1. Carregar os eventos reais =====
eventos1 = pd.read_csv("dados/habdata_algas.csv", encoding="utf-8-sig")
eventos2 = pd.read_csv("dados/habdata_sao_paulo.csv", encoding="utf-8-sig")

# Padronizar colunas em comum
eventos1 = eventos1[["eventId", "anoOcorrencia", "mes", "cidade", "latitude", "longitude", "tiposDeAlga"]].copy()
eventos1["mes"] = eventos1["mes"].fillna("")

eventos2 = eventos2[["eventId", "anoOcorrencia", "cidade", "latitude", "longitude", "tiposDeAlga"]].copy()
eventos2["mes"] = ""  # esse arquivo não tem mês

eventos = pd.concat([eventos1, eventos2], ignore_index=True)

# Corrigir erro de sinal na latitude (evento com latitude positiva por engano)
eventos.loc[eventos["latitude"] > 0, "latitude"] = -eventos["latitude"]

# Remover eventos sem coordenada
eventos = eventos.dropna(subset=["latitude", "longitude"])

print(f"Total de eventos reais carregados: {len(eventos)}")
print(f"Período coberto pelos eventos: {eventos['anoOcorrencia'].min()} a {eventos['anoOcorrencia'].max()}")

# ===== 2. Filtrar só eventos dentro do período que temos dados (2016-2024) =====
eventos_no_periodo = eventos[(eventos["anoOcorrencia"] >= 2016) & (eventos["anoOcorrencia"] <= 2024)].copy()
print(f"\nEventos dentro do período com dados Copernicus (2016-2024): {len(eventos_no_periodo)}")
print(eventos_no_periodo[["eventId", "anoOcorrencia", "mes", "cidade", "latitude", "longitude"]])

# Mapa de nomes de mês para número
meses_map = {
    "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4, "Maio": 5, "Junho": 6,
    "Julho": 7, "Agosto": 8, "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12
}

def extrair_mes(valor):
    if not valor or pd.isna(valor):
        return None
    primeiro = str(valor).split("-")[0].split(" ")[0].strip()
    return meses_map.get(primeiro, None)

eventos_no_periodo["mes_num"] = eventos_no_periodo["mes"].apply(extrair_mes)
eventos_com_mes = eventos_no_periodo.dropna(subset=["mes_num"])
print(f"\nEventos com mês identificado (utilizáveis para o modelo mensal): {len(eventos_com_mes)}")
print(eventos_com_mes[["anoOcorrencia", "mes_num", "cidade", "latitude", "longitude"]])

# ===== 3. Carregar nossa base sazonal e cruzar por proximidade =====
base = pd.read_csv("dados/dados_features_sazonal_sp_completo.csv")

def achar_ponto_mais_proximo(lat_evento, lon_evento, ano_evento, mes_evento, base):
    candidatos = base[(base["ano"] == ano_evento) & (base["mes"] == mes_evento)]
    if len(candidatos) == 0:
        return None
    distancias = np.sqrt((candidatos["latitude"] - lat_evento)**2 + (candidatos["longitude"] - lon_evento)**2)
    return candidatos.loc[distancias.idxmin()].name

indices_confirmados = []
for _, evento in eventos_com_mes.iterrows():
    idx = achar_ponto_mais_proximo(
        evento["latitude"], evento["longitude"],
        int(evento["anoOcorrencia"]), int(evento["mes_num"]),
        base
    )
    if idx is not None:
        indices_confirmados.append(idx)

print(f"\nEventos cruzados com sucesso na base de dados: {len(indices_confirmados)}")

# ===== 4. Marcar esses pontos como floração confirmada (sobrepõe o proxy) =====
base["floracao_confirmada_real"] = 0
base.loc[indices_confirmados, "floracao_confirmada_real"] = 1

print(f"\nPontos com floração REAL confirmada na base: {base['floracao_confirmada_real'].sum()}")

base.to_csv("dados/dados_features_sazonal_sp_com_eventos_reais.csv", index=False)
print("\nArquivo salvo: dados/dados_features_sazonal_sp_com_eventos_reais.csv")