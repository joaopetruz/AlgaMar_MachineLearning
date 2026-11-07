import pandas as pd
import requests
import time

base = pd.read_csv("dados/dados_features_sazonal_sp_com_salinidade.csv")
locais = base[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
print(f"Total de localizações únicas: {len(locais)}")

TAMANHO_LOTE = 50
MAX_TENTATIVAS = 3
resultados = []

def buscar_lote(lats, lons, tentativa=1):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": "2000-01-01",
        "end_date": "2024-12-31",
        "daily": "precipitation_sum",
        "timezone": "America/Sao_Paulo"
    }
    resposta = requests.get(url, params=params, timeout=60)
    dados_json = resposta.json()

    if isinstance(dados_json, dict):
        dados_json = [dados_json]

    # Verificar se a resposta veio vazia (sinal de rate limit silencioso)
    primeiro_valido = next((d for d in dados_json if "daily" in d), None)
    if primeiro_valido is not None and len(primeiro_valido["daily"]["time"]) == 0:
        if tentativa < MAX_TENTATIVAS:
            espera = 10 * tentativa
            print(f"    Resposta vazia (rate limit?). Tentativa {tentativa}, esperando {espera}s...")
            time.sleep(espera)
            return buscar_lote(lats, lons, tentativa + 1)
        else:
            print(f"    Falhou após {MAX_TENTATIVAS} tentativas, pulando esse lote.")
            return None

    return dados_json


for inicio in range(0, len(locais), TAMANHO_LOTE):
    lote = locais.iloc[inicio:inicio + TAMANHO_LOTE]
    lats = ",".join(lote["latitude"].astype(str))
    lons = ",".join(lote["longitude"].astype(str))

    print(f"Buscando lote {inicio // TAMANHO_LOTE + 1} de {(len(locais) // TAMANHO_LOTE) + 1}...")

    dados_json = buscar_lote(lats, lons)

    if dados_json is None:
        continue

    for i, resultado_local in enumerate(dados_json):
        lat = lote.iloc[i]["latitude"]
        lon = lote.iloc[i]["longitude"]

        if "daily" not in resultado_local or len(resultado_local["daily"]["time"]) == 0:
            print(f"  Aviso: sem dados para {lat}, {lon}")
            continue

        df_local = pd.DataFrame({
            "latitude": lat,
            "longitude": lon,
            "time": resultado_local["daily"]["time"],
            "precipitacao_mm": resultado_local["daily"]["precipitation_sum"]
        })
        resultados.append(df_local)

    time.sleep(4)  # pausa maior entre lotes, para evitar rate limit

tabela_final = pd.concat(resultados, ignore_index=True)
tabela_final["time"] = pd.to_datetime(tabela_final["time"])
tabela_final["ano"] = tabela_final["time"].dt.year
tabela_final["mes"] = tabela_final["time"].dt.month

mensal = tabela_final.groupby(["latitude", "longitude", "ano", "mes"])["precipitacao_mm"].sum().reset_index()
mensal = mensal.rename(columns={"precipitacao_mm": "precipitacao_mensal_mm"})

mensal.to_csv("dados/precipitacao_2000_2024_sp.csv", index=False)
print(f"\nArquivo salvo: dados/precipitacao_2000_2024_sp.csv")
print(f"Total de linhas: {len(mensal)}")
print(f"Localizações únicas: {mensal[['latitude','longitude']].drop_duplicates().shape[0]}")