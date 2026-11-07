"""Contrato de todas as features do modelo de previsão (gera contrato/features_ml.csv).

Momento: "t0" = usa dados até o mês de emissão; "alvo(<=t0)" = descreve o mês-alvo, mas só
com anos anteriores (todos <= t0); "fixo" = não depende do tempo.
"""
from pathlib import Path

import pandas as pd

API = "api (Copernicus: CHL L4 4 km / OSTIA SST / SSS multi-obs; MY ou NRT)"
HIST = "historical_dataset (painel 2000–último mês)"
FREQ = "mensal (mês civil UTC)"
RES = "grade CHL L4 4 km (SST e SSS interpoladas bilinearmente)"

_F = [
    # nome, unidade, origem, grupo, momento, descrição, ausentes
    ("h", "meses", "derivada", "base", "fixo", "horizonte de previsão (1..12)", "nunca ausente"),
    ("mes_emissao", "1..12", "derivada", "base", "t0", "mês do calendário de t0", "nunca ausente"),
    ("mes_alvo", "1..12", "derivada", "base", "fixo", "mês do calendário do mês previsto t0+h", "nunca ausente"),
    ("latitude", "graus norte", "grade CHL", "base", "fixo", "ponto (WGS84)", "nunca ausente"),
    ("longitude", "graus leste", "grade CHL", "base", "fixo", "ponto (WGS84)", "nunca ausente"),
    ("chl_media_t0", "mg m-3", API, "api", "t0", "clorofila média no mês t0", "NaN → XGBoost trata; linha de treino exige t0 observado"),
    ("chl_max_t0", "mg m-3", API, "api", "t0", "clorofila máxima diária no mês t0", "NaN nativo"),
    ("sst_t0", "degC", API, "api", "t0", "temperatura da superfície média no mês t0", "NaN nativo"),
    ("sss_t0", "PSU", API, "api", "t0", "salinidade da superfície média no mês t0 (costa mascarada pelo satélite)", "NaN nativo, nunca imputado"),
    ("chl_media_t0m1", "mg m-3", API, "api", "t0", "clorofila média em t0-1", "NaN se o mês faltar"),
    ("chl_media_t0m2", "mg m-3", API, "api", "t0", "clorofila média em t0-2", "NaN se o mês faltar"),
    ("sst_t0m1", "degC", API, "api", "t0", "SST em t0-1", "NaN se o mês faltar"),
    ("sst_t0m2", "degC", API, "api", "t0", "SST em t0-2", "NaN se o mês faltar"),
    ("sss_t0m1", "PSU", API, "api", "t0", "SSS em t0-1", "NaN se o mês faltar"),
    ("chl_media_3m", "mg m-3", API, "api", "t0", "média t0-2..t0 (mín. 2 meses)", "NaN se < 2 meses"),
    ("sst_3m", "degC", API, "api", "t0", "média t0-2..t0 (mín. 2 meses)", "NaN se < 2 meses"),
    ("chl_media_36m", "mg m-3", API, "api", "t0", "média t0-35..t0 (mín. 12 meses)", "NaN se < 12 meses"),
    ("chl_media_anom_t0", "mg m-3", API, "api", "t0", "clorofila média em t0 menos a climatologia do mesmo mês nos anos anteriores", "NaN sem histórico"),
    ("chl_max_anom_t0", "mg m-3", API, "api", "t0", "idem para a clorofila máxima", "NaN sem histórico"),
    ("sst_anom_t0", "degC", API, "api", "t0", "anomalia de SST em t0 (anos anteriores)", "NaN sem histórico"),
    ("sss_anom_t0", "PSU", API, "api", "t0", "anomalia de SSS em t0 (anos anteriores)", "NaN sem histórico"),
    ("clim_chl_media_alvo", "mg m-3", HIST, "historico", "alvo(<=t0)", "climatologia da clorofila média no mês-alvo (anos anteriores)", "NaN sem histórico"),
    ("clim_chl_max_alvo", "mg m-3", HIST, "historico", "alvo(<=t0)", "climatologia da clorofila máxima no mês-alvo", "NaN sem histórico"),
    ("freq_floracao_alvo", "0..1", HIST, "historico", "alvo(<=t0)", "fração dos anos anteriores com floração no mês-alvo", "NaN sem histórico"),
    ("n_anos_alvo", "anos", HIST, "historico", "alvo(<=t0)", "número de anos anteriores com dado no mês-alvo", "0 sem histórico"),
    ("clim_sst_alvo", "degC", HIST, "historico", "alvo(<=t0)", "climatologia de SST no mês-alvo", "NaN sem histórico"),
    ("chl_max_alvo_ano_anterior", "mg m-3", HIST, "historico", "alvo(<=t0)", "clorofila máxima no mesmo mês do ano anterior (t0+h-12)", "NaN se o mês faltar"),
    ("floracao_alvo_ano_anterior", "0/1", HIST, "historico", "alvo(<=t0)", "houve floração no mesmo mês do ano anterior", "NaN se o mês faltar"),
]


def tabela():
    linhas = []
    for nome, unidade, origem, grupo, momento, desc, ausentes in _F:
        linhas.append({"nome": nome, "tipo": "int" if nome in ("h", "mes_emissao", "mes_alvo") else "float",
                       "unidade": unidade, "origem": origem, "grupo_flag": grupo, "momento": momento,
                       "descricao": desc, "frequencia": FREQ if grupo != "base" else "-",
                       "resolucao_espacial": RES, "obrigatoria": "sim" if grupo == "base" else "não", "ausentes": ausentes})
    linhas.append({"nome": "ns_<nome>_t0", "tipo": "float", "unidade": "ver contrato/features_fisicas.csv",
                   "origem": "navier_stokes (feature store)", "grupo_flag": "fisica", "momento": "t0",
                   "descricao": "estado físico no mês de emissão, um por feature do conjunto escolhido",
                   "frequencia": FREQ, "resolucao_espacial": "nativa → ponto ML por vizinho (≤ dist_max_km)",
                   "obrigatoria": "não", "ausentes": "NaN nativo; pontos NAO_USAR viram NaN"})
    return pd.DataFrame(linhas)


def grava(caminho=Path(__file__).resolve().parents[2] / "contrato" / "features_ml.csv"):
    tabela().to_csv(caminho, index=False)
    return caminho


if __name__ == "__main__":
    print(grava())
