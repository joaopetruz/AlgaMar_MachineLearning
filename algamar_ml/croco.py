"""Entrada das features físicas do CROCO (Navier-Stokes, D2 1,5 km) no dataset do ML.

O arquivo do CROCO segue o contrato 13.0 do cfd_navier_stokes (contrato_croco/):
uma linha por (latitude, longitude, ano, mes) nos pontos da grade CHL do ML, colunas
<feature>_mean e a flag validade_espacial (D2_VALIDO, NAO_VALIDADO_COSTEIRO,
NAO_USAR_ESTUARIO_CANAL).

Modos temporais (nunca misturar numa mesma métrica):
- NOWCAST: valor do próprio mês t;
- FORECAST_SAZONAL: valor de t-12 meses, coluna <col>__lag12 (padrão, como o baseline);
- FORECAST_1M: valor de t-1 mês, coluna <col>__lag1.
"""
import json
from pathlib import Path

import pandas as pd

from algamar_ml.features import CHAVE, lag_por_data

CONTRATO = Path(__file__).resolve().parent.parent / "contrato_croco"
SUFIXO_MODO = {"NOWCAST": "", "FORECAST_SAZONAL": "__lag12", "FORECAST_1M": "__lag1"}
LAG_MODO = {"NOWCAST": 0, "FORECAST_SAZONAL": 12, "FORECAST_1M": 1}
CASAS_CHAVE = 5
NAO_USAR = "NAO_USAR_ESTUARIO_CANAL"


def conjuntos():
    with open(CONTRATO / "feature_sets.json", encoding="utf-8") as f:
        return json.load(f)


def colunas_base(nome):
    conjunto = conjuntos()[nome]
    return conjunto["colunas"] if isinstance(conjunto, dict) else list(conjunto)


def features_do_conjunto(nome, modo="FORECAST_SAZONAL"):
    """Nomes das colunas CROCO que entram no modelo para o conjunto e o modo pedidos."""
    if modo not in SUFIXO_MODO:
        raise ValueError(f"modo desconhecido: {modo}")
    return [c + SUFIXO_MODO[modo] for c in colunas_base(nome)]


def arredonda_chave(tabela):
    tabela = tabela.copy()
    tabela["latitude"] = tabela["latitude"].round(CASAS_CHAVE)
    tabela["longitude"] = tabela["longitude"].round(CASAS_CHAVE)
    tabela["ano"] = tabela["ano"].astype(int)
    tabela["mes"] = tabela["mes"].astype(int)
    return tabela


def le_croco_mensal(caminho):
    """Lê o arquivo mensal do CROCO (Parquet do contrato 13.0, ou CSV) e confere a chave."""
    caminho = Path(caminho)
    croco = pd.read_parquet(caminho) if caminho.suffix == ".parquet" else pd.read_csv(caminho)
    faltando = [c for c in CHAVE + ["validade_espacial"] if c not in croco.columns]
    if faltando:
        raise ValueError(f"{caminho.name}: faltam as colunas {faltando}")
    croco = arredonda_chave(croco)
    duplicadas = int(croco.duplicated(CHAVE).sum())
    if duplicadas:
        raise ValueError(f"{caminho.name}: {duplicadas} chaves (lat, lon, ano, mes) repetidas após round({CASAS_CHAVE})")
    return croco


def junta_croco(tabela_ml, croco, nome_conjunto, modo="FORECAST_SAZONAL", permite_nowcast=False):
    """Left join do ML com as colunas CROCO do conjunto, no modo temporal pedido.

    13.5: o E0_corrigido prevê o mês t com dados até t-1 (FORECAST_1M), então NOWCAST (valor do próprio mês t)
    vazaria o mês-alvo: só é aceito com permite_nowcast=True, para diagnóstico, nunca em métrica de previsão.

    Remove os pontos NAO_USAR_ESTUARIO_CANAL, mantém a flag validade_espacial e
    devolve (tabela, relatorio). O lag é feito por DATA dentro da tabela do CROCO.
    """
    if modo == "NOWCAST" and not permite_nowcast:
        raise ValueError("NOWCAST usa o próprio mês-alvo (vazamento em previsão); use FORECAST_1M/FORECAST_SAZONAL "
                         "ou permite_nowcast=True só para diagnóstico")
    base = colunas_base(nome_conjunto)
    faltando = [c for c in base if c not in croco.columns]
    if faltando:
        raise ValueError(f"o arquivo do CROCO não tem as colunas {faltando}")

    croco = arredonda_chave(croco)
    lado_croco = croco[CHAVE + ["validade_espacial"]].copy()
    for c in base:
        destino = c + SUFIXO_MODO[modo]
        lado_croco[destino] = croco[c].to_numpy() if modo == "NOWCAST" else lag_por_data(croco, c, LAG_MODO[modo])

    ml = arredonda_chave(tabela_ml)
    duplicadas_ml = int(ml.duplicated(CHAVE).sum())
    if duplicadas_ml:
        raise ValueError(f"dataset do ML com {duplicadas_ml} chaves repetidas após round({CASAS_CHAVE})")

    juntado = ml.merge(lado_croco, on=CHAVE, how="left", validate="one_to_one", indicator=True)
    # O filtro é por ponto: um ponto NAO_USAR sai em todos os meses, mesmo sem CROCO naquele mês
    pontos_nao_usar = croco.loc[croco["validade_espacial"] == NAO_USAR, ["latitude", "longitude"]].drop_duplicates()
    nao_usar = juntado.set_index(["latitude", "longitude"]).index.isin(pd.MultiIndex.from_frame(pontos_nao_usar))
    nao_usar = pd.Series(nao_usar, index=juntado.index)
    colunas = features_do_conjunto(nome_conjunto, modo)
    relatorio = {
        "conjunto": nome_conjunto,
        "modo": modo,
        "colunas": colunas,
        "linhas_ml": int(len(ml)),
        "linhas_croco": int(len(croco)),
        "matches": int((juntado["_merge"] == "both").sum()),
        "ml_sem_croco": int((juntado["_merge"] == "left_only").sum()),
        "pontos_NAO_USAR": int(juntado.loc[nao_usar, ["latitude", "longitude"]].drop_duplicates().shape[0]),
        "linhas_NAO_USAR_removidas": int(nao_usar.sum()),
        "linhas_NAO_VALIDADO_COSTEIRO": int((juntado["validade_espacial"] == "NAO_VALIDADO_COSTEIRO").sum()),
        "linhas_com_feature_croco_completa": int(juntado.loc[~nao_usar, colunas].notna().all(axis=1).sum()),
    }
    juntado = juntado.loc[~nao_usar].drop(columns="_merge").reset_index(drop=True)
    return juntado, relatorio
