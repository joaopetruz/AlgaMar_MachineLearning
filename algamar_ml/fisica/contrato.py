"""Contrato das features físicas (contrato/features_fisicas.csv): padronização e validação."""
from pathlib import Path

import numpy as np
import pandas as pd

CAMINHO = Path(__file__).resolve().parents[2] / "contrato" / "features_fisicas.csv"
CHAVE = ["latitude", "longitude", "ano", "mes"]


def carrega(caminho=CAMINHO):
    c = pd.read_csv(caminho, dtype=str).fillna("")
    c["valor_min"] = c["valor_min"].astype(float)
    c["valor_max"] = c["valor_max"].astype(float)
    return c.set_index("nome")


def conjunto(nome_conjunto, contrato=None):
    contrato = carrega() if contrato is None else contrato
    nomes = [n for n, linha in contrato.iterrows() if nome_conjunto in linha["conjuntos"].split(";")]
    if not nomes:
        raise ValueError(f"conjunto desconhecido: {nome_conjunto}")
    return nomes


def padroniza(tabela, contrato=None):
    """Renomeia as colunas da fonte (`coluna_fonte`) para os nomes do contrato. Colunas fora do contrato saem."""
    contrato = carrega() if contrato is None else contrato
    mapa = {linha["coluna_fonte"]: nome for nome, linha in contrato.iterrows()
            if linha["coluna_fonte"] and linha["coluna_fonte"] in tabela.columns and nome not in tabela.columns}
    tabela = tabela.rename(columns=mapa)
    manter = [c for c in tabela.columns if c in contrato.index or c in CHAVE or c in ("time", "validade_espacial")]
    return tabela[manter]


def valida(tabela, contrato=None, exige_conjunto=None):
    """Lista de problemas (vazia = válido): chave, duplicatas, tipos, faixas e obrigatoriedade."""
    contrato = carrega() if contrato is None else contrato
    problemas = []
    faltam_chave = [c for c in CHAVE if c not in tabela.columns]
    if faltam_chave:
        return [f"faltam colunas de chave: {faltam_chave}"]
    dup = int(tabela.duplicated(CHAVE).sum())
    if dup:
        problemas.append(f"{dup} chaves (latitude, longitude, ano, mes) repetidas")
    if not tabela["mes"].between(1, 12).all():
        problemas.append("mes fora de 1..12")
    if not (tabela["latitude"].between(-90, 90).all() and tabela["longitude"].between(-180, 180).all()):
        problemas.append("coordenadas fora de WGS84 (lon −180..180)")
    exigidas = conjunto(exige_conjunto, contrato) if exige_conjunto else []
    for nome in exigidas:
        if nome not in tabela.columns:
            problemas.append(f"feature obrigatória do conjunto {exige_conjunto} ausente: {nome}")
    for nome in [c for c in tabela.columns if c in contrato.index]:
        col = tabela[nome]
        if not np.issubdtype(col.dtype, np.number):
            problemas.append(f"{nome}: tipo {col.dtype}, esperado float")
            continue
        lim = contrato.loc[nome]
        fora = col.notna() & ~col.between(lim["valor_min"], lim["valor_max"])
        if fora.any():
            problemas.append(f"{nome}: {int(fora.sum())} valores fora de [{lim['valor_min']}, {lim['valor_max']}] {lim['unidade']}")
        if np.isinf(col.to_numpy(dtype=float)).any():
            problemas.append(f"{nome}: valores infinitos")
    return problemas
