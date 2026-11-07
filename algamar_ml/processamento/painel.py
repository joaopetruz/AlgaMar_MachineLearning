"""Painel mensal único: histórico da equipe (2000–2024) + meses novos baixados das APIs.

Cada linha = (ponto, ano, mes), com proveniência por variável (fonte_*). Os pontos são
os do histórico da equipe (1 790 pontos oceânicos da grade CHL de 4 km). Meses novos
nunca sobrescrevem o histórico.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algamar_ml.processamento.mensal import abre, chave_ponto, processa

VARIAVEIS = {"clorofila": ["clorofila_media", "clorofila_maxima"],
             "temperatura": ["temperatura_media"],
             "salinidade": ["salinidade_media"]}
FONTE_EQUIPE = "historico_equipe(MY/REP, até 2024-12)"


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 22), b""):
            h.update(bloco)
    return h.hexdigest()


def le_historico(caminho):
    hist = pd.read_csv(caminho)
    hist["latitude"], hist["longitude"] = chave_ponto(hist["latitude"], hist["longitude"])
    for v in VARIAVEIS:
        hist[f"fonte_{v}"] = FONTE_EQUIPE
    return hist[["latitude", "longitude", "ano", "mes"] + sum(VARIAVEIS.values(), []) + [f"fonte_{v}" for v in VARIAVEIS]]


def arquivos_por_mes(brutos, variavel):
    """{(ano, mes): (caminho, registro)} dos meses baixados; prefere MY a NRT quando houver os dois."""
    achados = {}
    for registro_json in sorted(Path(brutos, variavel).glob("*/*.json")):
        registro = json.loads(registro_json.read_text(encoding="utf-8"))
        chave = (registro["ano"], registro["mes"])
        if chave in achados and achados[chave][1]["tipo"] == "MY":
            continue
        achados[chave] = (registro_json.with_suffix(".nc"), registro)
    return achados


def meses_novos(brutos, depois_de):
    """Processa os meses baixados posteriores a `depois_de` (ano, mes)."""
    por_var = {v: arquivos_por_mes(brutos, v) for v in VARIAVEIS}
    meses = sorted(k for k in por_var["clorofila"] if k > depois_de)
    partes, entradas = [], []
    for ano, mes in meses:
        ds = {}
        for v in VARIAVEIS:
            item = por_var[v].get((ano, mes))
            ds[v] = abre([item[0]]) if item else None
            if item:
                entradas.append({"variavel": v, "ano": ano, "mes": mes, "arquivo": str(item[0]),
                                 "dataset_id": item[1]["dataset_id"], "tipo": item[1]["tipo"], "sha256": sha256(item[0])})
        tabela = processa(ds["clorofila"], ds["temperatura"], ds["salinidade"])
        tabela = tabela[(tabela["ano"] == ano) & (tabela["mes"] == mes)]
        for v in VARIAVEIS:
            item = por_var[v].get((ano, mes))
            tabela[f"fonte_{v}"] = f"{item[1]['dataset_id']}({item[1]['tipo']})" if item else "ausente"
        partes.append(tabela)
    return (pd.concat(partes, ignore_index=True) if partes else None), entradas


def _ultimo_ano_mes(t):
    i = int((t["ano"] * 12 + t["mes"] - 1).max())
    return i // 12, i % 12 + 1


def _primeiro_ano_mes(t):
    i = int((t["ano"] * 12 + t["mes"] - 1).min())
    return i // 12, i % 12 + 1


def constroi(cfg):
    hist_csv = cfg["caminhos"]["historico_equipe"]
    hist = le_historico(hist_csv)
    ultimo = _ultimo_ano_mes(hist)
    novos, entradas = meses_novos(cfg["caminhos"]["brutos"], ultimo)

    pontos = hist[["latitude", "longitude"]].drop_duplicates()
    if novos is not None:
        novos = novos.merge(pontos, on=["latitude", "longitude"], how="inner")
        painel = pd.concat([hist, novos], ignore_index=True)
    else:
        painel = hist
    painel = painel.sort_values(["latitude", "longitude", "ano", "mes"]).reset_index(drop=True)
    if painel.duplicated(["latitude", "longitude", "ano", "mes"]).any():
        raise ValueError("painel com chave (ponto, ano, mes) repetida")

    manifesto = {
        "historico_equipe": {"arquivo": str(hist_csv), "sha256": sha256(hist_csv), "linhas": int(len(hist))},
        "meses_novos": entradas,
        "pontos": int(len(pontos)),
        "primeiro_mes": "%04d-%02d" % _primeiro_ano_mes(painel),
        "ultimo_mes": "%04d-%02d" % _ultimo_ano_mes(painel),
        "linhas": int(len(painel)),
        "nan_por_coluna": {c: int(painel[c].isna().sum()) for c in sum(VARIAVEIS.values(), [])},
    }
    return painel, manifesto


def salva(painel, manifesto, caminho):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    painel.to_parquet(caminho, index=False)
    manifesto["sha256_painel"] = sha256(caminho)
    caminho.with_suffix(".manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=1), encoding="utf-8")
