"""Feature store das features físicas: fonte → mensal → pontos do ML → validação → Parquet + manifesto.

Estrutura: <feature_store>/<fonte>/features_mensal.parquet e manifesto.json (proveniência:
fonte, versão, contrato, métodos de alinhamento, cobertura e problemas de validação).
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algamar_ml.features.horizonte import indice_mes
from algamar_ml.fisica import contrato as ctr
from algamar_ml.fisica.alinhamento import mensal, para_pontos_ml

PREFIXO = "ns_"


def ingere(fonte, pontos_ml, pasta_store, dist_max_km=3.0, cobertura_minima=0.8):
    """Executa a cadeia completa e grava a store. Recusa gravar se a validação falhar."""
    bruto = fonte.carrega()
    contrato = ctr.carrega()
    colunas = [c for c in bruto.columns if c in contrato.index]
    if not colunas:
        raise ValueError("a fonte não tem nenhuma feature do contrato")
    if "ano" in bruto.columns and "mes" in bruto.columns:
        mensal_ = bruto
        metodo_tempo = "já mensal na fonte"
    else:
        mensal_ = mensal(bruto, colunas, cobertura_minima)
        metodo_tempo = f"média do mês civil UTC; mês com cobertura de dias < {cobertura_minima} vira NaN"
    no_ml, _ = para_pontos_ml(mensal_, pontos_ml, colunas, dist_max_km)
    no_ml = no_ml[[c for c in ("latitude", "longitude", "ano", "mes", *colunas, "validade_espacial",
                               "dist_associacao_km", "metodo_associacao", "cobertura_dias") if c in no_ml.columns]]
    problemas = ctr.valida(no_ml, contrato)
    if problemas:
        raise ValueError("feature store recusada: " + "; ".join(problemas))

    pasta = Path(pasta_store) / fonte.nome
    pasta.mkdir(parents=True, exist_ok=True)
    no_ml.to_parquet(pasta / "features_mensal.parquet", index=False)
    meses = indice_mes(no_ml["ano"], no_ml["mes"])
    manifesto = {
        **fonte.descricao(),
        "contrato": str(ctr.CAMINHO.name),
        "criado_em": dt.datetime.now(dt.UTC).isoformat(),
        "features": colunas,
        "alinhamento_temporal": metodo_tempo,
        "alinhamento_espacial": str(no_ml["metodo_associacao"].iloc[0]) if "metodo_associacao" in no_ml else "",
        "dist_max_km": dist_max_km,
        "pontos": int(no_ml[["latitude", "longitude"]].drop_duplicates().shape[0]),
        "pontos_ml": int(len(pontos_ml)),
        "primeiro_mes": f"{int(meses.min()) // 12}-{int(meses.min()) % 12 + 1:02d}",
        "ultimo_mes": f"{int(meses.max()) // 12}-{int(meses.max()) % 12 + 1:02d}",
        "linhas": int(len(no_ml)),
        "nan_por_feature": {c: int(no_ml[c].isna().sum()) for c in colunas},
        "status_validacao_contrato": {c: contrato.loc[c, "status_validacao"] for c in colunas},
    }
    (pasta / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=1), encoding="utf-8")
    return no_ml, manifesto


def le(pasta_store, fonte):
    pasta = Path(pasta_store) / fonte
    tabela = pd.read_parquet(pasta / "features_mensal.parquet")
    manifesto = json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))
    return tabela, manifesto


def para_cubo(tabela, cubo, nomes, remove_nao_usar=True):
    """Matrizes [mês, ponto] alinhadas ao cubo do ML: `ns_<nome>_t0` (estado físico no mês de emissão).
    Pontos NAO_USAR_ESTUARIO_CANAL ficam NaN."""
    t = tabela.copy()
    if remove_nao_usar and "validade_espacial" in t:
        t.loc[t["validade_espacial"] == "NAO_USAR_ESTUARIO_CANAL", nomes] = np.nan
    t["latitude"], t["longitude"] = t["latitude"].round(5), t["longitude"].round(5)
    pontos = cubo.pontos.assign(latitude=cubo.pontos["latitude"].round(5), longitude=cubo.pontos["longitude"].round(5))
    t = t.merge(pontos, on=["latitude", "longitude"], how="inner")
    ti = indice_mes(t["ano"], t["mes"]) - cubo.t_inicio
    dentro = (ti >= 0) & (ti < cubo.T)
    saida = {}
    for nome in nomes:
        a = np.full((cubo.T, len(cubo.pontos)), np.nan, dtype=np.float32)
        a[ti[dentro], t["point_id"].to_numpy()[dentro]] = t[nome].to_numpy(dtype=np.float32)[dentro]
        saida[f"{PREFIXO}{nome}_t0"] = a
    return saida
