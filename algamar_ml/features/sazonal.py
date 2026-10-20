"""Features sazonais sem vazamento de informação futura.

Regra: toda feature de uma linha (ponto, ano, mes) usa só meses ANTERIORES a ela.
- *_ano_anterior: valor de exatamente 12 meses antes, por data (NaN se o mês não existir);
- *_historica_mes: média do mesmo mês nos anos ANTERIORES ao da linha;
- *_3anos: média dos 36 meses ANTERIORES (t-36..t-1), exige 12 meses válidos.

A versão antiga (feature_engineering_sazonal.py até cc3145c/83d5f2b) usava todos os
anos na climatologia, incluía o mês t na média de 3 anos e fazia o shift por posição.
Ela fica em constroi_features_original() só como referência para medir o efeito.
"""
import numpy as np
import pandas as pd

CHAVE_PONTO = ["latitude", "longitude"]
CHAVE = CHAVE_PONTO + ["ano", "mes"]

HOTSPOTS = [(-23.98, -46.35), (-23.45, -45.75)]   # originais (mapa de 2024); só na constroi_features_original
ANO_FIM_HOTSPOT = 2018   # 13.2B (H2): hotspots recalculados com os anos de treino do baseline
LIMITE_FLORACAO = 1.5

FEATURES_BASELINE = [
    "mes", "latitude", "longitude", "dist_hotspot",
    "clorofila_media_ano_anterior", "clorofila_maxima_ano_anterior",
    "temperatura_ano_anterior", "salinidade_ano_anterior",
    "clorofila_media_historica_mes", "salinidade_media_historica_mes",
    "clorofila_media_3anos", "salinidade_media_3anos",
]


def indice_mensal(tabela):
    """Número de meses desde o ano 0 (ano*12 + mes-1)."""
    return tabela["ano"].astype(int) * 12 + (tabela["mes"].astype(int) - 1)


def confere_chave_unica(tabela):
    duplicadas = tabela.duplicated(CHAVE).sum()
    if duplicadas:
        raise ValueError(f"{duplicadas} linhas com (latitude, longitude, ano, mes) repetidos")


def lag_por_data(tabela, coluna, meses):
    """Valor de `coluna` exatamente `meses` antes, no mesmo ponto. NaN se esse mês não existir."""
    t = indice_mensal(tabela)
    origem = tabela[CHAVE_PONTO].assign(_t=t + meses, _v=tabela[coluna].to_numpy())
    destino = tabela[CHAVE_PONTO].assign(_t=t)
    juntado = destino.merge(origem, on=CHAVE_PONTO + ["_t"], how="left", validate="one_to_one")
    return juntado["_v"].to_numpy()


def climatologia_passada(tabela, coluna):
    """Média de `coluna` no mesmo (ponto, mês) só nos anos anteriores ao da linha."""
    ordem = tabela.sort_values(CHAVE_PONTO + ["mes", "ano"])
    chaves = [ordem[c] for c in CHAVE_PONTO + ["mes"]]
    valor = ordem[coluna]
    valido = valor.notna().astype(int)
    # cumsum até o ano anterior (shift), para o valor do ano t não entrar nem por arredondamento
    soma = valor.fillna(0).groupby(chaves).cumsum().groupby(chaves).shift(1)
    n = valido.groupby(chaves).cumsum().groupby(chaves).shift(1)
    media = (soma / n.where(n > 0)).reindex(tabela.index)
    return media.to_numpy()


def media_movel_passada(tabela, coluna, janela=36, min_periodos=12):
    """Média dos `janela` meses de calendário anteriores a t (exclui t)."""
    t = indice_mensal(tabela)
    ponto = tabela.groupby(CHAVE_PONTO, sort=False).ngroup()
    larga = pd.DataFrame({"t": t, "p": ponto, "v": tabela[coluna]}).pivot(index="t", columns="p", values="v")
    larga = larga.reindex(range(int(t.min()), int(t.max()) + 1))
    media = larga.shift(1).rolling(janela, min_periods=min_periodos).mean()
    return media.to_numpy()[(t - t.min()).to_numpy(), ponto.to_numpy()]


def distancia_hotspot(tabela, hotspots=HOTSPOTS):
    distancias = [
        np.sqrt((tabela["latitude"] - lat_h) ** 2 + (tabela["longitude"] - lon_h) ** 2)
        for lat_h, lon_h in hotspots
    ]
    return pd.concat(distancias, axis=1).min(axis=1)


def distancia_hotspot_km(tabela, hotspots=HOTSPOTS):
    """Distância (km, haversine) ao hotspot mais próximo."""
    la = np.radians(tabela["latitude"].to_numpy(dtype=float))[:, None]
    lo = np.radians(tabela["longitude"].to_numpy(dtype=float))[:, None]
    h = np.radians(np.asarray(hotspots, dtype=float))
    a = np.sin((h[:, 0] - la) / 2) ** 2 + np.cos(la) * np.cos(h[:, 0]) * np.sin((h[:, 1] - lo) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a)).min(axis=1)


def hotspots_do_treino(mensal, ano_fim, n=2, separacao_km=50.0):
    """Regra objetiva só com anos <= ano_fim: os `n` pontos de maior clorofila_media média, separados >= separacao_km."""
    media = (mensal[mensal["ano"] <= ano_fim].groupby(CHAVE_PONTO)["clorofila_media"].mean()
             .reset_index().sort_values("clorofila_media", ascending=False, kind="mergesort"))
    escolhidos = []
    for r in media.itertuples():
        p = pd.DataFrame({"latitude": [r.latitude], "longitude": [r.longitude]})
        if all(distancia_hotspot_km(p, [h])[0] >= separacao_km for h in escolhidos):
            escolhidos.append((float(r.latitude), float(r.longitude)))
        if len(escolhidos) == n:
            break
    return escolhidos


def _finaliza(tabela, dist=None):
    tabela["dist_hotspot"] = distancia_hotspot(tabela) if dist is None else dist
    tabela = tabela.dropna(subset=["clorofila_media_ano_anterior", "temperatura_ano_anterior", "salinidade_ano_anterior"])
    tabela = tabela.copy()
    tabela["floracao"] = (tabela["clorofila_maxima"] > LIMITE_FLORACAO).astype(int)
    return tabela


def constroi_features(mensal, hotspots=None):
    """Features do baseline corrigido (sem vazamento) a partir da tabela mensal.

    dist_hotspot (13.2B, opção H2 escolhida pelo Tulio): distância em km aos 2 hotspots recalculados só com os anos de
    treino (<= ANO_FIM_HOTSPOT); `hotspots` permite passar uma lista fixa (ex.: congelada num modelo salvo)."""
    tabela = mensal.sort_values(CHAVE).reset_index(drop=True)
    confere_chave_unica(tabela)

    tabela["clorofila_media_ano_anterior"] = lag_por_data(tabela, "clorofila_media", 12)
    tabela["clorofila_maxima_ano_anterior"] = lag_por_data(tabela, "clorofila_maxima", 12)
    tabela["temperatura_ano_anterior"] = lag_por_data(tabela, "temperatura_media", 12)
    tabela["salinidade_ano_anterior"] = lag_por_data(tabela, "salinidade_media", 12)

    tabela["clorofila_media_historica_mes"] = climatologia_passada(tabela, "clorofila_media")
    tabela["salinidade_media_historica_mes"] = climatologia_passada(tabela, "salinidade_media")

    tabela["clorofila_media_3anos"] = media_movel_passada(tabela, "clorofila_media")
    tabela["salinidade_media_3anos"] = media_movel_passada(tabela, "salinidade_media")

    if hotspots is None:
        hotspots = hotspots_do_treino(mensal, ano_fim=ANO_FIM_HOTSPOT)
    return _finaliza(tabela, distancia_hotspot_km(tabela, hotspots))


def constroi_features_original(mensal):
    """Construção antiga, COM vazamento. Só para comparação; não usar para treinar."""
    tabela = mensal.sort_values(CHAVE).reset_index(drop=True)
    grupo = tabela.groupby(CHAVE_PONTO)

    tabela["clorofila_media_ano_anterior"] = grupo["clorofila_media"].shift(12)
    tabela["clorofila_maxima_ano_anterior"] = grupo["clorofila_maxima"].shift(12)
    tabela["temperatura_ano_anterior"] = grupo["temperatura_media"].shift(12)
    tabela["salinidade_ano_anterior"] = grupo["salinidade_media"].shift(12)

    tabela["clorofila_media_historica_mes"] = tabela.groupby(CHAVE_PONTO + ["mes"])["clorofila_media"].transform("mean")
    tabela["salinidade_media_historica_mes"] = tabela.groupby(CHAVE_PONTO + ["mes"])["salinidade_media"].transform("mean")

    tabela["clorofila_media_3anos"] = grupo["clorofila_media"].transform(lambda x: x.rolling(window=36, min_periods=12).mean())
    tabela["salinidade_media_3anos"] = grupo["salinidade_media"].transform(lambda x: x.rolling(window=36, min_periods=12).mean())

    return _finaliza(tabela)


def climatologia_para_previsao(mensal):
    """Climatologia por (ponto, mês) com todos os anos da base.

    Serve para prever meses DEPOIS do último ano da base (todos os anos já são passado).
    Não usar para montar linhas de treino.
    """
    return (
        mensal.groupby(CHAVE_PONTO + ["mes"])[["clorofila_media", "salinidade_media"]]
        .mean()
        .rename(columns={
            "clorofila_media": "clorofila_media_historica_mes",
            "salinidade_media": "salinidade_media_historica_mes",
        })
    )
