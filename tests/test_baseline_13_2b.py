"""Etapa 13.2B — split temporal, limiar só da validação e hotspots sem dados futuros."""
import numpy as np
import pandas as pd
import pytest

from algamar_ml import avaliacao
from algamar_ml.features.sazonal import distancia_hotspot_km, hotspots_do_treino


def tabela_sintetica(seed=0):
    rng = np.random.default_rng(seed)
    linhas = []
    for ano in range(2001, 2025):
        for p in range(30):
            x = rng.normal()
            linhas.append(dict(ano=ano, mes=1 + p % 12, f1=x, f2=rng.normal(), floracao=int(x + rng.normal(0, 0.5) > 0.8)))
    return pd.DataFrame(linhas)


def test_split_disjunto_e_por_ano():
    t = tabela_sintetica()
    tr, va, te = avaliacao.divide(t)
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all()
    assert t.loc[tr, "ano"].max() == 2018 and t.loc[va, "ano"].min() == 2019 and t.loc[va, "ano"].max() == 2021
    assert t.loc[te, "ano"].min() == 2022 and t.loc[te, "ano"].max() == 2024


def test_limiar_escolhido_so_na_validacao(monkeypatch):
    t = tabela_sintetica()
    _, va, te = avaliacao.divide(t)
    chamadas = []
    original = avaliacao.escolhe_limiar

    def espiao(y, p):
        chamadas.append(np.asarray(y).copy())
        return original(y, p)

    monkeypatch.setattr(avaliacao, "escolhe_limiar", espiao)
    _, res, _ = avaliacao.treina_e_avalia(t, ["f1", "f2"])
    assert len(chamadas) == 1                                    # uma única escolha
    assert np.array_equal(chamadas[0], t.loc[va, "floracao"].to_numpy())
    assert res["teste"]["limiar"] == res["limiar_escolhido_na_validacao"] == res["validacao"]["limiar"]


def test_limiar_nao_muda_quando_o_teste_muda():
    """Trocar os rótulos do teste não pode mudar o limiar (ele vem só da validação)."""
    a = tabela_sintetica()
    b = a.copy()
    te = b["ano"] >= 2022
    b.loc[te, "floracao"] = 1 - b.loc[te, "floracao"]
    _, ra, _ = avaliacao.treina_e_avalia(a, ["f1", "f2"])
    _, rb, _ = avaliacao.treina_e_avalia(b, ["f1", "f2"])
    assert ra["limiar_escolhido_na_validacao"] == rb["limiar_escolhido_na_validacao"]
    assert ra["validacao"] == rb["validacao"]


def test_hotspots_do_treino_ignoram_anos_futuros():
    rng = np.random.default_rng(1)
    pts = [(-24.0 + 0.5 * i, -46.0 + 0.5 * j) for i in range(4) for j in range(4)]
    m = pd.DataFrame([dict(latitude=la, longitude=lo, ano=a, mes=1, clorofila_media=rng.gamma(2, 0.5))
                      for la, lo in pts for a in range(2000, 2025)])
    h1 = hotspots_do_treino(m, ano_fim=2018)
    m2 = m.copy()
    m2.loc[m2.ano > 2018, "clorofila_media"] = 100.0 * rng.random((m2.ano > 2018).sum())
    assert hotspots_do_treino(m2, ano_fim=2018) == h1
    assert len(h1) == 2 and distancia_hotspot_km(pd.DataFrame({"latitude": [h1[0][0]], "longitude": [h1[0][1]]}), [h1[1]])[0] >= 50


def test_distancia_km():
    p = pd.DataFrame({"latitude": [-24.0], "longitude": [-46.0]})
    assert distancia_hotspot_km(p, [(-23.0, -46.0)])[0] == pytest.approx(111.19, abs=0.05)


def test_constroi_features_usa_hotspots_so_do_treino():
    """H2: mudar a clorofila de 2019+ não altera o dist_hotspot de nenhuma linha."""
    from algamar_ml.features import constroi_features
    rng = np.random.default_rng(2)
    pts = [(-24.0 + 0.5 * i, -46.0 + 0.5 * j) for i in range(3) for j in range(3)]
    linhas = []
    for la, lo in pts:
        for ano in range(2015, 2023):
            for mes in range(1, 13):
                c = rng.gamma(2, 0.5)
                linhas.append(dict(latitude=la, longitude=lo, ano=ano, mes=mes, clorofila_media=c, clorofila_maxima=1.8 * c,
                                   temperatura_media=22.0, salinidade_media=35.0))
    m = pd.DataFrame(linhas)
    m2 = m.copy()
    fut = m2.ano > 2018
    m2.loc[fut, "clorofila_media"] = 50.0 * rng.random(int(fut.sum()))
    m2.loc[fut, "clorofila_maxima"] = 1.8 * m2.loc[fut, "clorofila_media"]
    a, b = constroi_features(m), constroi_features(m2)
    assert np.array_equal(a["dist_hotspot"].to_numpy(), b["dist_hotspot"].to_numpy())
