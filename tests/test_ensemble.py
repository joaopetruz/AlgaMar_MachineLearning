"""Etapa 14.1 — ensemble de ordens: seeds derivadas sem reuso, funções puras, calibração só na validação,
registro imutável com os membros no manifesto e fluxo completo (treino → previsão) com o ensemble."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from algamar_ml import config, fluxo, metricas, registro
from algamar_ml.features.horizonte import cubo_do_painel, nomes_features
from algamar_ml.modelos import ensemble as ens
from algamar_ml.modelos import treino as tr
from algamar_ml.pipeline import dataset, treina_e_avalia
from tests.test_horizonte import painel_sintetico

FLAGS = {"use_historical_features": True, "use_api_features": True, "use_navier_stokes_features": False}
ENS = {"k": 3, "seed_mestra": 7, "calibracao": "isotonica"}


@pytest.fixture(scope="module")
def dados():
    cfg = config.carrega()
    cfg["split"] = {"fim_treino": "2006-12", "inicio_validacao": "2007-01", "fim_validacao": "2008-12", "inicio_teste": "2009-01"}
    cfg["modelo"].update(n_estimators=30)
    p = painel_sintetico(anos=range(2000, 2011), pontos=((-24.0, -46.8), (-24.5, -46.0), (-23.8, -45.2)))
    linhas = dataset(cfg, cubo_do_painel(p))
    return cfg, linhas, nomes_features(FLAGS)


@pytest.fixture(scope="module")
def treinado(dados):
    cfg, linhas, feats = dados
    modelo, cartao, extra = treina_e_avalia(cfg, linhas, feats, "ens-teste", FLAGS, ensemble=ENS)
    return modelo, cartao, extra


# ---------------------------------------------------------------- seeds
def test_seeds_derivadas_reprodutiveis_e_sem_reuso():
    a, b = ens.seeds_derivadas(42, 8), ens.seeds_derivadas(42, 8)
    assert a == b
    todas = [s["seed_permutacao"] for s in a] + [s["seed_xgboost"] for s in a]
    assert len(set(todas)) == 16                                       # nenhuma chave reaproveitada
    assert len({tuple(s["spawn_key"]) for s in a}) == 8
    assert ens.seeds_derivadas(43, 8) != a
    assert ens.seeds_derivadas(42, 4) == a[:4]                         # aumentar k não muda os membros já existentes


def test_permutacao_e_permutacao_reprodutivel():
    p = ens.permutacao(1000, 123)
    assert sorted(p) == list(range(1000)) and (p == ens.permutacao(1000, 123)).all()
    assert not (p == ens.permutacao(1000, 124)).all()


# ---------------------------------------------------------------- pureza
def test_treino_nao_altera_o_dataframe(dados):
    cfg, linhas, feats = dados
    antes = linhas.copy(deep=True)
    ens.treina_ensemble(linhas, feats, cfg, k=2, seed_mestra=1)
    pd.testing.assert_frame_equal(linhas, antes)


def test_mesma_seed_mestra_da_o_mesmo_ensemble(dados):
    cfg, linhas, feats = dados
    uso = {"features": feats, "base_climatologica": True}
    p = []
    for _ in range(2):
        m = ens.treina_ensemble(linhas, feats, cfg, k=2, seed_mestra=5)
        p.append(tr.probabilidade(m, linhas, {**uso, "p0": m.algamar_p0}))
    np.testing.assert_array_equal(p[0], p[1])


# ---------------------------------------------------------------- média e calibração
def test_probabilidade_bruta_e_media_dos_membros(treinado, dados):
    modelo, cartao, _ = treinado
    _, linhas, feats = dados
    X, bm = linhas[feats], tr.margem_base(linhas, cartao["p0"])
    np.testing.assert_allclose(modelo.predict_proba(X, base_margin=bm, calibrada=False)[:, 1],
                               modelo.prob_membros(X, bm).mean(axis=1), rtol=1e-6)


def test_isotonica_monotona_e_ajustada_so_na_validacao(treinado, dados):
    modelo, cartao, _ = treinado
    cfg, linhas, feats = dados
    _, va, _ = tr.divide(linhas, cfg["split"])
    bm = tr.margem_base(linhas[va], cartao["p0"])
    p_bruto = modelo.predict_proba(linhas.loc[va, feats], base_margin=bm, calibrada=False)[:, 1]
    assert modelo.isotonica == ens.ajusta_isotonica(linhas.loc[va, "floracao"].to_numpy(int), p_bruto)
    grade = np.linspace(0, 1, 201)
    c = ens.aplica_isotonica(modelo.isotonica, grade)
    assert (np.diff(c) >= -1e-12).all() and c.min() >= 0 and c.max() <= 1
    # limiar re-escolhido na validação com a probabilidade calibrada
    p_cal = tr.probabilidade(modelo, linhas[va], cartao)
    assert cartao["limiar_alto"] == metricas.escolhe_limiar(linhas.loc[va, "floracao"].to_numpy(int), p_cal)


def test_com_calibracao_nao_altera_o_original(treinado):
    modelo, _, _ = treinado
    bruto = modelo.com_calibracao(None)
    assert bruto.isotonica is None and modelo.isotonica is not None and bruto.membros == modelo.membros


def test_cartao_registra_seeds_e_membros(treinado):
    _, cartao, _ = treinado
    e = cartao["ensemble"]
    assert e["k"] == 3 and e["seed_mestra"] == 7 and e["seeds"] == ens.seeds_derivadas(7, 3)
    assert e["calibracao"] == "isotonica" and len(e["melhores_iteracoes"]) == 3


# ---------------------------------------------------------------- registro
def test_registro_ensemble_roundtrip_e_manifesto_cobre_membros(treinado, dados, tmp_path):
    modelo, cartao, _ = treinado
    _, linhas, _ = dados
    man = registro.registra(modelo, cartao, tmp_path, "v-ens")
    assert set(man["arquivos"]) == {"modelo.json", "cartao.json", "membro_00.json", "membro_01.json", "membro_02.json"}
    m2, c2, _ = registro.carrega(tmp_path, "v-ens")
    assert isinstance(m2, ens.EnsembleOrdens) and m2.seeds == modelo.seeds
    np.testing.assert_allclose(tr.probabilidade(m2, linhas, c2), tr.probabilidade(modelo, linhas, cartao), rtol=1e-6)
    with pytest.raises(registro.ErroRegistro):
        registro.registra(modelo, cartao, tmp_path, "v-ens")          # nunca sobrescreve
    alvo = tmp_path / "v-ens" / "membro_01.json"
    alvo.write_text(alvo.read_text(encoding="utf-8").replace("0", "1", 1), encoding="utf-8")
    with pytest.raises(registro.ErroRegistro, match="MANIFESTO"):
        registro.carrega(tmp_path, "v-ens")


def test_modelo_simples_continua_carregando(dados, tmp_path):
    cfg, linhas, feats = dados
    modelo, cartao, _ = treina_e_avalia(cfg, linhas, feats, "simples", FLAGS)
    registro.registra(modelo, cartao, tmp_path, "simples")
    m2, c2, man = registro.carrega(tmp_path, "simples")
    assert set(man["arquivos"]) == {"modelo.json", "cartao.json"} and "ensemble" not in c2
    np.testing.assert_allclose(tr.probabilidade(m2, linhas, c2), tr.probabilidade(modelo, linhas, cartao), rtol=1e-6)


# ---------------------------------------------------------------- fluxo completo
def test_fluxo_treina_e_preve_com_ensemble(tmp_path):
    cfg = config.carrega()
    cfg["caminhos"].update(painel=tmp_path / "painel.parquet", brutos=tmp_path / "brutos", modelos=tmp_path / "modelos",
                           croco_mensal=tmp_path / "nao_existe.parquet")
    cfg["pipeline"].update(saidas=str(tmp_path / "saidas"), logs=str(tmp_path / "logs"), hoje="")
    cfg["split"] = {"fim_treino": "2006-12", "inicio_validacao": "2007-01", "fim_validacao": "2008-12", "inicio_teste": "2009-01"}
    cfg["modelo"].update(n_estimators=30)
    cfg["ensemble"] = {**ENS, "ativo": True}
    painel_sintetico(anos=range(2000, 2011), pontos=((-24.0, -46.8), (-24.5, -46.0), (-23.8, -45.2))).to_parquet(cfg["caminhos"]["painel"])
    (tmp_path / "brutos").mkdir()
    status, log = fluxo.executa_treino(copy.deepcopy(cfg), "ens1")
    assert status == "OK", log["erros"]
    cartao = json.loads((tmp_path / "modelos" / "ens1" / "cartao.json").read_text(encoding="utf-8"))
    assert cartao["ensemble"]["k"] == 3
    status, log = fluxo.executa_previsao(copy.deepcopy(cfg), "ens1", hoje=pd.Timestamp("2011-01-15").date(),
                                         consulta_api=lambda _: (_ for _ in ()).throw(ConnectionError("simulada")))
    assert status in ("OK", "ATRASADO", "DESATUALIZADO"), log["erros"]
    prev = pd.read_parquet(f"{log['resultado']['saida']}/previsao.parquet")
    p = prev["probabilidade"].dropna()
    assert len(p) and p.between(0, 1).all() and (prev["versao"] == "ens1").all()
