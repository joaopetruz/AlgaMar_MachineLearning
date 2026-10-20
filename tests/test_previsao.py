import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from algamar_ml import config, metricas
from algamar_ml.apis.copernicus import PRODUTOS, escolhe_produto
from algamar_ml.espacial import haversine_km, regiao, vizinho_mais_proximo
from algamar_ml.features.horizonte import cubo_do_painel, nomes_features
from algamar_ml.modelos import treino as tr
from algamar_ml.pipeline import dataset, treina_e_avalia
from algamar_ml.previsao import pico_por_ponto, preve, resumo_regioes
from algamar_ml.processamento.mensal import processa

from tests.test_horizonte import painel_sintetico

COLUNAS_SAIDA = ["prediction_date", "issue_month", "target_date", "forecast_horizon", "point_id", "latitude",
                 "longitude", "region", "bloom_probability", "climatological_probability", "risk_level",
                 "model_version", "feature_flags", "data_sources"]


@pytest.fixture(scope="module")
def cfg_pequeno():
    cfg = config.carrega()
    cfg["split"] = {"fim_treino": "2006-12", "inicio_validacao": "2007-01", "fim_validacao": "2008-12", "inicio_teste": "2009-01"}
    cfg["modelo"].update(n_estimators=30)
    return cfg


@pytest.fixture(scope="module")
def treinado(cfg_pequeno):
    p = painel_sintetico(anos=range(2000, 2011), pontos=((-24.0, -46.8), (-24.5, -46.0), (-23.8, -45.2)))
    cubo = cubo_do_painel(p)
    linhas = dataset(cfg_pequeno, cubo)
    flags = {"use_historical_features": True, "use_api_features": True, "use_navier_stokes_features": False}
    modelo, cartao, _ = treina_e_avalia(cfg_pequeno, linhas, nomes_features(flags), "teste", flags)
    return modelo, cartao, cubo, linhas


def test_split_sem_sobreposicao_de_alvos(treinado, cfg_pequeno):
    _, _, _, linhas = treinado
    tre, val, tes = tr.divide(linhas, cfg_pequeno["split"])
    assert not (tre & val).any() and not (tre & tes).any() and not (val & tes).any()
    assert linhas.loc[tre, "tau"].max() < linhas.loc[val, "tau"].min()
    assert linhas.loc[val, "t0"].min() > linhas.loc[tre, "tau"].max() - 1


def test_saida_da_previsao_tem_schema_12_meses_e_regioes(treinado, cfg_pequeno):
    modelo, cartao, cubo, _ = treinado
    prev = preve(modelo, cartao, cubo, cfg_pequeno, fontes="teste")
    assert all(c in prev.columns for c in COLUNAS_SAIDA)
    assert sorted(prev["forecast_horizon"].unique()) == list(range(1, 13))
    assert prev["issue_month"].nunique() == 1 and prev["issue_month"].iloc[0] == "2010-12"
    assert prev["target_date"].min() == "2011-01" and prev["target_date"].max() == "2011-12"
    assert prev["bloom_probability"].between(0, 1).all()
    assert set(prev["risk_level"]) <= {"BAIXO", "MODERADO", "ALTO"}
    assert prev["region"].notna().all()
    assert len(resumo_regioes(prev)) == 12 * prev["region"].nunique()
    assert len(pico_por_ponto(prev)) == prev["point_id"].nunique()


def test_cartao_registra_status_honesto_e_metricas_por_horizonte(treinado):
    _, cartao, _, _ = treinado
    assert cartao["status"].startswith("HINDCAST_VALIDADO")      # fechamento: equivale à climatologia, não é operacional
    assert set(cartao["teste"]["por_h"]) == {str(h) for h in range(1, 13)}
    assert {"modelo", "climatologia", "persistencia"} <= set(cartao["teste"]["geral"])


def test_salva_e_carrega_modelo(treinado, tmp_path):
    modelo, cartao, cubo, linhas = treinado
    tr.salva(modelo, cartao, tmp_path / "m")
    m2, c2 = tr.carrega(tmp_path / "m")
    x = linhas[cartao["features"]].head(50)
    assert np.allclose(modelo.predict_proba(x), m2.predict_proba(x))
    assert c2["features"] == cartao["features"]


def test_regioes_e_haversine():
    cfg = config.carrega()
    nomes = regiao([-46.9, -46.0, -45.0], cfg["regioes"])
    assert nomes[0].startswith("sul") and nomes[1].startswith("centro") and nomes[2].startswith("norte")
    assert haversine_km(-24, -46, -24, -45) == pytest.approx(101.7, abs=0.5)
    idx, dist = vizinho_mais_proximo([-24.0, -30.0], [-46.0, -46.0], [-24.01, -23.0], [-46.0, -46.0], 5.0)
    assert idx[0] == 0 and idx[1] == -1 and dist[0] == pytest.approx(1.11, abs=0.05)


def test_escolha_de_produto_my_ou_nrt():
    cob = {p.dataset_id: (dt.date(2000, 1, 1), dt.date(2026, 3, 31) if p.tipo == "MY" else dt.date(2026, 10, 7))
           for ps in PRODUTOS.values() for p in ps}
    assert escolhe_produto("temperatura", 2026, 3, cob).tipo == "MY"
    assert escolhe_produto("temperatura", 2026, 4, cob).tipo == "NRT"
    assert escolhe_produto("temperatura", 2026, 10, cob) is None  # mês incompleto


def test_processamento_mensal_sintetico():
    tempo = pd.date_range("2025-01-01", "2025-02-28", freq="D")
    lat, lon = np.array([-24.0, -23.96]), np.array([-46.0, -45.96])
    chl = xr.Dataset({"CHL": (("time", "latitude", "longitude"), np.ones((len(tempo), 2, 2)))},
                     coords={"time": tempo, "latitude": lat, "longitude": lon})
    chl["CHL"][31:] = 2.0
    sst = xr.Dataset({"analysed_sst": (("time", "latitude", "longitude"), np.full((len(tempo), 3, 3), 298.15))},
                     coords={"time": tempo, "latitude": [-24.1, -24.0, -23.9], "longitude": [-46.1, -46.0, -45.9]})
    t = processa(chl, sst, None)
    jan = t[(t.mes == 1)].iloc[0]
    fev = t[(t.mes == 2)].iloc[0]
    assert jan["clorofila_media"] == 1 and fev["clorofila_maxima"] == 2
    assert jan["temperatura_media"] == pytest.approx(25.0)
    assert np.isnan(jan["salinidade_media"])


def test_bootstrap_delta_detecta_melhora():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    ruim = rng.random(4000)
    bom = np.clip(y * 0.6 + rng.random(4000) * 0.5, 0, 1)
    r = metricas.bootstrap_delta(y, ruim, bom, blocos=np.arange(4000) % 20, n=50)
    assert r["delta"] > 0 and r["ic95"][0] > 0
