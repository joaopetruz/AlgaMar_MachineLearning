import json

import pandas as pd
import pytest
from fastapi import HTTPException

from algamar_ml import api_previsao


@pytest.fixture()
def pasta_previsao(tmp_path, monkeypatch):
    pasta = tmp_path / "2026-09"
    pasta.mkdir()
    linhas = []
    for pid, (lat, lon, reg) in enumerate([(-24.0, -46.3, "centro"), (-23.6, -45.0, "norte")]):
        for h in range(1, 13):
            mes = (8 + h) % 12 + 1
            ano = 2026 if 9 + h <= 12 else 2027
            linhas.append(dict(prediction_date="2026-10-08", issue_month="2026-09", target_date=f"{ano}-{mes:02d}",
                               forecast_horizon=h, point_id=pid, latitude=lat, longitude=lon, region=reg,
                               bloom_probability=0.1 + 0.05 * h if pid == 0 else 0.05, climatological_probability=0.2,
                               risk_level="ALTO" if pid == 0 and h > 8 else "BAIXO", model_version="t", feature_flags="",
                               data_sources="", estado_t0_completo=True))
    prev = pd.DataFrame(linhas)
    prev.to_parquet(pasta / "previsao_pontos.parquet")
    prev.groupby(["region", "target_date", "forecast_horizon"])["bloom_probability"].mean().rename("prob_media").reset_index().to_csv(pasta / "resumo_regioes.csv", index=False)
    (pasta / "metadados.json").write_text(json.dumps({"issue_month": "2026-09", "model_version": "t"}), encoding="utf-8")
    monkeypatch.setattr(api_previsao, "PASTA", tmp_path)
    monkeypatch.setattr(api_previsao, "SAIDAS", tmp_path / "sem_saidas")
    api_previsao._carrega.cache_clear()
    return tmp_path


def test_previsao_no_local_devolve_12_meses_com_ano_e_pico(pasta_previsao):
    r = api_previsao.previsao_no_local(latitude=-24.01, longitude=-46.31)
    assert len(r["meses"]) == 12 and r["meses"][0]["target_date"] == "2026-10" and r["meses"][-1]["target_date"] == "2027-09"
    assert r["mes_de_maior_risco"] == "2027-09" and r["region"] == "centro"


def test_local_longe_da_grade_da_404(pasta_previsao):
    with pytest.raises(HTTPException) as e:
        api_previsao.previsao_no_local(latitude=-24.9, longitude=-47.85)
    assert e.value.status_code == 404


def test_mapa_filtra_por_nivel_e_valida_mes(pasta_previsao):
    r = api_previsao.mapa(target_date="2027-09", nivel_minimo="ALTO")
    assert [p["point_id"] for p in r["pontos"]] == [0]
    with pytest.raises(HTTPException):
        api_previsao.mapa(target_date="2030-01", nivel_minimo="BAIXO")


def test_sem_previsao_gerada_da_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api_previsao, "PASTA", tmp_path)
    monkeypatch.setattr(api_previsao, "SAIDAS", tmp_path / "sem_saidas")
    with pytest.raises(HTTPException) as e:
        api_previsao.meta()
    assert e.value.status_code == 503


def _saida_pipeline(raiz, run_id, com_parquet=True, versao="oficial-x"):
    pasta = raiz / run_id
    pasta.mkdir(parents=True)
    linhas = []
    for pid, (lat, lon, reg, ok) in enumerate([(-24.0, -46.3, "centro", True), (-23.6, -45.0, "norte", False)]):
        for h in range(1, 13):
            mes = (8 + h) % 12 + 1
            ano = 2026 if 9 + h <= 12 else 2027
            linhas.append(dict(run_id=run_id, prediction_date="2026-09", target_date=f"{ano}-{mes:02d}", horizon=h, point_id=pid,
                               latitude=lat, longitude=lon, regiao=reg, probabilidade=(0.1 + 0.05 * h) if ok else None,
                               probabilidade_climatologica=0.2, classe=None, nivel_de_risco=("ALTO" if h > 8 else "BAIXO") if ok else "DADOS_INSUFICIENTES",
                               status_ponto="OK" if ok else "DADOS_INSUFICIENTES", modelo="xgb", versao=versao,
                               status_execucao="OK", classificacao_produto="HINDCAST_VALIDADO"))
    prev = pd.DataFrame(linhas)
    if com_parquet:
        prev.to_parquet(pasta / "previsao.parquet")
        reg = prev.groupby(["regiao", "target_date", "horizon"])["probabilidade"].mean().rename("prob_media").reset_index()
        reg["frac_alto"] = 0.5
        reg.to_parquet(pasta / "risco_regioes.parquet")
    (pasta / "previsao.json").write_text(json.dumps({"run_id": run_id, "status": "OK" if com_parquet else "DADOS_INSUFICIENTES",
        "modelo": versao, "classificacao_produto": "HINDCAST_VALIDADO", "periodo": {"emissao_t0": "2026-09"},
        "produto": {"nome": "Mapa", "fontes_de_dados": ["a", "b"]}, "resumo": {"pontos": 2, "pontos_insuficientes": 1}}), encoding="utf-8")
    (raiz / "ULTIMA.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")


def test_serve_saida_oficial_antes_da_legada(pasta_previsao, monkeypatch):
    saidas = pasta_previsao / "saidas"
    _saida_pipeline(saidas, "20261009T100000Z_aaaaaa", versao="oficial-x")
    _saida_pipeline(saidas, "20261009T110000Z_bbbbbb", com_parquet=False)   # última execução sem probabilidades
    monkeypatch.setattr(api_previsao, "SAIDAS", saidas)
    m = api_previsao.meta()
    assert m["model_version"] == "oficial-x" and m["issue_month"] == "2026-09" and m["origem"] == "saidas/20261009T100000Z_aaaaaa"
    assert m["meses_previstos"][0] == "2026-10" and m["pontos"] == 2 and m["fontes_mes_emissao"] == "a; b"
    r = api_previsao.previsao_no_local(latitude=-24.01, longitude=-46.31)
    assert r["mes_de_maior_risco"] == "2027-09" and r["meses"][0]["forecast_horizon"] == 1
    sem = api_previsao.previsao_no_local(latitude=-23.6, longitude=-45.0)
    assert sem["mes_de_maior_risco"] is None and sem["meses"][0]["risk_level"] == "DADOS_INSUFICIENTES"
    assert [p["point_id"] for p in api_previsao.mapa(target_date="2027-09", nivel_minimo="ALTO")["pontos"]] == [0]
    assert {"region", "forecast_horizon", "prob_media", "frac_pontos_alto"} <= set(api_previsao.por_regiao()["regioes"][0])


def test_legada_ignora_pasta_sem_metadados(pasta_previsao):
    (pasta_previsao / "operacional_2026-09").mkdir()
    pd.read_parquet(pasta_previsao / "2026-09" / "previsao_pontos.parquet").to_parquet(pasta_previsao / "operacional_2026-09" / "previsao_pontos.parquet")
    assert api_previsao.meta()["issue_month"] == "2026-09"
