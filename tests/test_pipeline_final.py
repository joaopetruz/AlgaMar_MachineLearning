"""Etapa 13.8 — testes de ponta a ponta do pipeline final (algamar_ml/fluxo.py) e dos modos de falha.
Tudo roda em pasta temporária com painel sintético; nenhuma API real é consultada (a consulta é simulada)."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algamar_ml import config, fluxo, registro
from tests.test_horizonte import painel_sintetico

HOJE = pd.Timestamp("2011-01-15").date()


def api_fora(_):
    raise ConnectionError("API simulada indisponível")


def api_ok(_):
    return (pd.Timestamp("2000-01-01").date(), pd.Timestamp("2010-12-31").date())


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("algamar")
    cfg = config.carrega()
    cfg["caminhos"].update(painel=d / "painel.parquet", brutos=d / "brutos", modelos=d / "modelos", croco_mensal=d / "nao_existe.parquet")
    cfg["pipeline"].update(saidas=str(d / "saidas"), logs=str(d / "logs"), hoje="")
    cfg["split"] = {"fim_treino": "2006-12", "inicio_validacao": "2007-01", "fim_validacao": "2008-12", "inicio_teste": "2009-01"}
    cfg["modelo"].update(n_estimators=30)
    p = painel_sintetico(anos=range(2000, 2011), pontos=((-24.0, -46.8), (-24.5, -46.0), (-23.8, -45.2), (-24.2, -45.0)))
    p.to_parquet(cfg["caminhos"]["painel"])
    (d / "brutos").mkdir()
    status, log = fluxo.executa_treino(copy.deepcopy(cfg), "m1")
    assert status == "OK", log["erros"]
    return cfg, d, p


def roda(cfg, **kw):
    kw.setdefault("hoje", HOJE); kw.setdefault("consulta_api", api_fora)
    return fluxo.executa_previsao(copy.deepcopy(cfg), "m1", **kw)


def com_painel(cfg, d, p, nome):
    c = copy.deepcopy(cfg); c["caminhos"]["painel"] = d / f"{nome}.parquet"
    if isinstance(p, bytes):
        c["caminhos"]["painel"].write_bytes(p)
    else:
        p.to_parquet(c["caminhos"]["painel"])
    return c


def saida(log):
    return Path(log["resultado"]["saida"])


# ---------------------------------------------------------------- fluxo completo
def test_fluxo_completo_gera_saida_padrao_e_log(base):
    cfg, d, _ = base
    status, log = roda(cfg, consulta_api=api_ok)
    assert status == "OK"
    prev = pd.read_parquet(saida(log) / "previsao.parquet")
    assert list(prev.columns) == fluxo.COLUNAS_SAIDA
    assert sorted(prev.horizon.unique()) == list(range(1, 13))
    assert prev.prediction_date.unique().tolist() == ["2010-12"] and prev.target_date.max() == "2011-12"
    assert prev.probabilidade.between(0, 1).all() and (prev.status_ponto == "OK").all()
    assert set(prev.nivel_de_risco) <= {"BAIXO", "MODERADO", "ALTO"}
    assert (prev.classificacao_produto == "HINDCAST_VALIDADO").all()
    meta = json.loads((saida(log) / "previsao.json").read_text(encoding="utf-8"))
    assert meta["schema"] == fluxo.SCHEMA_SAIDA and meta["sha256"]
    assert (saida(log) / "risco_regioes.parquet").exists()
    assert json.loads((Path(cfg["pipeline"]["saidas"]) / "ULTIMA.json").read_text(encoding="utf-8"))["run_id"] == log["run_id"]
    reg = json.loads((Path(cfg["pipeline"]["logs"]) / f"{log['run_id']}.json").read_text(encoding="utf-8"))
    for k in ("timestamp_inicio", "dados", "modelo", "features", "periodo", "resultado", "erros", "config", "commit_git"):
        assert k in reg
    assert reg["dados"]["painel"]["sha256"] and reg["modelo"]["manifesto_sha256"]["modelo.json"]
    assert reg["features"] and reg["periodo"]["emissao_t0"] == "2010-12"
    assert (Path(cfg["pipeline"]["logs"]) / "execucoes.jsonl").read_text(encoding="utf-8").count(log["run_id"]) == 1


def test_reprodutivel(base):
    cfg, _, _ = base
    a = pd.read_parquet(saida(roda(cfg)[1]) / "previsao.parquet")
    b = pd.read_parquet(saida(roda(cfg)[1]) / "previsao.parquet")
    assert np.array_equal(a.probabilidade.to_numpy(), b.probabilidade.to_numpy())


# ---------------------------------------------------------------- modelo versionado
def test_nao_sobrescreve_modelo(base):
    cfg, d, _ = base
    antes = registro.sha256(d / "modelos" / "m1" / "modelo.json")
    status, log = fluxo.executa_treino(copy.deepcopy(cfg), "m1")
    assert status == "ERRO_MODELO" and "não são sobrescritos" in log["erros"][0]
    assert registro.sha256(d / "modelos" / "m1" / "modelo.json") == antes
    with pytest.raises(registro.ErroRegistro):
        registro.registra(None, {}, d / "modelos", "m1")


def test_modelo_alterado_e_recusado(base, tmp_path):
    cfg, d, _ = base
    import shutil
    shutil.copytree(d / "modelos" / "m1", tmp_path / "m1")
    c = copy.deepcopy(cfg); c["caminhos"]["modelos"] = tmp_path
    (tmp_path / "m1" / "modelo.json").write_text("{}", encoding="utf-8")
    status, log = fluxo.executa_previsao(c, "m1", hoje=HOJE, consulta_api=api_fora)
    assert status == "ERRO_MODELO" and not (saida(log) / "previsao.parquet").exists()


# ---------------------------------------------------------------- modos de falha
def test_api_indisponivel_segue_com_dados_locais(base):
    status, log = roda(base[0], consulta_api=api_fora)
    assert status == "OK" and any("API indisponível" in a for a in log["avisos"])


def test_croco_indisponivel_quando_o_modelo_exige(base):
    cfg, d, _ = base
    m, c, _ = registro.carrega(d / "modelos", "m1")
    registro.registra(m, {**c, "flags": {**c["flags"], "use_croco_core": True}}, d / "modelos", "m_croco")
    status, log = fluxo.executa_previsao(copy.deepcopy(cfg), "m_croco", hoje=HOJE, consulta_api=api_fora)
    assert status == "DADOS_INSUFICIENTES" and "CROCO" in log["erros"][0]
    assert not (saida(log) / "previsao.parquet").exists()


def test_croco_nao_exigido_e_registrado(base):
    status, log = roda(base[0])
    assert status == "OK" and log["dados"]["croco"] == "não exigido pelo modelo"


def test_feature_ausente(base):
    cfg, d, _ = base
    m, c, _ = registro.carrega(d / "modelos", "m1")
    registro.registra(m, {**c, "features": c["features"] + ["feature_que_nao_existe"]}, d / "modelos", "m_feat")
    status, log = fluxo.executa_previsao(copy.deepcopy(cfg), "m_feat", hoje=HOJE, consulta_api=api_fora)
    assert status == "DADOS_INSUFICIENTES" and "feature_que_nao_existe" in log["erros"][0]


def test_painel_corrompido(base):
    cfg, d, _ = base
    status, log = roda(com_painel(cfg, d, b"isto nao e parquet", "corrompido"))
    assert status == "ERRO_DADOS" and "corrompido" in log["erros"][0]


def test_arquivo_bruto_corrompido_anula_mes_e_recua(base, tmp_path):
    cfg, d, p = base
    c = copy.deepcopy(cfg); c["caminhos"]["brutos"] = tmp_path
    pasta = tmp_path / "clorofila" / "produto_x"; pasta.mkdir(parents=True)
    (pasta / "clorofila_2010-12.json").write_text(json.dumps({"dataset_id": "produto_x", "tipo": "MY", "ano": 2010, "mes": 12}), encoding="utf-8")
    (pasta / "clorofila_2010-12.nc").write_bytes(b"lixo")
    status, log = roda(c)
    assert status == "ATRASADO" and log["periodo"]["emissao_t0"] == "2010-11"
    assert log["etapas"]["3_alinhamento_temporal"]["variaveis_anuladas"] == ["clorofila 2010-12"]


def test_data_fora_de_ordem_e_reordenada_com_mesmo_resultado(base):
    cfg, d, p = base
    ref = pd.read_parquet(saida(roda(cfg)[1]) / "previsao.parquet")
    status, log = roda(com_painel(cfg, d, p.sample(frac=1, random_state=3), "embaralhado"))
    assert status == "OK" and any("fora de ordem" in a for a in log["avisos"])
    novo = pd.read_parquet(saida(log) / "previsao.parquet")
    assert np.array_equal(ref.probabilidade.to_numpy(), novo.probabilidade.to_numpy())


def test_chave_repetida(base):
    cfg, d, p = base
    status, log = roda(com_painel(cfg, d, pd.concat([p, p.tail(1)]), "duplicado"))
    assert status == "ERRO_DADOS" and "repetidas" in log["erros"][0]


def test_coordenada_invalida_descartada(base):
    cfg, d, p = base
    ruim = p.tail(2).copy(); ruim["latitude"] = [np.nan, -24.0]; ruim["longitude"] = [-45.0, 10.0]; ruim["ano"] = 2010
    status, log = roda(com_painel(cfg, d, pd.concat([p, ruim]), "coord"))
    assert status == "OK" and log["etapas"]["2_validacao"]["coordenadas_invalidas"] == 2


# ---------------------------------------------------------------- nunca inventar previsão
def test_ponto_sem_estado_em_t0_nao_recebe_probabilidade(base):
    cfg, d, p = base
    q = p.copy(); sel = (q.ano == 2010) & (q.mes == 12) & (q.latitude == -24.0)
    q.loc[sel, ["clorofila_media", "clorofila_maxima"]] = np.nan
    status, log = roda(com_painel(cfg, d, q, "ponto_nan"))
    prev = pd.read_parquet(saida(log) / "previsao.parquet")
    sem = prev[prev.latitude == -24.0]
    assert status == "OK" and sem.probabilidade.isna().all() and sem.classe.isna().all()
    assert (sem.nivel_de_risco == "DADOS_INSUFICIENTES").all()
    assert prev[prev.latitude != -24.0].probabilidade.notna().all()


def test_dados_insuficientes_nao_gera_probabilidade(base):
    cfg, d, p = base
    q = p.copy(); sel = (q.ano == 2010) & (q.mes >= 9)
    q.loc[sel, ["clorofila_media", "clorofila_maxima", "temperatura_media"]] = np.nan
    status, log = roda(com_painel(cfg, d, q, "sem_meses"))
    assert status == "DADOS_INSUFICIENTES"
    assert not (saida(log) / "previsao.parquet").exists()
    meta = json.loads((saida(log) / "previsao.json").read_text(encoding="utf-8"))
    assert meta["status"] == "DADOS_INSUFICIENTES" and meta["arquivos"] == {}


def test_dados_desatualizados_demais(base):
    status, log = roda(base[0], hoje=pd.Timestamp("2012-06-15").date())
    assert status == "DADOS_INSUFICIENTES" and not (saida(log) / "previsao.parquet").exists()
