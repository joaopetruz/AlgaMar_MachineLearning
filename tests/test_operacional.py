"""Etapa 13.4 — dados atuais: timestamp, fuso, duplicatas, coordenadas, unidades, ausentes, atualização,
faixa física, alinhamento com a grade do ML, fallback e proveniência."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from algamar_ml import operacional as op
from algamar_ml.features.horizonte import E0_FORECAST_FEATURES, _NOMES_API, _NOMES_HIST

CFG = {"area": {"lon_min": -47.0, "lon_max": -44.8, "lat_min": -25.3, "lat_max": -23.3},
       "regioes": {"setores": [{"nome": "sul", "lon_max": -46.6}, {"nome": "centro", "lon_max": -45.6}, {"nome": "norte", "lon_max": 0.0}]}}


def grade():
    lats = np.round(np.arange(-25.25, -23.4, 0.25), 5)
    lons = np.round(np.arange(-46.95, -44.85, 0.25), 5)
    return pd.DataFrame([(a, b) for a in lats for b in lons], columns=["latitude", "longitude"])


def painel(meses=(("2026", 7), ("2026", 8), ("2026", 9)), seed=0):
    rng = np.random.default_rng(seed)
    g = grade()
    partes = []
    for ano, mes in meses:
        t = g.copy()
        t["ano"], t["mes"] = int(ano), mes
        t["clorofila_media"] = rng.gamma(2, 0.5, len(t))
        t["clorofila_maxima"] = t["clorofila_media"] * 1.7
        t["temperatura_media"] = 22 + rng.normal(0, 1, len(t))
        t["salinidade_media"] = 35 + rng.normal(0, 0.3, len(t))
        for v in ("clorofila", "temperatura", "salinidade"):
            t[f"fonte_{v}"] = f"{v}_MY"
        partes.append(t)
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------- timestamp e fuso
def test_tempos_diarios_ok_utc_sem_fuso():
    t = pd.date_range("2026-09-01", "2026-09-30", freq="D").to_numpy()
    r = op.valida_tempos_diarios(t, 2026, 9)
    assert r["status"] == "OK" and r["cobertura"] == 1.0


def test_fuso_diferente_de_utc_e_recusado():
    t = pd.date_range("2026-09-01", periods=30, freq="D", tz="America/Sao_Paulo")
    assert op.valida_tempos_diarios(t, 2026, 9)["status"] == "FALHA"


def test_tempo_duplicado_fora_do_mes_e_cobertura_baixa():
    base = list(pd.date_range("2026-09-01", periods=20, freq="D"))
    assert op.valida_tempos_diarios(base + [base[3]], 2026, 9)["status"] == "FALHA"            # duplicata
    assert op.valida_tempos_diarios(base + [pd.Timestamp("2026-10-01")], 2026, 9)["status"] == "FALHA"   # fora do mês
    assert op.valida_tempos_diarios(base[:10], 2026, 9)["status"] == "FALHA"                  # 33 % dos dias


# ---------------------------------------------------------------- tabela mensal × grade do ML
def test_mes_valido():
    r = op.valida_mes(painel(), grade(), 2026, 9, CFG)
    assert r["status"] == "OK", r


def test_duplicatas_e_coordenadas_fora():
    p = painel()
    dup = pd.concat([p, p[(p.ano == 2026) & (p.mes == 9)].iloc[:2]])
    assert op.valida_mes(dup, grade(), 2026, 9, CFG)["status"] == "FALHA"
    fora = p.copy()
    sel = (fora.ano == 2026) & (fora.mes == 9)
    fora.loc[fora[sel].index[0], "longitude"] = -43.0
    assert op.valida_mes(fora, grade(), 2026, 9, CFG)["status"] == "FALHA"


def test_ponto_fora_da_grade_do_ml():
    p = painel()
    sel = p[(p.ano == 2026) & (p.mes == 9)].index[0]
    p.loc[sel, "latitude"] += 0.013
    r = op.valida_mes(p, grade(), 2026, 9, CFG)
    assert dict((c[0], c[1]) for c in r["checagens"])["alinhado_grade_ml"] == "FALHA"


def test_unidade_kelvin_detectada():
    p = painel()
    sel = (p.ano == 2026) & (p.mes == 9)
    p.loc[sel, "temperatura_media"] += 273.15
    r = op.valida_mes(p, grade(), 2026, 9, CFG)
    assert r["por_variavel"]["temperatura_media"]["status"] == "FALHA"


def test_faixa_fisica():
    p = painel()
    sel = p[(p.ano == 2026) & (p.mes == 9)].index[:3]
    p.loc[sel, "salinidade_media"] = 55.0
    assert op.valida_mes(p, grade(), 2026, 9, CFG)["por_variavel"]["salinidade_media"]["status"] == "FALHA"


def test_ausentes_aviso_e_falha_e_regiao_insuficiente():
    p = painel()
    sel = (p.ano == 2026) & (p.mes == 9)
    idx = p[sel].index
    p.loc[idx[: int(0.2 * len(idx))], "clorofila_media"] = np.nan
    r = op.valida_mes(p, grade(), 2026, 9, CFG)
    assert r["por_variavel"]["clorofila_media"]["status"] == "AVISO"
    p.loc[idx, "temperatura_media"] = np.nan
    assert op.valida_mes(p, grade(), 2026, 9, CFG)["por_variavel"]["temperatura_media"]["status"] == "FALHA"
    q = painel()
    norte = (q.ano == 2026) & (q.mes == 9) & (q.longitude > -45.6)
    q.loc[norte, "clorofila_media"] = np.nan
    r = op.valida_mes(q, grade(), 2026, 9, CFG)
    assert r["regioes"]["norte"]["status"] == "INSUFICIENTE" and r["status"] != "FALHA"


# ---------------------------------------------------------------- atualização temporal e fallback
def test_emissao_no_ultimo_mes_completo():
    e = op.decide_emissao(painel(), grade(), CFG, dt.date(2026, 10, 9))
    assert e["issue_month"] == "2026-09" and e["status"] == "OK" and e["atraso_meses"] == 0


def test_mes_faltando_recua_e_marca_atraso():
    p = painel()
    p = p[~((p.ano == 2026) & (p.mes == 9))]                     # API não entregou setembro
    e = op.decide_emissao(p, grade(), CFG, dt.date(2026, 10, 9))
    assert e["issue_month"] == "2026-08" and e["status"] == "ATRASADO" and e["atraso_meses"] == 1
    assert e["tentativas"][0] == {"mes": "2026-09", "status_validacao": "FALHA"}


def test_dado_velho_demais_marca_desatualizado_ou_nao_emite():
    e = op.decide_emissao(painel(), grade(), CFG, dt.date(2027, 1, 5), max_recuo=6)
    assert e["issue_month"] == "2026-09" and e["status"] == "DESATUALIZADO"
    assert op.decide_emissao(painel(), grade(), CFG, dt.date(2027, 6, 1))["status"] == "SEM_EMISSAO"


def test_variavel_ausente_bloqueia_mes_sem_inventar():
    p = painel()
    sel = (p.ano == 2026) & (p.mes == 9)
    p.loc[sel, "temperatura_media"] = np.nan
    e = op.decide_emissao(p, grade(), CFG, dt.date(2026, 10, 9))
    assert e["issue_month"] == "2026-08"                          # recua; setembro não é preenchido
    assert p.loc[sel, "temperatura_media"].isna().all()


# ---------------------------------------------------------------- disponibilidade em t0 e proveniência
@pytest.mark.parametrize("h", range(1, 13))
def test_todas_as_features_disponiveis_em_t0(h):
    t0 = 2026 * 12 + 8
    for f in set(_NOMES_API) | set(_NOMES_HIST) | set(E0_FORECAST_FEATURES):
        assert op.disponivel_em_t0(f, t0, h), f


def test_proveniencia_registra_fonte_por_feature():
    p = painel()
    p.loc[(p.mes == 9), "fonte_temperatura"] = "SST_NRT"
    t0 = 2026 * 12 + 8
    prov = op.proveniencia_features(p, ["sst_t0", "sst_t0m1", "chl_media_3m", "latitude"], t0)
    assert prov["sst_t0"]["fontes"] == ["SST_NRT"] and prov["sst_t0m1"]["fontes"] == ["temperatura_MY"]
    assert prov["chl_media_3m"]["n_meses"] == 3 and prov["chl_media_3m"]["todos_ate_t0"]
    assert prov["latitude"]["variavel"] == "fixo"


def test_catalogo_classificado():
    c = op.catalogo_df()
    assert set(c.classe) <= {"OBSERVACAO", "ANALISE", "REANALISE", "FORECAST", "CLIMATOLOGIA"}
    assert c.loc[c.dataset_id.str.contains("anfc"), "classe"].iloc[0] == "FORECAST"
    assert c.loc[c.nome.str.startswith("ERA5"), "classe"].iloc[0] == "REANALISE"
