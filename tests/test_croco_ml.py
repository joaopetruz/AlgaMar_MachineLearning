"""Etapa 13.5 — entrada CROCO → ML: schema, join, duplicatas, NaN físico, unidades, coordenadas, cobertura,
origem (proveniência), defasagem temporal e validade espacial; baseline intacto sem CROCO."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from algamar_ml import croco_ml
from algamar_ml.features.horizonte import constroi_linhas, cubo_do_painel, indice_mes, nomes_features

PONTOS = [(-24.0, -46.0, "D2_VALIDO", 80.0), (-24.5, -45.5, "NAO_VALIDADO_COSTEIRO", 15.0),
          (-23.9, -46.3, "NAO_USAR_ESTUARIO_CANAL", 30.0)]
TODAS = [f for g in croco_ml.GRUPOS.values() for f in g]


def croco_mensal(anos=range(2003, 2006), seed=0):
    rng = np.random.default_rng(seed)
    linhas = []
    for lat, lon, flag, h in PONTOS:
        for ano in anos:
            for mes in range(1, 13):
                d = dict(latitude=lat + 1e-7, longitude=lon, ano=ano, mes=mes, validade_espacial=flag, h_d2_m=h)
                for f in set(TODAS):
                    lo, hi = croco_ml._faixa(f)
                    d[croco_ml.coluna_arquivo(f)] = float(rng.uniform(max(lo, -1), min(hi, 30))) if hi > 1 else float(rng.uniform(lo, hi) / 10)
                d["transport_east_mean"] = 100.0 * (ano - 2000) + mes   # marca o mês para testar defasagem
                for f, z in croco_ml.PROFUNDIDADE.items():
                    if h < z:
                        d[croco_ml.coluna_arquivo(f)] = np.nan
                linhas.append(d)
    return pd.DataFrame(linhas)


def painel(anos=range(2002, 2006)):
    rng = np.random.default_rng(1)
    return pd.DataFrame([dict(latitude=lat, longitude=lon, ano=a, mes=m, clorofila_media=rng.gamma(2, .5),
                              clorofila_maxima=rng.gamma(2, .9), temperatura_media=22.0, salinidade_media=35.0)
                         for lat, lon, _, _ in PONTOS for a in anos for m in range(1, 13)])


def test_grupos_e_flags():
    assert croco_ml.GRUPOS["CROCO_CORE"][:2] == croco_ml.GRUPOS["CROCO_TS"]
    assert set(croco_ml.GRUPOS["CROCO_VERTICAL"]) == {"temperature_20m", "salinity_20m", "temperature_50m", "salinity_50m",
                                                       "temperature_bottom", "salinity_bottom"}
    assert croco_ml.features_ativas({}) == []
    f = croco_ml.features_ativas({"use_croco_ts": True, "use_croco_core": True})
    assert f == ["temperature_surface", "salinity_surface", "mld", "n2_max"]          # sem repetir o TS
    c = croco_ml.contrato_13()
    assert all(c[x]["status"] == "NAO_VALIDADA_OBS" for x in croco_ml.GRUPOS["CROCO_DYNAMICS"] + croco_ml.GRUPOS["CROCO_CURRENT"])
    assert all(c[x]["status"] in ("VALIDADA", "DERIVADA_VALIDADA") for x in croco_ml.GRUPOS["CROCO_CORE"] + croco_ml.GRUPOS["CROCO_VERTICAL"])


def test_baseline_sem_croco_nao_muda():
    base = nomes_features({"use_api_features": True, "use_salinity": False, "use_baseline_features": True}, ["croco_x_t0"])
    assert not any(n.startswith("croco_") for n in base)
    com = nomes_features({"use_api_features": True, "use_salinity": False, "use_croco_core": True}, ["croco_x_t0"])
    assert com == base + ["croco_x_t0"]


def test_schema_e_coluna_faltando():
    t = croco_mensal().drop(columns="mld_mean")
    with pytest.raises(ValueError, match="schema"):
        croco_ml.carrega(t, croco_ml.GRUPOS["CROCO_CORE"])
    with pytest.raises(ValueError, match="schema"):
        croco_ml.carrega(croco_mensal().drop(columns="validade_espacial"), ["mld"])


def test_duplicata_apos_round5_recusada():
    t = croco_mensal()
    t2 = t.iloc[:1].copy()
    t2["latitude"] += 3e-6                                   # igual após round(5)
    with pytest.raises(ValueError, match="repetidas"):
        croco_ml.carrega(pd.concat([t, t2]), ["mld"])


def test_nan_fisico_nao_pode_ser_preenchido():
    t, _ = croco_ml.carrega(croco_mensal(), croco_ml.GRUPOS["CROCO_VERTICAL"])
    raso = t["h_d2_m"] < 50
    assert t.loc[raso, "temperature_50m_mean"].isna().all() and t.loc[~raso, "temperature_50m_mean"].notna().all()
    falso = croco_mensal()
    falso.loc[falso.h_d2_m < 50, "temperature_50m_mean"] = 18.0          # alguém "imputou"
    with pytest.raises(ValueError, match="NaN físico"):
        croco_ml.carrega(falso, croco_ml.GRUPOS["CROCO_VERTICAL"])


def test_unidade_do_metadado_e_faixa(tmp_path):
    t = croco_mensal()
    tab = pa.Table.from_pandas(t, preserve_index=False)
    campos = [f.with_metadata({b"unidade": b"K"}) if f.name == "temperature_surface_mean" else f for f in tab.schema]
    caminho = tmp_path / "c.parquet"
    pq.write_table(tab.cast(pa.schema(campos)), caminho)
    with pytest.raises(ValueError, match="unidade"):
        croco_ml.carrega(caminho, ["temperature_surface"])
    ruim = croco_mensal()
    ruim.loc[0, "salinity_surface_mean"] = 60.0
    with pytest.raises(ValueError, match="faixa"):
        croco_ml.carrega(ruim, ["salinity_surface"])


def test_validade_desconhecida_e_coordenadas_e_cobertura():
    t = croco_mensal()
    t.loc[0, "validade_espacial"] = "TALVEZ"
    with pytest.raises(ValueError, match="validade"):
        croco_ml.carrega(t, ["mld"])
    t = croco_mensal()
    t = t[~((t.ano == 2004) & (t.mes == 6))]
    pontos = pd.DataFrame({"latitude": [-24.0, -24.5, -23.9, -25.0], "longitude": [-46.0, -45.5, -46.3, -45.0]})
    _, rel = croco_ml.carrega(t, ["mld"], pontos_ml=pontos)
    assert rel["cobertura"]["meses_faltando"] == ["2004-06"] and rel["pontos_ml_sem_croco"] == 1
    assert rel["validade"]["NAO_USAR_ESTUARIO_CANAL"] == 1


def test_proveniencia_por_feature():
    prov = {"build_croco": "build_d2_algamar_final_v1", "sha256_grade": "e7346752", "commit_croco": "ab63b583"}
    _, rel = croco_ml.carrega(croco_mensal(), ["mld", "w_10m"], proveniencia=prov)
    p = rel["proveniencia"]["croco_mld_t0"]
    assert p["source"] == "CROCO" and p["unit"] == "m" and p["status_validation"] == "VALIDADA"
    assert p["grid_version"]["build_croco"] == "build_d2_algamar_final_v1" and p["feature_version"] == "13.0.0"
    assert rel["proveniencia"]["croco_w_10m_t0"]["status_validation"] == "NAO_VALIDADA_OBS"


def test_join_um_para_um_defasagem_por_horizonte_e_nao_usar():
    """A física entra como o estado de t0: no alvo τ = t0 + h, o valor é do mês t0 (defasagem h), nunca de τ."""
    t, _ = croco_ml.carrega(croco_mensal(), ["transport_east", "mld"])
    cubo = cubo_do_painel(painel())
    mats, rel = croco_ml.para_cubo(t, cubo, ["transport_east", "mld"])
    assert rel["pontos_nao_usar"] == 1
    t0 = indice_mes(2004, 7)
    lin = constroi_linhas(cubo, 1.5, [t0], horizontes=range(1, 13), fisica=mats)
    validos = lin[lin.latitude.round(1) == -24.0]
    assert (validos["croco_transport_east_t0"] == 100.0 * 4 + 7).all()            # sempre o mês t0, em todo h
    assert (validos["tau"] - validos["t0"] == validos["h"]).all()
    nao_usar = lin[lin.latitude.round(1) == -23.9]
    assert nao_usar["croco_mld_t0"].isna().all()                                           # NAO_USAR nunca entra
    v = croco_ml.validade_por_ponto(t, cubo)
    assert sorted(v) == sorted(["D2_VALIDO", "NAO_VALIDADO_COSTEIRO", "NAO_USAR_ESTUARIO_CANAL"])


def test_croco_posterior_a_t0_nao_muda_nenhuma_feature():
    t, _ = croco_ml.carrega(croco_mensal(), ["temperature_surface", "mld", "ssh"])
    alt = t.copy()
    fut = alt.ano * 12 + alt.mes - 1 > indice_mes(2004, 7)
    alt.loc[fut, ["temperature_surface_mean", "mld_mean", "ssh_mean"]] = 1.0
    cubo = cubo_do_painel(painel())
    a = constroi_linhas(cubo, 1.5, [indice_mes(2004, 7)], fisica=croco_ml.para_cubo(t, cubo, ["temperature_surface", "mld", "ssh"])[0])
    b = constroi_linhas(cubo, 1.5, [indice_mes(2004, 7)], fisica=croco_ml.para_cubo(alt, cubo, ["temperature_surface", "mld", "ssh"])[0])
    cols = [c for c in a.columns if c.startswith("croco_")]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_mes_sem_croco_fica_nan():
    t = croco_mensal()
    t = t[~((t.ano == 2004) & (t.mes == 7))]
    t, _ = croco_ml.carrega(t, ["mld"])
    cubo = cubo_do_painel(painel())
    lin = constroi_linhas(cubo, 1.5, [indice_mes(2004, 7)], fisica=croco_ml.para_cubo(t, cubo, ["mld"])[0])
    assert lin["croco_mld_t0"].isna().all()
