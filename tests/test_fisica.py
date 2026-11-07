import numpy as np
import pandas as pd
import pytest
import xarray as xr

from algamar_ml.features.horizonte import constroi_linhas, cubo_do_painel, indice_mes
from algamar_ml.fisica import contrato as ctr
from algamar_ml.fisica import feature_store
from algamar_ml.fisica.alinhamento import mensal
from algamar_ml.fisica.derivadas import deriva
from algamar_ml.fisica.fontes import FonteGradeUV, FonteTabelaContrato13

from tests.test_horizonte import painel_sintetico


def test_contrato_tem_campos_obrigatorios_e_conjuntos():
    c = ctr.carrega()
    for campo in ("tipo", "unidade", "origem", "descricao", "frequencia", "resolucao_espacial", "obrigatoria",
                  "valor_min", "valor_max", "ausentes", "status_validacao"):
        assert campo in c.columns
    assert (c["valor_min"] < c["valor_max"]).all()
    assert {"u", "v", "velocidade", "vorticidade", "divergencia", "energia_cinetica", "pressao"} <= set(c.index)
    assert ctr.conjunto("NS_CORE") == ["temperature_surface", "salinity_surface", "mld", "n2_max"]


def test_validacao_recusa_faixa_chave_repetida_e_tipo():
    t = pd.DataFrame({"latitude": [-24.0, -24.0], "longitude": [-46.0, -46.0], "ano": [2017, 2017], "mes": [11, 11],
                      "velocidade": [0.5, 9.0]})
    problemas = ctr.valida(t)
    assert any("repetidas" in p for p in problemas) and any("velocidade" in p for p in problemas)
    t2 = pd.DataFrame({"latitude": [-24.0], "longitude": [-46.0], "ano": [2017], "mes": [11], "u": ["x"]})
    assert any("tipo" in p for p in ctr.valida(t2))
    assert any("ausente" in p for p in ctr.valida(t.iloc[:1], exige_conjunto="NS_CORE"))


def test_derivadas_em_campo_de_rotacao_solida():
    lat = np.linspace(-24.1, -23.9, 21)
    lon = np.linspace(-46.1, -45.9, 21)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    y = (LA - lat.mean()) * 111_195.0
    x = (LO - lon.mean()) * 111_195.0 * np.cos(np.radians(LA))
    omega = 1e-5
    u, v = -omega * y, omega * x            # rotação sólida anti-horária: ζ = 2ω, δ = 0
    d = deriva(u, v, lat, lon)
    centro = (slice(5, -5), slice(5, -5))
    assert np.allclose(d["vorticidade"][centro], 2 * omega, rtol=0.02)
    assert np.allclose(d["divergencia"][centro], 0, atol=2e-7)
    assert (d["okubo_weiss"][centro] < 0).all()        # dominado por vorticidade
    assert np.allclose(d["energia_cinetica"], 0.5 * (u ** 2 + v ** 2))
    # direção para onde vai: u>0, v=0 → leste (sin=1)
    d2 = deriva(np.ones((3, 3)), np.zeros((3, 3)), lat[:3], lon[:3])
    assert np.allclose(d2["direcao_sin"], 1) and np.allclose(d2["direcao_cos"], 0, atol=1e-12)


def test_mensal_exige_utc_e_respeita_cobertura():
    dias = pd.date_range("2017-11-01", "2017-11-30", freq="D", tz="UTC")
    t = pd.DataFrame({"time": dias, "latitude": -24.0, "longitude": -46.0, "u": 0.1})
    m = mensal(t, ["u"])
    assert m["u"].iloc[0] == pytest.approx(0.1) and m["cobertura_dias"].iloc[0] == 1.0
    curto = mensal(t.iloc[:10], ["u"])
    assert np.isnan(curto["u"].iloc[0])  # 10/30 dias < 80 %
    with pytest.raises(ValueError):
        mensal(t.assign(time=dias.tz_localize(None)), ["u"])
    with pytest.raises(ValueError):
        mensal(pd.concat([t, t.iloc[:1]]), ["u"])


def _pontos_ml():
    return pd.DataFrame({"latitude": [-24.0, -24.5, -23.8], "longitude": [-46.0, -45.5, -45.2]})


def test_feature_store_de_grade_uv_associa_por_vizinho_e_grava_manifesto(tmp_path):
    lat = np.arange(-24.6, -23.7, 0.01)
    lon = np.arange(-46.1, -45.1, 0.01)
    tempo = pd.date_range("2017-11-01", periods=30, freq="D")
    u = np.full((len(tempo), len(lat), len(lon)), 0.2)
    v = np.full_like(u, -0.1)
    ds = xr.Dataset({"u": (("time", "latitude", "longitude"), u), "v": (("time", "latitude", "longitude"), v)},
                    coords={"time": tempo, "latitude": lat, "longitude": lon})
    ds.to_netcdf(tmp_path / "ns.nc")
    fonte = FonteGradeUV(tmp_path / "ns.nc", nome="ns_teste", versao="0.1")
    tabela, manifesto = feature_store.ingere(fonte, _pontos_ml(), tmp_path / "store", dist_max_km=2.0)
    assert manifesto["pontos"] == 3 and manifesto["ultimo_mes"] == "2017-11"
    assert tabela["velocidade"].iloc[0] == pytest.approx(np.hypot(0.2, 0.1))
    assert (tabela["dist_associacao_km"] <= 2.0).all()
    lida, man = feature_store.le(tmp_path / "store", "ns_teste")
    assert man["fonte"] == "ns_teste" and len(lida) == 3


def test_feature_store_recusa_dado_invalido(tmp_path):
    t = pd.DataFrame({"latitude": [-24.0], "longitude": [-46.0], "ano": [2017], "mes": [11], "speed_of_mean_current": [50.0]})
    t.to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError):
        feature_store.ingere(FonteTabelaContrato13(tmp_path / "x.csv"), _pontos_ml(), tmp_path / "store")


def test_fisica_entra_como_estado_em_t0_e_nao_no_mes_alvo(tmp_path):
    p = painel_sintetico()
    cubo = cubo_do_painel(p)
    pts = cubo.pontos
    linhas = []
    for ano in range(2000, 2006):
        for mes in range(1, 13):
            for _, r in pts.iterrows():
                linhas.append(dict(latitude=r.latitude, longitude=r.longitude, ano=ano, mes=mes,
                                   temperature_surface_mean=float(ano - 2000) + mes / 100, salinity_surface_mean=35.0,
                                   mld_mean=20.0, n2_max_mean=1e-4, validade_espacial="D2_VALIDO"))
    pd.DataFrame(linhas).to_parquet(tmp_path / "croco.parquet")
    feature_store.ingere(FonteTabelaContrato13(tmp_path / "croco.parquet"), pts, tmp_path / "store")
    tabela, _ = feature_store.le(tmp_path / "store", "croco_d2")
    mats = feature_store.para_cubo(tabela, cubo, ctr.conjunto("NS_CORE"))
    t0 = indice_mes(2003, 5)
    l = constroi_linhas(cubo, 1.5, [t0], horizontes=[3], fisica=mats)
    assert np.allclose(l["ns_temperature_surface_t0"], 3.05)  # valor de t0, não de t0+3
