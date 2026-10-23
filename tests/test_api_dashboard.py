import pandas as pd
import pytest

from algamar_ml import api_dashboard

CANANEIA = (-25.10, -47.85)
UBATUBA = (-23.47, -44.98)
MAR_ABERTO = (-25.6, -46.0)     # longe de todo município: só entra no resumo


@pytest.fixture()
def fontes(monkeypatch):
    def ponto(lat_lon, regiao, h, p):
        return dict(target_date=f"2026-{9 + h:02d}", forecast_horizon=h, latitude=lat_lon[0], longitude=lat_lon[1],
                    region=regiao, bloom_probability=p, climatological_probability=0.3)
    sul, norte = "extremo sul (Cananéia–Ilha Comprida–Iguape)", "norte (São Sebastião–Ilhabela–Caraguatatuba–Ubatuba)"
    prev = pd.DataFrame([ponto(CANANEIA, sul, 1, 0.9), ponto(CANANEIA, sul, 1, 0.5), ponto(CANANEIA, sul, 2, 0.1),
                         ponto(CANANEIA, sul, 2, float("nan")), ponto(UBATUBA, norte, 1, 0.3), ponto(MAR_ABERTO, sul, 1, 0.6)])
    meta = {"issue_month": "2026-09", "model_version": "t", "gerado_em": "x"}
    obs = pd.DataFrame([dict(latitude=CANANEIA[0], longitude=CANANEIA[1], ano=2026, mes=9, floracao=f, temperatura_media=20.0,
                             salinidade_media=34.5, clorofila_media=1.2, clorofila_maxima=3.0)
                        for f in (1.0, 1.0, 1.0, 0.0)])
    obs["macro_region"], obs["municipio"] = "Litoral Sul", api_dashboard.municipio(obs["latitude"], obs["longitude"])
    monkeypatch.setattr(api_dashboard, "dados_previsao", lambda: (prev, meta, None))
    monkeypatch.setattr(api_dashboard, "limiares", lambda versao: (0.48, 0.24))
    monkeypatch.setattr(api_dashboard, "observado", lambda: obs)


def test_regioes_do_ml_viram_as_tres_do_dashboard():
    assert api_dashboard.regiao_dashboard("sul (Peruíbe–Itanhaém–Mongaguá)") == "Baixada Santista"
    assert api_dashboard.regiao_dashboard("extremo sul (Cananéia–Ilha Comprida–Iguape)") == "Litoral Sul"
    assert api_dashboard.regiao_dashboard("norte (São Sebastião–Ilhabela–Caraguatatuba–Ubatuba)") == "Litoral Norte"


def test_ponto_vai_para_o_municipio_mais_proximo_ou_para_nenhum():
    nomes = api_dashboard.municipio([CANANEIA[0], -24.03, MAR_ABERTO[0]], [CANANEIA[1], -46.33, MAR_ABERTO[1]])
    assert list(nomes) == ["Cananéia", "Santos", None]


def test_quatro_niveis_com_os_limiares_do_modelo():
    assert [api_dashboard.nivel(p, 0.48, 0.24) for p in (0.85, 0.80, 0.5, 0.48, 0.3, 0.24, 0.1)] == \
        ["MUITO ALTO", "MUITO ALTO", "ALTO", "ALTO", "MODERADO", "MODERADO", "BAIXO"]
    assert [f["nivel"] for f in api_dashboard.faixas(0.48, 0.24)] == list(api_dashboard.NIVEIS)


def test_filtro_aceita_os_nomes_que_o_frontend_manda():
    assert [api_dashboard.nivel_do_filtro(t) for t in ("MÉDIO", "moderado", "Muito alto", "MUITO_ALTO", "ALTO", "Baixo")] == \
        ["MODERADO", "MODERADO", "MUITO ALTO", "MUITO ALTO", "ALTO", "BAIXO"]


def test_um_registro_por_municipio_com_a_sua_localizacao(fontes):
    r = api_dashboard.monta()
    assert r["success"] and {d["tipo"] for d in r["data"]} == {"observado", "previsao"}
    can = next(d for d in r["data"] if d["id"] == "Cananéia-2026-10")
    assert (can["latitude"], can["longitude"]) == CANANEIA and can["macro_region"] == "Litoral Sul"
    assert can["probability"] == 0.7 and can["risk_level"] == "ALTO" and can["high_risk_points"] == 2
    assert next(d for d in r["data"] if d["id"] == "Cananéia-2026-11")["probability"] == 0.1   # NaN fica de fora
    assert next(d for d in r["data"] if d["id"] == "Cananéia-2026-09")["probability"] == 0.75   # observado
    assert len({(d["latitude"], d["longitude"]) for d in r["data"]}) == 2                        # Cananéia e Ubatuba
    # o mar aberto não vira registro, mas conta no resumo
    assert r["summary"]["total_monitored"] == 4 and r["summary"]["high_risk_count"] == 3
    assert r["summary"]["general_status"] == "ALERTA" and r["summary"]["target_month"] == "2026-10"
    assert [(t["year"], t["month"], t["tipo"]) for t in r["trends"]] == [(2026, 9, "observado"), (2026, 10, "previsao")]


def test_filtros(fontes):
    r = api_dashboard.monta(region="Litoral Norte")
    assert {d["region"] for d in r["data"]} == {"Ubatuba"} and r["summary"]["general_status"] == "ATENÇÃO"
    assert {d["region"] for d in api_dashboard.monta(region="Cananéia")["data"]} == {"Cananéia"}
    assert {d["month"] for d in api_dashboard.monta(month=9)["data"]} == {9}
    assert {d["risk_level"] for d in api_dashboard.monta(risk_level="BAIXO")["data"]} == {"BAIXO"}
