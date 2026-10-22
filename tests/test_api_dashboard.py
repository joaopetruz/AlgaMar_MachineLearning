import pandas as pd
import pytest

from algamar_ml import api_dashboard


@pytest.fixture()
def fontes(monkeypatch):
    prev = pd.DataFrame([
        dict(target_date=f"2026-{9 + h:02d}", forecast_horizon=h, latitude=-25.2, longitude=-47.5,
             region="extremo sul (Cananéia–Ilha Comprida–Iguape)", bloom_probability=p, climatological_probability=0.3)
        for h, p in [(1, 0.9), (1, 0.5), (2, 0.1), (2, float("nan"))]
    ] + [dict(target_date="2026-10", forecast_horizon=1, latitude=-23.6, longitude=-45.0,
              region="norte (São Sebastião–Ilhabela–Caraguatatuba–Ubatuba)", bloom_probability=0.3,
              climatological_probability=0.2)])
    meta = {"issue_month": "2026-09", "model_version": "t", "gerado_em": "x"}
    obs = pd.DataFrame([dict(region="Litoral Sul", ano=2026, mes=9, latitude=-25.2, longitude=-47.5, probability=0.75,
                             high_risk_points=3, temperature_celsius=20.0, salinity_psu=34.5, chlorophyll_mg_m3=1.2,
                             chlorophyll_max_mg_m3=3.0)])
    monkeypatch.setattr(api_dashboard, "dados_previsao", lambda: (prev, meta, None))
    monkeypatch.setattr(api_dashboard, "limiares", lambda versao: (0.48, 0.24))
    monkeypatch.setattr(api_dashboard, "observado", lambda: obs)


def test_regioes_do_ml_viram_as_tres_do_dashboard():
    assert api_dashboard.regiao_dashboard("sul (Peruíbe–Itanhaém–Mongaguá)") == "Baixada Santista"
    assert api_dashboard.regiao_dashboard("extremo sul (Cananéia–Ilha Comprida–Iguape)") == "Litoral Sul"
    assert api_dashboard.regiao_dashboard("norte (São Sebastião–Ilhabela–Caraguatatuba–Ubatuba)") == "Litoral Norte"


def test_niveis_usam_os_nomes_do_frontend():
    assert [api_dashboard.nivel(p, 0.48, 0.24) for p in (0.5, 0.3, 0.1)] == ["ALTO", "MÉDIO", "BAIXO"]


def test_resposta_tem_observado_e_previsao_atuais(fontes):
    r = api_dashboard.monta()
    assert r["success"] and {d["tipo"] for d in r["data"]} == {"observado", "previsao"}
    sul_out = next(d for d in r["data"] if d["id"] == "Litoral Sul-2026-10")
    assert sul_out["probability"] == 0.7 and sul_out["risk_level"] == "ALTO" and sul_out["high_risk_points"] == 2
    assert (sul_out["latitude"], sul_out["longitude"]) == api_dashboard.MARCADORES["Litoral Sul"]
    # ponto sem probabilidade fica de fora
    assert next(d for d in r["data"] if d["id"] == "Litoral Sul-2026-11")["probability"] == 0.1
    assert r["summary"]["total_monitored"] == 3 and r["summary"]["high_risk_count"] == 2
    assert r["summary"]["general_status"] == "ALERTA" and r["summary"]["target_month"] == "2026-10"
    assert [(t["year"], t["month"], t["tipo"]) for t in r["trends"]] == [(2026, 9, "observado"), (2026, 10, "previsao")]


def test_filtros(fontes):
    r = api_dashboard.monta(region="Litoral Norte")
    assert {d["region"] for d in r["data"]} == {"Litoral Norte"} and r["summary"]["general_status"] == "ATENÇÃO"
    assert {d["month"] for d in api_dashboard.monta(month=9)["data"]} == {9}
    assert {d["risk_level"] for d in api_dashboard.monta(risk_level="BAIXO")["data"]} == {"BAIXO"}
