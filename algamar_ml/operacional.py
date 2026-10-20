"""Etapa 13.4 — dados ambientais atuais: catálogo classificado, validação, regras de fallback e proveniência.

Regra de ouro: no forecast operacional só entra o que existe NA DATA DA PREVISÃO. Nada é inventado: falta vira NaN
com flag, ou a emissão recua para o último mês válido, sempre registrado em `qualidade`.

Classes (obrigatórias):
  OBSERVACAO   medição direta (in situ, eventos);
  ANALISE      produto L4 baseado em observações (satélite + in situ, interpolado/gap-filled), sem modelo dinâmico;
  REANALISE    modelo + assimilação reprocessado (GLORYS, ERA5);
  FORECAST     previsão de modelo (GLORYS ANFC além da data de hoje);
  CLIMATOLOGIA estatística de anos passados.
Situação "MY/REP" = versão reprocessada/consolidada (chega com atraso); "NRT" = versão de tempo quase real.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from algamar_ml.espacial import regiao

# ------------------------------------------------------------------ catálogo
@dataclass(frozen=True)
class Fonte:
    nome: str
    dataset_id: str
    variavel: str            # nome no painel (clorofila, temperatura, salinidade, ...)
    unidade_origem: str
    unidade_ml: str
    classe: str              # OBSERVACAO | ANALISE | REANALISE | FORECAST | CLIMATOLOGIA
    versao: str              # MY/REP | NRT | ANFC | —
    res_temporal: str
    res_espacial: str
    latencia: str            # atraso típico até o dado existir
    uso: str                 # TREINO | INFERENCIA | TREINO+INFERENCIA | NAO_USADO | FUTURO
    observacao: str = ""


CATALOGO = [
    Fonte("CHL L4 gap-free (GlobColour) MY", "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D", "clorofila",
          "mg m-3", "mg m-3", "ANALISE", "MY/REP", "diária", "4 km (1/24°)", "~dias a semanas (parte interina)",
          "TREINO+INFERENCIA", "satélite multissensor, lacunas preenchidas por interpolação"),
    Fonte("CHL L4 gap-free (GlobColour) NRT", "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D", "clorofila",
          "mg m-3", "mg m-3", "ANALISE", "NRT", "diária", "4 km", "~1 dia; catálogo mantém só ~1 semana",
          "INFERENCIA (reserva)", "usado só se o MY não cobrir o mês"),
    Fonte("OSTIA SST REP", "METOFFICE-GLO-SST-L4-REP-OBS-SST", "temperatura", "K", "degC", "ANALISE", "MY/REP", "diária",
          "0,05°", "~6 meses", "TREINO+INFERENCIA", "SST foundation (sem ciclo diurno)"),
    Fonte("OSTIA SST NRT", "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2", "temperatura", "K", "degC", "ANALISE", "NRT", "diária",
          "0,05°", "~1 dia", "INFERENCIA", "viés NRT−REP em 2024: +0,07 °C, r 0,998 (consistente)"),
    Fonte("SSS multi-obs MY", "cmems_obs-mob_glo_phy-sss_my_multi_P1D", "salinidade", "PSU", "PSU", "ANALISE", "MY/REP",
          "diária", "0,125°", "~1–2 anos (termina em 2024-12-15)", "TREINO", "satélite (SMOS/SMAP) + in situ"),
    Fonte("SSS multi-obs NRT", "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D", "salinidade", "PSU", "PSU", "ANALISE", "NRT",
          "diária", "0,125°", "~1 semana", "NAO_USADO", "viés NRT−MY em 2024: +0,60 PSU, r 0,61 → desligada (use_salinity=false)"),
    Fonte("GLORYS12 MY", "cmems_mod_glo_phy_my_0.083deg_P1D-m", "fisica_glorys", "vários", "vários", "REANALISE", "MY/REP",
          "diária", "1/12°", "~1–2 meses (parte interina)", "FUTURO", "forçante do CROCO; não é feature do ML hoje"),
    Fonte("GLORYS12 ANFC", "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", "fisica_glorys", "vários", "vários", "FORECAST",
          "ANFC", "diária", "1/12°", "análise até hoje + previsão ~10 dias", "NAO_USADO",
          "só ~10 dias à frente; não cobre horizontes de 1–12 meses"),
    Fonte("Vento L4 NRT", "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H", "vento", "m s-1", "m s-1", "ANALISE", "NRT",
          "horária", "0,125°", "~1 dia", "NAO_USADO", "disponível desde 2024-06; sem histórico MY no mesmo produto (DÚVIDA 3)"),
    Fonte("ERA5", "reanalysis-era5-single-levels (C3S)", "atmosfera", "vários", "vários", "REANALISE", "ERA5/ERA5T", "horária",
          "0,25°", "~5 dias (ERA5T)", "FUTURO", "forçante do CROCO"),
    Fonte("Open-Meteo archive (precipitação)", "archive-api.open-meteo.com", "precipitacao", "mm", "mm", "REANALISE", "—",
          "diária", "~0,1–0,25°", "~5 dias", "NAO_USADO", "removida no commit 83d5f2b; baseada em reanálises"),
    Fonte("HABdata (eventos de floração)", "habdata_*.csv da equipe", "eventos", "—", "—", "OBSERVACAO", "—", "evento",
          "ponto", "irregular", "NAO_USADO", "só validação externa; não é alvo de treino"),
    Fonte("Climatologias do ML", "derivadas do painel (anos anteriores)", "derivada", "—", "—", "CLIMATOLOGIA", "—", "mensal",
          "4 km", "nenhuma (calculada do histórico)", "TREINO+INFERENCIA", "expansivas, só anos anteriores"),
    Fonte("CROCO D2 (Navier–Stokes)", "contrato 13.0 (cfd_navier_stokes)", "fisica_croco", "vários", "vários", "REANALISE",
          "hindcast", "diária→mensal", "1,5 km → pontos 4 km", "depende das forçantes (GLORYS/ERA5)", "FUTURO",
          "modelo forçado por reanálises: hindcast, não observação"),
]


def catalogo_df():
    return pd.DataFrame([asdict(f) for f in CATALOGO])


# ------------------------------------------------------------------ disponibilidade no tempo T
# Features do baseline operacional e de onde vêm (variável do painel, meses usados relativos a t0 / τ).
ORIGEM_FEATURES = {
    "h": ("fixo", None), "mes_emissao": ("fixo", None), "mes_alvo": ("fixo", None), "latitude": ("fixo", None),
    "longitude": ("fixo", None), "dist_hotspot": ("fixo", None),
    "chl_media_t0": ("clorofila", (0, 0)), "chl_max_t0": ("clorofila", (0, 0)), "sst_t0": ("temperatura", (0, 0)),
    "sss_t0": ("salinidade", (0, 0)), "chl_media_t0m1": ("clorofila", (1, 1)), "chl_media_t0m2": ("clorofila", (2, 2)),
    "sst_t0m1": ("temperatura", (1, 1)), "sst_t0m2": ("temperatura", (2, 2)), "sss_t0m1": ("salinidade", (1, 1)),
    "chl_media_3m": ("clorofila", (0, 2)), "sst_3m": ("temperatura", (0, 2)), "chl_media_36m": ("clorofila", (0, 35)),
    "sss_36m": ("salinidade", (0, 35)),
    "chl_media_anom_t0": ("clorofila", "t0+clim"), "chl_max_anom_t0": ("clorofila", "t0+clim"),
    "sst_anom_t0": ("temperatura", "t0+clim"), "sss_anom_t0": ("salinidade", "t0+clim"),
    "clim_chl_media_alvo": ("clorofila", "clim_alvo"), "clim_chl_max_alvo": ("clorofila", "clim_alvo"),
    "freq_floracao_alvo": ("clorofila", "clim_alvo"), "n_anos_alvo": ("clorofila", "clim_alvo"),
    "clim_sst_alvo": ("temperatura", "clim_alvo"), "clim_sss_alvo": ("salinidade", "clim_alvo"),
    "chl_max_alvo_ano_anterior": ("clorofila", "alvo-12"), "floracao_alvo_ano_anterior": ("clorofila", "alvo-12"),
    "chl_media_alvo_ano_anterior": ("clorofila", "alvo-12"), "sst_alvo_ano_anterior": ("temperatura", "alvo-12"),
    "sss_alvo_ano_anterior": ("salinidade", "alvo-12"),
}


def meses_usados(feature, t0, h):
    """Índices de mês (ano*12+mes-1) que a feature lê na emissão t0 e horizonte h. Todos devem ser <= t0."""
    var, regra = ORIGEM_FEATURES[feature]
    if var == "fixo":
        return []
    tau = t0 + h
    if isinstance(regra, tuple):
        return list(range(t0 - regra[1], t0 - regra[0] + 1))
    if regra == "t0+clim":
        return [t0] + list(range(t0 - 12, -1, -12))
    if regra == "clim_alvo":
        return list(range(tau - 12, -1, -12))
    if regra == "alvo-12":
        return [tau - 12]
    raise ValueError(regra)


def disponivel_em_t0(feature, t0, h):
    return all(m <= t0 for m in meses_usados(feature, t0, h))


# ------------------------------------------------------------------ validação
FAIXA = {  # faixa física aceitável (fora = FALHA) e faixa típica no litoral de SP (fora = AVISO)
    "clorofila_media": ((0.0, 200.0), (0.01, 60.0)), "clorofila_maxima": ((0.0, 500.0), (0.01, 150.0)),
    "temperatura_media": ((-2.0, 40.0), (14.0, 32.0)), "salinidade_media": ((0.0, 42.0), (25.0, 38.0)),
}


def _status(lista):
    return "FALHA" if any(s == "FALHA" for s in lista) else "AVISO" if any(s == "AVISO" for s in lista) else "OK"


def valida_tempos_diarios(tempos, ano, mes, cobertura_min=0.8):
    """Timestamps diários de um arquivo mensal: UTC, únicos, crescentes, dentro do mês e cobertura >= 80 %."""
    t = pd.to_datetime(pd.Series(tempos))
    r = {"n_dias": int(len(t))}
    chk = []
    if t.dt.tz is not None:
        if str(t.dt.tz) not in ("UTC", "utc"):
            chk.append(("fuso", "FALHA", f"fuso {t.dt.tz}; esperado UTC"))
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        chk.append(("fuso", "OK", "sem fuso: tratado como UTC (convenção CF/Copernicus)"))
    dup = int(t.duplicated().sum())
    chk.append(("duplicatas_tempo", "FALHA" if dup else "OK", f"{dup} tempos repetidos"))
    chk.append(("monotonico", "OK" if t.is_monotonic_increasing else "FALHA", ""))
    fora = int(((t.dt.year != ano) | (t.dt.month != mes)).sum())
    chk.append(("dentro_do_mes", "FALHA" if fora else "OK", f"{fora} tempos fora de {ano}-{mes:02d}"))
    dias_mes = pd.Period(f"{ano}-{mes:02d}").days_in_month
    cob = t.dt.normalize().nunique() / dias_mes
    chk.append(("cobertura_dias", "OK" if cob >= cobertura_min else "FALHA", f"{cob:.0%} dos dias"))
    r.update(cobertura=cob, checagens=chk, status=_status([c[1] for c in chk]))
    return r


def valida_mes(tabela, grade_ml, ano, mes, cfg, variaveis=("clorofila_media", "clorofila_maxima", "temperatura_media", "salinidade_media")):
    """Tabela mensal (painel) de um mês contra a grade do ML: chave, coordenadas, unidades, faixa, ausentes, regiões."""
    t = tabela[(tabela.ano == ano) & (tabela.mes == mes)]
    chk, por_var = [], {}
    if t.empty:
        return {"status": "FALHA", "checagens": [("mes_presente", "FALHA", f"{ano}-{mes:02d} ausente")], "por_variavel": {}}
    dup = int(t.duplicated(["latitude", "longitude", "ano", "mes"]).sum())
    chk.append(("duplicatas_chave", "FALHA" if dup else "OK", f"{dup}"))
    a = cfg["area"]
    fora = int(((t.latitude < a["lat_min"] - 1e-6) | (t.latitude > a["lat_max"] + 1e-6) |
                (t.longitude < a["lon_min"] - 1e-6) | (t.longitude > a["lon_max"] + 1e-6)).sum())
    chk.append(("coordenadas_na_caixa", "FALHA" if fora else "OK", f"{fora} fora da caixa do ML"))
    chave = lambda d: set(zip(d.latitude.round(5), d.longitude.round(5)))  # noqa: E731
    pts_mes, pts_grade = chave(t), chave(grade_ml)
    fora_grade = len(pts_mes - pts_grade)
    faltam = len(pts_grade - pts_mes)
    chk.append(("alinhado_grade_ml", "FALHA" if fora_grade else "OK", f"{fora_grade} pontos fora da grade CHL do ML"))
    chk.append(("pontos_presentes", "AVISO" if faltam else "OK", f"{faltam} de {len(pts_grade)} pontos sem linha"))
    for v in variaveis:
        if v not in t:
            continue
        x = t[v].to_numpy(float)
        frac_nan = float(np.isnan(x).mean()) + faltam / max(len(pts_grade), 1)
        (lo, hi), (tlo, thi) = FAIXA[v]
        val = x[~np.isnan(x)]
        s = []
        if v == "temperatura_media" and len(val) and np.median(val) > 200:
            s.append(("unidade", "FALHA", "temperatura em K (esperado °C)"))
        else:
            s.append(("unidade", "OK", ""))
        n_fora = int(((val < lo) | (val > hi)).sum())
        s.append(("faixa_fisica", "FALHA" if n_fora else "OK", f"{n_fora} valores fora de [{lo}, {hi}]"))
        n_atip = int(((val < tlo) | (val > thi)).sum())
        s.append(("faixa_tipica", "AVISO" if n_atip > 0.01 * max(len(val), 1) else "OK", f"{n_atip} fora de [{tlo}, {thi}]"))
        s.append(("ausentes", "FALHA" if frac_nan > 0.5 else "AVISO" if frac_nan > 0.1 else "OK", f"{frac_nan:.1%}"))
        por_var[v] = {"status": _status([c[1] for c in s]), "checagens": s, "frac_ausente": frac_nan}
    reg = regiao(t.longitude, cfg["regioes"])
    cob_reg = {}
    for nome in sorted(set(r for r in regiao(grade_ml.longitude, cfg["regioes"]))):
        total = int((regiao(grade_ml.longitude, cfg["regioes"]) == nome).sum())
        com = int(((reg == nome) & t.get("clorofila_media", pd.Series(np.nan, index=t.index)).notna().to_numpy()).sum())
        cob_reg[nome] = {"pontos": total, "com_chl": com, "status": "OK" if com >= 0.5 * total else "INSUFICIENTE"}
    insuf = [k for k, v in cob_reg.items() if v["status"] != "OK"]
    chk.append(("cobertura_regioes", "AVISO" if insuf else "OK", f"regiões INSUFICIENTES: {insuf}" if insuf else ""))
    return {"status": _status([c[1] for c in chk] + [p["status"] for p in por_var.values()]),
            "checagens": chk, "por_variavel": por_var, "regioes": cob_reg}


# ------------------------------------------------------------------ fallback e decisão da emissão
def decide_emissao(painel, grade_ml, cfg, hoje: dt.date, variaveis_modelo=("clorofila", "temperatura"),
                   max_atraso_meses=2, max_recuo=3):
    """Mês de emissão t0 = último mês completo cujas variáveis do modelo passam na validação.

    Fallback (nunca inventa valores):
      - API indisponível / mês ausente / variável com > 50 % ausente → o mês não serve; recua até `max_recuo` meses;
      - NaN em alguns pontos (<= 50 %) → mantém NaN (o XGBoost trata ausentes) e marca estado_t0_completo = False;
      - região com < 50 % dos pontos com dado → previsão da região marcada INSUFICIENTE;
      - atraso: t0 mais de `max_atraso_meses` antes do último mês completo → status DESATUALIZADO (emite, com flag).
    """
    col = {"clorofila": "clorofila_media", "temperatura": "temperatura_media", "salinidade": "salinidade_media"}
    ultimo_completo = (hoje.year * 12 + hoje.month - 1) - 1
    tentativas = []
    for k in range(0, max_recuo + 1):
        t0 = ultimo_completo - k
        ano, mes = divmod(t0, 12)
        v = valida_mes(painel, grade_ml, ano, mes + 1, cfg, [col[x] for x in variaveis_modelo])
        tentativas.append({"mes": f"{ano}-{mes + 1:02d}", "status_validacao": v["status"]})
        if v["status"] != "FALHA":
            atraso = ultimo_completo - t0
            return {"t0": t0, "issue_month": f"{ano}-{mes + 1:02d}", "atraso_meses": atraso,
                    "status": "OK" if atraso == 0 else ("ATRASADO" if atraso <= max_atraso_meses else "DESATUALIZADO"),
                    "validacao_t0": v, "tentativas": tentativas}
    return {"t0": None, "status": "SEM_EMISSAO", "tentativas": tentativas,
            "motivo": f"nenhum dos últimos {max_recuo + 1} meses passou na validação; previsão não emitida"}


# ------------------------------------------------------------------ proveniência por feature
def proveniencia_features(painel, features, t0, h_max=12):
    """Para cada feature: variável, meses usados (para h = 1 e h = h_max) e as fontes (produto/versão) desses meses."""
    idx = painel.ano.to_numpy() * 12 + painel.mes.to_numpy() - 1
    fontes_mes = {}
    for v in ("clorofila", "temperatura", "salinidade"):
        c = f"fonte_{v}"
        if c in painel:
            g = pd.DataFrame({"i": idx, "f": painel[c].to_numpy()}).drop_duplicates()
            fontes_mes[v] = g.groupby("i")["f"].agg(lambda s: sorted(set(s)))
    out = {}
    for f in features:
        var, _ = ORIGEM_FEATURES.get(f, ("desconhecida", None))
        if var in ("fixo", "desconhecida"):
            out[f] = {"variavel": var, "fontes": ["grade/configuração" if var == "fixo" else "?"], "meses": []}
            continue
        meses = sorted((set(meses_usados(f, t0, 1)) | set(meses_usados(f, t0, h_max))) & set(np.unique(idx).tolist()))
        if not meses:
            out[f] = {"variavel": var, "fontes": [], "meses": [], "todos_ate_t0": True, "nota": "nenhum mês do painel"}
            continue
        fontes = sorted({x for m in meses for x in fontes_mes.get(var, pd.Series(dtype=object)).get(m, [])})
        txt = lambda m: f"{m // 12}-{m % 12 + 1:02d}"  # noqa: E731
        out[f] = {"variavel": var, "meses_de": txt(min(meses)), "meses_ate": txt(max(meses)), "n_meses": len(meses),
                  "todos_ate_t0": all(m <= t0 for m in meses), "fontes": fontes}
    return out
