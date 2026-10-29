"""Índice de CONDIÇÕES PROPÍCIAS para espécies tóxicas (regras da literatura, docs/fichas_especies_condicoes.md).

Não usa dado de floração para prever: compara o mar de cada mês com as condições que a literatura associa a cada
espécie. Os eventos conhecidos (Mafra Jr. et al. 2026) servem só para CONFERIR se o índice acende quando deveria.

Etapas:
  1. séries mensais por município (SP) e por zona de validação (PR, SC): GLORYS (reanálise) emendada com o sistema
     operacional do Copernicus (corrigido pela diferença média no período em comum), vento L4 + NRT, clorofila;
  2. índice = fração ponderada das condições satisfeitas (0 a 1) em cada mês de 2000 até agora;
  3. validação: o índice acendeu nos meses das florações conhecidas? quantas vezes acende no total?
  4. previsão 1–3 meses: salinidade e temperatura por persistência da anomalia (regressão no próprio histórico);
     camada de mistura, vento e clorofila pela frequência climatológica do mês (não são previsíveis por persistência).
     Na previsão o índice vira a fração ESPERADA de condições satisfeitas.

Uso: python indice_condicoes.py   (depois de baixar_condicoes_regiao.py e baixar_condicoes_nrt.py)
Saídas: dados/condicoes/indice_mensal.parquet, saidas/condicoes_propicias.json, docs/indice_condicoes_validacao.md
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

from algamar_ml import config
from algamar_ml.api_dashboard import MUNICIPIOS

BASE = Path(config.carrega()["caminhos"]["painel"]).parent.parent
COND = BASE / "condicoes"
RAIO_KM = 25          # células oceânicas em volta do ponto do município
RAIO_ZONA_KM = 40     # zonas de validação (PR, SC) são maiores
RAIO_MAX_KM = 80      # se não houver célula no raio (vento, grade de 0,25°), a mais próxima até aqui
CLIM = ("2000-01", "2025-12")
HORIZONTES = (1, 2, 3)
CORTES = (0.4, 0.7)   # escolha do projeto (fichas)
TETO_FORA_EPOCA = 0.69  # fora da época da espécie o nível máximo é "parcialmente propício" (escolha do projeto)
SUL = "condições propícias no Sul (PR/SC) neste mês ou no anterior"
CHUVA = "chuva forte local (≥ 50% acima do normal neste mês ou no anterior; estação CEBIMar)"
# estação meteorológica do CEBIMar/USP (São Sebastião): só vale para o Litoral Norte; dados locais, fora do Git
CEBIMAR = BASE / "cebimar" / "cebimar_mensal.parquet"
MUNICIPIOS_CEBIMAR = {"Ubatuba", "Caraguatatuba", "Ilhabela", "São Sebastião"}
TETO_OBRIGATORIA = 0.39  # sem a condição "obrigatoria" da ficha (nunca houve floração sem ela), no máximo "desfavorável"
NIVEIS = ("desfavorável", "parcialmente propício", "propício")

# zonas onde as florações de Dinophysis foram documentadas (Mafra Jr. et al. 2026, Harmful Algae 158:103177)
ZONAS = [("PR (Paranaguá)", "validação", -25.62, -48.25), ("SC norte (Penha/Itajaí)", "validação", -26.85, -48.55),
         ("SC centro (Florianópolis)", "validação", -27.55, -48.45)]
EVENTOS = {  # meses de floração documentados; "secundario" = menos intensos ou texto ambíguo no artigo
    "principal": {"2008-04", "2008-05", "2008-06", "2016-04", "2016-05", "2016-06", "2016-07", "2024-06", "2024-07"},
    "secundario": {"2011-05", "2012-05", "2013-05", "2014-05", "2021-07", "2021-08", "2021-09", "2021-10",
                   "2022-04", "2022-05", "2022-06"},
}

# condição: (nome, peso pela confiança da ficha [alta 1, média 0,6, baixa 0,3], função(linha) -> bool ou NaN)
def _cond(f):
    def g(r):
        try:
            v = f(r)
        except (TypeError, KeyError):
            return np.nan
        return np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else float(bool(v))
    return g

ESPECIES = {
    "Dinophysis": {
        "toxina": "diarreica (DSP)",
        "condicoes": [
            ("época (mar–set)", 1.0, _cond(lambda r: 3 <= r["m"] <= 9)),
            ("salinidade baixa (anomalia ≤ −1 ou < 33)", 1.0,
             _cond(lambda r: None if np.isnan(r["sal_a"]) else (r["sal_a"] <= -1 or r["sal"] < 33))),
            ("água mais fria que o normal (≤ −1 °C)", 1.0, _cond(lambda r: None if np.isnan(r["sst_a"]) else r["sst_a"] <= -1)),
            ("coluna estratificada (MLD no percentil ≤ 40)", 1.0, _cond(lambda r: None if np.isnan(r["mld_p"]) else r["mld_p"] <= 40)),
            ("vento de sul (componente norte > 0)", 0.6, _cond(lambda r: None if np.isnan(r["v"]) else r["v"] > 0)),
            # "quedas bruscas de salinidade" e "cold spells" [1]: queda contra o mês anterior (limiares: escolha do projeto)
            ("mudança brusca (esfriou ≥ 1 °C ou salinidade caiu ≥ 0,5 desde o mês anterior)", 1.0,
             _cond(lambda r: None if np.isnan(r["dsst"]) and np.isnan(r["dsal"]) else
                   (r["dsst"] <= -1 if not np.isnan(r["dsst"]) else False) or (r["dsal"] <= -0.5 if not np.isnan(r["dsal"]) else False))),
            # plano de contingência de SP (Resolução Conjunta SES/SIMA/SAA-1, 23/04/2021): Alerta 1 começa com floração em SC/PR;
            # a de 2016 "trazida pelas correntes" chegou a Santos, São Sebastião e Peruíbe [1]. Só nos municípios de SP.
            (SUL, 0.6, _cond(lambda r: None if "sul" not in r or np.isnan(r["sul"]) else r["sul"] > 0)),
            # aporte de água doce e nutrientes de rios: pluma de 2016/2024 [1], canal de São Sebastião mar/2019 (CEBIMar),
            # Itajaí-Açu em 2006 (Schramm et al.). Chuva local em 2016, abr/2018, mai/2018 e mar/2019. Limiar: escolha do projeto.
            (CHUVA, 0.6, _cond(lambda r: None if "chuvoso" not in r or np.isnan(r["chuvoso"]) else r["chuvoso"] > 0)),
        ],
    },
    # Tavares et al. 2009 (Penha, mar/2000: 26–28 °C, S 31–32, semanas após entrada de água fria); Schramm et al. 2006
    # (Penha, abr–mai/2006: 75.837 células/L; hipótese: menos sal e nutrientes do rio Itajaí-Açu); Band-Schmidt et al. 2019
    # (revisão: janela de 18–25 °C, ressurgência e descarga de rios; Uruguai 21,8–24 °C e S 18,4–32; Paranaguá fev/2003 a 29 °C).
    "Gymnodinium catenatum": {
        "toxina": "paralisante (PSP)",
        "condicoes": [
            ("época (fev–jun)", 1.0, _cond(lambda r: 2 <= r["m"] <= 6)),
            ("água de 17 a 29 °C", 1.0, _cond(lambda r: None if np.isnan(r["sst"]) else 17 <= r["sst"] <= 29)),
            ("influência de água doce (salinidade ≤ 34 ou 0,5 abaixo do normal)", 0.6,
             _cond(lambda r: None if np.isnan(r["sal"]) else (r["sal"] <= 34 or r["sal_a"] <= -0.5))),
            ("entrada de água fria (esfriou ≥ 1 °C neste mês ou no anterior)", 0.6,
             _cond(lambda r: None if np.isnan(r["dsst"]) and np.isnan(r["dsst_1"]) else
                   bool((r["dsst"] <= -1 if not np.isnan(r["dsst"]) else False) or (r["dsst_1"] <= -1 if not np.isnan(r["dsst_1"]) else False)))),
        ],
    },
    # Band-Schmidt et al. 2019 (Uruguai: florações no início da primavera a 11–15 °C, S 32,2, frente termohalina; Chile 9–14 °C;
    # "Alexandrium spp. florescem em temperaturas mais frias, 5–14 °C"); Condie et al. 2019 (Tasmânia: jun–out, janela de
    # 10–15 °C, depois de chuva forte ou de esfriamento anômalo, com vento fraco; motor comum: estratificação).
    # Nenhuma floração registrada fora da água fria: sem 9–16 °C o índice fica no máximo "desfavorável" ("obrigatoria").
    "Alexandrium catenella": {
        "toxina": "paralisante (PSP)",
        "obrigatoria": "água fria (9 a 16 °C)",
        "condicoes": [
            ("época (jun–out)", 1.0, _cond(lambda r: 6 <= r["m"] <= 10)),
            ("água fria (9 a 16 °C)", 1.0, _cond(lambda r: None if np.isnan(r["sst"]) else 9 <= r["sst"] <= 16)),
            ("coluna estratificada (MLD no percentil ≤ 40)", 1.0, _cond(lambda r: None if np.isnan(r["mld_p"]) else r["mld_p"] <= 40)),
            ("gatilho: água doce (salinidade 0,5 abaixo do normal) ou esfriamento (≤ −1 °C)", 0.6,
             _cond(lambda r: None if np.isnan(r["sal_a"]) and np.isnan(r["sst_a"]) else
                   bool((r["sal_a"] <= -0.5 if not np.isnan(r["sal_a"]) else False) or (r["sst_a"] <= -1 if not np.isnan(r["sst_a"]) else False)))),
        ],
    },
    # Mafra et al. 2006 e 2024 (Paranaguá: picos no fim do verão–início do outono, "quando nutrientes e turbulência são
    # máximos"; 2º pico com frentes frias que misturam a coluna); Braga 1999 (Ubatuba: nutrientes chegam pela intrusão da
    # ACAS, ~14 °C, no fim da primavera e no verão); Méndez et al. 2012 (Uruguai: > 17 °C; florações com salinidade > 30).
    "Pseudo-nitzschia": {
        "toxina": "amnésica (ASP)",
        "condicoes": [
            ("época (out–abr)", 1.0, _cond(lambda r: r["m"] >= 10 or r["m"] <= 4)),
            ("temperatura > 17 °C", 0.6, _cond(lambda r: None if np.isnan(r["sst"]) else r["sst"] > 17)),
            ("salinidade ≥ 30", 0.6, _cond(lambda r: None if np.isnan(r["sal"]) else r["sal"] >= 30)),
            ("chegada de nutrientes (ACAS: ≤ −1 °C; clorofila +25%; ou mistura forte, MLD percentil ≥ 70)", 1.0,
             _cond(lambda r: None if np.isnan(r["sst_a"]) and np.isnan(r["chl_a"]) and np.isnan(r["mld_p"]) else
                   bool((r["sst_a"] <= -1 if not np.isnan(r["sst_a"]) else False)
                        or (r["chl_a"] >= 0.1 if not np.isnan(r["chl_a"]) else False)
                        or (r["mld_p"] >= 70 if not np.isnan(r["mld_p"]) else False)))),
        ],
    },
}


# ---------------------------------------------------------------- 1. séries
def km(lat1, lon1, lat2, lon2):
    return 111.2 * np.hypot(lat2 - lat1, (lon2 - lon1) * np.cos(np.radians(lat1)))


def media_local(da, lat, lon, raio):
    """Média das células válidas no raio (ou a mais próxima até RAIO_MAX_KM) -> série temporal."""
    a = da.sel(latitude=slice(lat - 1, lat + 1), longitude=slice(lon - 1, lon + 1))
    LA, LO = np.meshgrid(a.latitude.values, a.longitude.values, indexing="ij")
    d = km(lat, lon, LA, LO)
    valida = a.notnull().any("time").values
    sel = valida & (d <= raio)
    if not sel.any():
        dd = np.where(valida, d, np.inf)
        if dd.min() > RAIO_MAX_KM:
            return pd.Series(np.nan, index=pd.DatetimeIndex(da.time.values))
        sel = dd == dd.min()
    v = a.values[:, sel]
    return pd.Series(np.nanmean(v, axis=1), index=pd.DatetimeIndex(a.time.values).to_period("M").to_timestamp())


def abre(nome, var):
    ds = xr.open_dataset(COND / f"{nome}.nc")
    da = ds[var]
    return da.isel(depth=0) if "depth" in da.dims else da


def emenda(my, nrt, var):
    """Reanálise + sistema operacional, este corrigido pela diferença média no período em comum."""
    comum = my.dropna().index.intersection(nrt.dropna().index)
    vies = float((my[comum] - nrt[comum]).mean()) if len(comum) >= 6 else 0.0
    r = float(np.corrcoef(my[comum], nrt[comum])[0, 1]) if len(comum) >= 6 else np.nan
    novo = (nrt + vies)[nrt.index > my.dropna().index.max()]
    return pd.concat([my.dropna(), novo.dropna()]).sort_index(), {"var": var, "vies": round(vies, 3), "r": round(r, 3),
                                                                   "meses_comuns": len(comum)}


def mes_corrente(nome, var, lat, lon, raio):
    """Média do mês corrente a partir do diário do sistema operacional (inclui os dias de previsão)."""
    arq = COND / f"{nome}.nc"
    if not arq.exists():
        return pd.Series(dtype=float)
    da = abre(nome, var)
    dias = da.time.to_series().dt.to_period("M").value_counts()
    completos = [p.to_timestamp() for p, n in dias.items() if n >= 10]   # descarta pedaço de mês (1–2 dias)
    s = media_local(da.resample(time="MS").mean(), lat, lon, raio)
    return s[s.index.isin(completos)]


def series(lat, lon, raio):
    out, qual = {}, []
    for var, my_nome, my_var, nrt_nome in [("sst", "glorys_mensal", "thetao", "nrt_thetao"),
                                           ("sal", "glorys_mensal", "so", "nrt_so"),
                                           ("mld", "glorys_mensal", "mlotst", "nrt_mlotst")]:
        nrt_var = {"sst": "thetao", "sal": "so", "mld": "mlotst"}[var]
        my = media_local(abre(my_nome, my_var), lat, lon, raio)
        nrt = media_local(abre(f"{nrt_nome}_mensal", nrt_var), lat, lon, raio)
        cur = mes_corrente(f"{nrt_nome}_diario", nrt_var, lat, lon, raio)
        nrt = pd.concat([nrt, cur[~cur.index.isin(nrt.index)]]).sort_index()
        out[var], q = emenda(my, nrt, var)
        qual.append(q)
    w = media_local(abre("vento_mensal", "northward_wind"), lat, lon, raio)
    if (COND / "nrt_vento_mensal.nc").exists():
        wn = media_local(abre("nrt_vento_mensal", "northward_wind"), lat, lon, raio)
        w = pd.concat([w.dropna(), wn[wn.index > w.dropna().index.max()].dropna()])
    out["v"] = w
    out["chl"] = np.log10(media_local(abre("clorofila_grupos_mensal", "CHL"), lat, lon, raio))
    df = pd.DataFrame(out).sort_index()
    df = df[df.index >= "2000-01-01"]
    return df, qual


def anomalias(df):
    df = df.copy()
    df["m"] = df.index.month
    base = df[(df.index >= CLIM[0]) & (df.index <= pd.Timestamp(CLIM[1]) + pd.offsets.MonthEnd(0))]
    for v in ("sst", "sal", "mld", "chl"):
        clim = base.groupby("m")[v].mean()
        df[f"{v}_c"] = df["m"].map(clim)
        df[f"{v}_a"] = df[v] - df[f"{v}_c"]
    # variação da anomalia contra o mês anterior (só se o mês anterior existe)
    contiguo = df.index.to_series().diff().dt.days.le(31)
    for v in ("sst", "sal"):
        df[f"d{v}"] = df[f"{v}_a"].diff().where(contiguo)
    df["dsst_1"] = df["dsst"].shift(1).where(contiguo)
    # percentil da MLD contra o mesmo mês dos anos de referência
    df["mld_p"] = [np.nan if np.isnan(x) else 100 * (base.loc[base.m == m, "mld"].dropna() < x).mean()
                   for x, m in zip(df.mld, df.m)]
    return df


# ---------------------------------------------------------------- 2. índice
def indice(df, especie):
    conds = ESPECIES[especie]["condicoes"]
    linhas = []
    for t, r in df.iterrows():
        ok = {nome: f(r) for nome, _, f in conds}
        pesos = {nome: p for nome, p, _ in conds}
        tem = [n for n, v in ok.items() if not np.isnan(v)]
        # exige o essencial: época + pelo menos metade do peso com dado
        if sum(pesos[n] for n in tem) < 0.5 * sum(pesos.values()):
            val = np.nan
        else:
            val = sum(pesos[n] * ok[n] for n in tem) / sum(pesos[n] for n in tem)
            if ok[conds[0][0]] == 0:   # a primeira condição de cada ficha é a época
                val = min(val, TETO_FORA_EPOCA)
            obrig = ESPECIES[especie].get("obrigatoria")
            if obrig and ok[obrig] == 0:
                val = min(val, TETO_OBRIGATORIA)
        linhas.append({"mes": t.strftime("%Y-%m"), "indice": val, **{f"ok:{n}": ok[n] for n in ok}})
    return pd.DataFrame(linhas)


def nivel(x):
    if x is None or np.isnan(x):
        return None
    return NIVEIS[0] if x < CORTES[0] else NIVEIS[1] if x < CORTES[1] else NIVEIS[2]


# ---------------------------------------------------------------- 4. previsão
def persistencia(df, v, h):
    """Regressão anomalia(t+h) = b·anomalia(t); devolve b, desvio do resíduo e correlação."""
    a = df[f"{v}_a"]
    x, y = a.values[:-h], a.shift(-h).values[:-h]
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    b = float((x * y).sum() / (x * x).sum())
    return b, float(np.std(y - b * x)), float(np.corrcoef(x, y)[0, 1])


def prob_condicoes(df, t0, h, especie, pers):
    """Probabilidade de cada condição no mês t0+h (prevista, não observada)."""
    alvo = t0 + pd.DateOffset(months=h)
    m = alvo.month
    base = df[(df.index >= CLIM[0]) & (df.index <= pd.Timestamp(CLIM[1]) + pd.offsets.MonthEnd(0))]
    bm = base[base.m == m]
    r0 = df.loc[t0]

    def media_sd(v):
        b, sd, _ = pers[(v, h)]
        a0 = r0[f"{v}_a"]
        clim = float(bm[v].mean())
        return (clim + b * a0 if not np.isnan(a0) else clim), (sd if not np.isnan(a0) else float(bm[v].std()))

    sst_mu, sst_sd = media_sd("sst")
    sal_mu, sal_sd = media_sd("sal")
    sal_c, sst_c = float(bm.sal.mean()), float(bm.sst.mean())
    P = {}
    if especie == "Dinophysis":
        P["época (mar–set)"] = float(3 <= m <= 9)
        P["salinidade baixa (anomalia ≤ −1 ou < 33)"] = float(norm.cdf((max(sal_c - 1, 33) - sal_mu) / sal_sd))
        P["água mais fria que o normal (≤ −1 °C)"] = float(norm.cdf((sst_c - 1 - sst_mu) / sst_sd))
        P["coluna estratificada (MLD no percentil ≤ 40)"] = float((bm.mld_p <= 40).mean())
        P["vento de sul (componente norte > 0)"] = float((bm.v.dropna() > 0).mean())
        brusca = ((bm.dsst <= -1) | (bm.dsal <= -0.5))[bm.dsst.notna() | bm.dsal.notna()]
        P["mudança brusca (esfriou ≥ 1 °C ou salinidade caiu ≥ 0,5 desde o mês anterior)"] = float(brusca.mean())
        if "sul" in df and df["sul"].notna().any():   # só nos municípios de SP
            # 1 mês à frente, "no mês anterior" é o mês atual, que já se conhece
            P[SUL] = float(df.loc[t0, "sul_mes"] > 0) if h == 1 and not np.isnan(df.loc[t0, "sul_mes"]) else \
                float((bm["sul"].dropna() > 0).mean())
        if "chuvoso" in df and df["chuvoso"].notna().any():   # só no Litoral Norte
            ch = df[df.m == m]["chuvoso"].dropna()   # a estação começa em 2013, fora do período de CLIM inteiro
            if h == 1 and not np.isnan(df.loc[t0, "chuva_mes"]):
                P[CHUVA] = float(df.loc[t0, "chuva_mes"] > 0)
            else:
                P[CHUVA] = float(ch.mean()) if len(ch) else np.nan
            if np.isnan(P[CHUVA]):
                del P[CHUVA]
    elif especie == "Gymnodinium catenatum":
        P["época (fev–jun)"] = float(2 <= m <= 6)
        P["água de 17 a 29 °C"] = float(norm.cdf((29 - sst_mu) / sst_sd) - norm.cdf((17 - sst_mu) / sst_sd))
        P["influência de água doce (salinidade ≤ 34 ou 0,5 abaixo do normal)"] = float(
            norm.cdf((max(34, sal_c - 0.5) - sal_mu) / sal_sd))
        frio = ((bm.dsst <= -1) | (bm.dsst_1 <= -1))[bm.dsst.notna() | bm.dsst_1.notna()]
        P["entrada de água fria (esfriou ≥ 1 °C neste mês ou no anterior)"] = float(frio.mean())
    elif especie == "Alexandrium catenella":
        P["época (jun–out)"] = float(6 <= m <= 10)
        P["água fria (9 a 16 °C)"] = float(norm.cdf((16 - sst_mu) / sst_sd) - norm.cdf((9 - sst_mu) / sst_sd))
        P["coluna estratificada (MLD no percentil ≤ 40)"] = float((bm.mld_p <= 40).mean())
        P["gatilho: água doce (salinidade 0,5 abaixo do normal) ou esfriamento (≤ −1 °C)"] = float(
            1 - (1 - norm.cdf((sal_c - 0.5 - sal_mu) / sal_sd)) * (1 - norm.cdf((sst_c - 1 - sst_mu) / sst_sd)))
    else:
        P["época (out–abr)"] = float(m >= 10 or m <= 4)
        P["temperatura > 17 °C"] = float(1 - norm.cdf((17 - sst_mu) / sst_sd))
        P["salinidade ≥ 30"] = float(1 - norm.cdf((30 - sal_mu) / sal_sd))
        nutr = (bm.sst_a <= -1) | (bm.chl_a >= 0.1) | (bm.mld_p >= 70)
        P["chegada de nutrientes (ACAS: ≤ −1 °C; clorofila +25%; ou mistura forte, MLD percentil ≥ 70)"] = float(nutr.mean())
    pesos = {n: p for n, p, _ in ESPECIES[especie]["condicoes"]}
    val = sum(pesos[n] * P[n] for n in P) / sum(pesos[n] for n in P)
    if P[ESPECIES[especie]["condicoes"][0][0]] == 0:
        val = min(val, TETO_FORA_EPOCA)
    obrig = ESPECIES[especie].get("obrigatoria")
    if obrig and val > TETO_OBRIGATORIA:   # condição obrigatória improvável puxa o índice para "desfavorável"
        val = TETO_OBRIGATORIA + (val - TETO_OBRIGATORIA) * P[obrig]
    return alvo, val, P


# ---------------------------------------------------------------- execução
def roda():
    # zonas do Sul primeiro: o índice de Dinophysis delas entra como condição nos municípios de SP
    locais = [(n, g, la, lo, RAIO_ZONA_KM) for n, g, la, lo in ZONAS] + [(n, mr, la, lo, RAIO_KM) for n, mr, la, lo in MUNICIPIOS]
    tabelas, dfs, qualidade = [], {}, {}
    sul = None
    for nome, grupo, lat, lon, raio in locais:
        df, q = series(lat, lon, raio)
        df = anomalias(df)
        if grupo == "validação":
            df["sul"] = np.nan
        else:
            if sul is None:   # alguma zona do Sul em "propício" no mês
                z = pd.concat([t for t in tabelas if t.especie.iat[0] == "Dinophysis"])
                sul = (z.assign(p=z.indice >= CORTES[1]).groupby("mes").p.max().astype(float))
                sul.index = pd.to_datetime(sul.index + "-01")
            df["sul_mes"] = sul.reindex(df.index)
            df["sul"] = df["sul_mes"].combine(df["sul_mes"].shift(1), np.fmax)   # neste mês ou no anterior
            if nome in MUNICIPIOS_CEBIMAR and CEBIMAR.exists():
                c = pd.read_parquet(CEBIMAR).asfreq("MS").chuva_anom_pct.reindex(df.index)
                df["chuva_mes"] = (c >= 50).astype(float).where(c.notna())
                df["chuvoso"] = df["chuva_mes"].combine(df["chuva_mes"].shift(1), np.fmax)
        dfs[nome], qualidade[nome] = df, q
        for esp in ESPECIES:
            ix = indice(df, esp)
            ix.insert(0, "especie", esp)
            ix.insert(0, "grupo", grupo)
            ix.insert(0, "local", nome)
            tabelas.append(ix)
    tab = pd.concat(tabelas, ignore_index=True)
    tab["nivel"] = tab.indice.map(nivel)
    tab.to_parquet(COND / "indice_mensal.parquet", index=False)
    return tab, dfs, qualidade


def valida(tab, dfs):
    """Validação com os eventos de Dinophysis nas zonas do Sul + teste da previsão feita antes do evento."""
    linhas = []
    z = tab[(tab.grupo == "validação") & (tab.especie == "Dinophysis")].dropna(subset=["indice"])
    for local, g in z.groupby("local"):
        prin = g[g.mes.isin(EVENTOS["principal"])]
        sec = g[g.mes.isin(EVENTOS["secundario"])]
        epoca = g[g.mes.str[5:7].astype(int).between(3, 9)]
        linhas.append({
            "zona": local, "meses": len(g),
            "% meses propício (todos)": round(100 * (g.indice >= CORTES[1]).mean(), 1),
            "% meses propício (só mar–set)": round(100 * (epoca.indice >= CORTES[1]).mean(), 1),
            "eventos principais propício": f"{int((prin.indice >= CORTES[1]).sum())}/{len(prin)}",
            "eventos principais ≥ parcial": f"{int((prin.indice >= CORTES[0]).sum())}/{len(prin)}",
            "eventos secundários propício": f"{int((sec.indice >= CORTES[1]).sum())}/{len(sec)}",
            "índice médio eventos": round(prin.indice.mean(), 2),
            "índice médio mar–set sem evento": round(epoca[~epoca.mes.isin(EVENTOS['principal'] | EVENTOS['secundario'])].indice.mean(), 2),
        })
    resumo = pd.DataFrame(linhas)
    # calibração: com que frequência cada condição estava presente nos eventos x nos outros meses mar–set
    cols = [c for c in z.columns if c.startswith("ok:")]
    em_epoca = z[z.mes.str[5:7].astype(int).between(3, 9)]
    sem = em_epoca[~em_epoca.mes.isin(EVENTOS["principal"] | EVENTOS["secundario"])]
    por_cond = pd.DataFrame({"% nos eventos": z[z.mes.isin(EVENTOS["principal"])][cols].mean() * 100,
                             "% em mar–set sem evento": sem[cols].mean() * 100}).round(0)
    por_cond.index = [c[3:] for c in cols]
    detalhe = z[z.mes.isin(EVENTOS["principal"])].pivot(index="mes", columns="local", values="indice").round(2)
    # previsão feita h meses antes de cada evento principal (o que o dashboard teria mostrado)
    prev = []
    for local in z.local.unique():
        df = dfs[local]
        pers = {(v, h): persistencia(df, v, h) for v in ("sst", "sal") for h in HORIZONTES}
        for mes in sorted(EVENTOS["principal"]):
            t = pd.Timestamp(mes + "-01")
            for h in HORIZONTES:
                t0 = t - pd.DateOffset(months=h)
                if t0 in df.index:
                    _, p, _ = prob_condicoes(df, t0, h, "Dinophysis", pers)
                    prev.append({"zona": local, "mes": mes, "h": h, "previsto": p})
    prev = pd.DataFrame(prev)
    # e a mesma previsão em todos os meses mar–set, para comparar (o índice previsto sobe antes dos eventos?)
    base_prev = []
    for local in z.local.unique():
        df = dfs[local]
        pers = {(v, h): persistencia(df, v, h) for v in ("sst", "sal") for h in HORIZONTES}
        for t in df.index[24:]:
            if 3 <= t.month <= 9 and t.strftime("%Y-%m") not in EVENTOS["principal"] | EVENTOS["secundario"]:
                for h in HORIZONTES:
                    t0 = t - pd.DateOffset(months=h)
                    if t0 in df.index:
                        base_prev.append({"zona": local, "h": h, "previsto": prob_condicoes(df, t0, h, "Dinophysis", pers)[1]})
    base_prev = pd.DataFrame(base_prev)
    comp = (prev.groupby("h").previsto.mean().rename("previsto nos eventos").to_frame()
            .join(base_prev.groupby("h").previsto.mean().rename("previsto em mar–set sem evento")).round(3))
    return resumo, detalhe, comp, por_cond


def previsoes(tab, dfs):
    """Para o dashboard: por município e espécie, o mês mais recente com dados + previsão 1–3 meses."""
    out = []
    for nome, macro, lat, lon in MUNICIPIOS:
        df = dfs[nome]
        pers = {(v, h): persistencia(df, v, h) for v in ("sst", "sal") for h in HORIZONTES}
        # "agora": último mês com temperatura e salinidade (o vento pode atrasar)
        t0 = df.dropna(subset=["sst", "sal", "mld"]).index.max()
        r0 = df.loc[t0]
        for esp in ESPECIES:
            linha = tab[(tab.local == nome) & (tab.especie == esp) & (tab.mes == t0.strftime("%Y-%m"))].iloc[0]
            agora = {"mes": t0.strftime("%Y-%m"), "tipo": "observado", "indice": round(float(linha.indice), 3),
                     "nivel": linha.nivel,
                     "condicoes": {n: (None if np.isnan(linha[f"ok:{n}"]) else bool(linha[f"ok:{n}"]))
                                   for n, _, _ in ESPECIES[esp]["condicoes"]}}
            futuros = []
            for h in HORIZONTES:
                alvo, p, P = prob_condicoes(df, t0, h, esp, pers)
                futuros.append({"mes": alvo.strftime("%Y-%m"), "tipo": "previsto", "horizonte_meses": h,
                                "indice": round(p, 3), "nivel": nivel(p),
                                "prob_condicoes": {k: round(v, 2) for k, v in P.items()},
                                # validação: 1 mês antes a previsão não separou os meses de floração (mudanças bruscas)
                                "confianca": "moderada: temperatura e salinidade persistem, mas frentes frias e plumas "
                                             "não são previsíveis" if h == 1 else "baixa: tendência da época"})
            out.append({"municipio": nome, "macro_region": macro, "latitude": lat, "longitude": lon, "especie": esp,
                        "toxina": ESPECIES[esp]["toxina"],
                        "ambiente_agora": {"mes": t0.strftime("%Y-%m"), "temperatura_c": round(float(r0.sst), 2),
                                           "anomalia_temperatura_c": round(float(r0.sst_a), 2),
                                           "salinidade_psu": round(float(r0.sal), 2),
                                           "anomalia_salinidade_psu": round(float(r0.sal_a), 2),
                                           "camada_mistura_m": round(float(r0.mld), 1),
                                           "vento_norte_m_s": None if np.isnan(r0.v) else round(float(r0.v), 2)},
                        "agora": agora, "proximos": futuros})
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    tab, dfs, qual = roda()
    print("qualidade da emenda (reanálise x operacional), Santos:", qual["Santos"])
    resumo, detalhe, comp, por_cond = valida(tab, dfs)
    print(resumo.to_string(index=False)); print(detalhe.to_string()); print(comp.to_string()); print(por_cond.to_string())
    # G. catenatum: Penha mar/2000 (Tavares et al. 2009) e abr–mai/2006 (Schramm et al. 2006); Paranaguá fev–mar/2003 (Mafra et al. 2006)
    g = tab[(tab.especie == "Gymnodinium catenatum") & (tab.grupo == "validação")].dropna(subset=["indice"])
    ev_g = {("SC norte (Penha/Itajaí)", "2000-03"), ("SC norte (Penha/Itajaí)", "2006-04"), ("SC norte (Penha/Itajaí)", "2006-05"),
            ("PR (Paranaguá)", "2003-02"), ("PR (Paranaguá)", "2003-03")}
    ev = g[[(lo, me) in ev_g for lo, me in zip(g.local, g.mes)]]
    epoca = g[g.mes.str[5:7].astype(int).between(2, 6)]
    print("G. catenatum, eventos:", ev[["local", "mes", "indice", "nivel"]].round(2).to_dict("records"),
          "| fev–jun: % propício", round(100 * (epoca.indice >= CORTES[1]).mean(), 1), "| índice médio", round(epoca.indice.mean(), 2))
    a = tab[tab.especie == "Alexandrium catenella"].dropna(subset=["indice"])
    print("A. catenella, % dos meses com água de 9–16 °C:",
          a.groupby("local")["ok:água fria (9 a 16 °C)"].mean().mul(100).round(1).to_dict())
    sp = tab[tab.grupo != "validação"].dropna(subset=["indice"])
    freq = sp.groupby("especie").nivel.value_counts(normalize=True).unstack().mul(100).round(1)
    print("SP, % dos meses por nível:\n", freq.to_string())
    prev = previsoes(tab, dfs)
    saida = {"gerado_em": dt.datetime.now(dt.UTC).isoformat(), "metodo": "índice de condições da literatura",
             "aviso": "Condições propícias NÃO significam floração: a espécie precisa estar presente. "
                      "O mês atual é o mais confiável; os próximos meses mostram a tendência da época.",
             "cortes": {"parcialmente_propicio": CORTES[0], "propicio": CORTES[1]},
             "fontes": "docs/fichas_especies_condicoes.md", "municipios": prev}
    destino = Path(config.carrega()["caminhos"].get("saidas", "saidas"))
    destino.mkdir(exist_ok=True)
    (destino / "condicoes_propicias.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in prev:
        if p["municipio"] in ("Santos", "Ubatuba", "Cananéia"):
            print(p["municipio"], p["especie"], p["agora"]["mes"], p["agora"]["indice"], p["agora"]["nivel"], "|",
                  [(f["mes"], f["indice"], f["nivel"]) for f in p["proximos"]])
