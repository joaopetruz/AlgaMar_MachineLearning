"""Etapa 13.5 — entrada única das features físicas do CROCO/Navier–Stokes no ML (sem refazer a física).

Arquivo de entrada: tabela MENSAL do contrato 13.0 (cfd_navier_stokes, etapa 13.1): uma linha por
(latitude, longitude, ano, mes) nos pontos da grade CHL do ML, colunas `<coluna>_mean` (+ `speed_of_mean_current`),
`validade_espacial` e, de preferência, `h_d2_m`.

Grupos (flags em config/ml.toml [features]; cada grupo liga um conjunto de colunas):
  CROCO_TS        T e S de superfície                                   (VALIDADA)
  CROCO_CORE      TS + MLD + N² máx                                     (VALIDADA / DERIVADA_VALIDADA)
  CROCO_VERTICAL  T e S a 20 m, 50 m e no fundo                         (VALIDADA; NaN onde h < profundidade)
  CROCO_CURRENT   u leste, v norte e módulo da corrente média do mês    (NAO_VALIDADA_OBS)
  CROCO_DYNAMICS  SSH, W 10/50 m, vorticidade, ζ/f, divergência, HBL, transportes (NAO_VALIDADA_OBS)
Justificativas de agrupamento (sem reclassificar o contrato 13.0):
  - 5 m e 10 m ficam fora do VERTICAL: redundantes com a superfície (Spearman 0,89–0,99 na amostra 13.0, 13.2A);
  - direção da corrente fica fora: variável circular (use u/v);
  - speed_of_mean_current (módulo da corrente MÉDIA) é o análogo da antiga corrente_velocidade do ML.
Tempo: na previsão (ponto, t0, h) a física entra SEMPRE como estado do mês de emissão t0 (`croco_<col>_t0`):
para o alvo τ = t0 + h isso é uma defasagem de h meses; nunca há valor do mês-alvo nem de meses > t0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTRATO = Path(__file__).resolve().parent.parent / "contrato_croco"
CHAVE = ["latitude", "longitude", "ano", "mes"]
CASAS = 5
VALIDADES = {"D2_VALIDO", "NAO_VALIDADO_COSTEIRO", "NAO_USAR_ESTUARIO_CANAL"}
NAO_USAR = "NAO_USAR_ESTUARIO_CANAL"
PREFIXO = "croco_"

_TS = ["temperature_surface", "salinity_surface"]
GRUPOS = {
    "CROCO_TS": _TS,
    "CROCO_CORE": _TS + ["mld", "n2_max"],
    "CROCO_VERTICAL": ["temperature_20m", "salinity_20m", "temperature_50m", "salinity_50m", "temperature_bottom", "salinity_bottom"],
    "CROCO_CURRENT": ["u_east_surface", "v_north_surface", "speed_of_mean_current"],
    "CROCO_DYNAMICS": ["ssh", "w_10m", "w_50m", "relative_vorticity_surface", "normalized_vorticity_surface",
                       "horizontal_divergence_surface", "hbl", "transport_east", "transport_north"],
}
FLAG_GRUPO = {"use_croco_ts": "CROCO_TS", "use_croco_core": "CROCO_CORE", "use_croco_vertical": "CROCO_VERTICAL",
              "use_croco_current": "CROCO_CURRENT", "use_croco_dynamics": "CROCO_DYNAMICS"}
PROFUNDIDADE = {"temperature_20m": 20, "salinity_20m": 20, "temperature_50m": 50, "salinity_50m": 50, "w_10m": 10, "w_50m": 50}
FAIXA = {"temperature": (-2, 40), "salinity": (0, 42), "mld": (0, 2000), "n2_max": (-0.01, 0.5), "u_east": (-3, 3),
         "v_north": (-3, 3), "speed_of_mean_current": (0, 4), "ssh": (-3, 3), "w_": (-0.05, 0.05),
         "relative_vorticity": (-0.01, 0.01), "normalized_vorticity": (-50, 50), "horizontal_divergence": (-0.01, 0.01),
         "hbl": (0, 2000), "transport": (-5000, 5000)}


def coluna_arquivo(feature):
    return feature if feature == "speed_of_mean_current" else f"{feature}_mean"


def contrato_13():
    """unidade e status de validação por coluna (contrato 13.0)."""
    c = pd.read_csv(CONTRATO / "feature_contract.csv").set_index("coluna")
    out = {k: {"unidade": r.unidade, "status": r.status} for k, r in c.iterrows()}
    out["speed_of_mean_current"] = {"unidade": "m s-1", "status": "NAO_VALIDADA_OBS"}
    return out


def grupos_ativos(cfg_features):
    return [g for f, g in FLAG_GRUPO.items() if cfg_features.get(f, False)]


def features_ativas(cfg_features):
    """Features físicas (nomes do contrato 13.0) ligadas pelas flags, sem repetição e em ordem estável."""
    out = []
    for g in grupos_ativos(cfg_features):
        out += [f for f in GRUPOS[g] if f not in out]
    return out


def nomes_no_modelo(features):
    return [f"{PREFIXO}{f}_t0" for f in features]


def _faixa(feature):
    for k, v in FAIXA.items():
        if feature.startswith(k):
            return v
    raise KeyError(feature)


def carrega(caminho_ou_tabela, features, proveniencia=None, pontos_ml=None):
    """Lê e VALIDA a tabela mensal do CROCO. Devolve (tabela, relatorio). Levanta ValueError se algo crítico falhar.

    Checagens: schema (chave, validade, colunas pedidas), chave única após round(5) (one-to-one), valores de validade,
    unidades (metadado do Parquet, quando houver, contra o contrato 13.0), faixa física, NaN físico (profundidade
    abaixo do fundo deve ser NaN; nada imputado), cobertura temporal e coordenadas na grade do ML."""
    if isinstance(caminho_ou_tabela, pd.DataFrame):
        t, meta_campos, meta_arq = caminho_ou_tabela.copy(), {}, {}
    else:
        caminho = Path(caminho_ou_tabela)
        t = pd.read_parquet(caminho)
        import pyarrow.parquet as pq
        sch = pq.read_schema(caminho)
        meta_campos = {f.name: {k.decode(): v.decode() for k, v in (f.metadata or {}).items()} for f in sch}
        meta_arq = json.loads(sch.metadata[b"algamar"]) if sch.metadata and b"algamar" in sch.metadata else {}
    rel = {"linhas": int(len(t)), "problemas": [], "avisos": []}
    falta = [c for c in CHAVE + ["validade_espacial"] if c not in t.columns]
    falta += [coluna_arquivo(f) for f in features if coluna_arquivo(f) not in t.columns]
    if falta:
        raise ValueError(f"schema: faltam as colunas {falta}")
    t["latitude"], t["longitude"] = t["latitude"].round(CASAS), t["longitude"].round(CASAS)
    t["ano"], t["mes"] = t["ano"].astype(int), t["mes"].astype(int)
    dup = int(t.duplicated(CHAVE).sum())
    if dup:
        raise ValueError(f"join: {dup} chaves (lat, lon, ano, mes) repetidas após round({CASAS})")
    if not t["mes"].between(1, 12).all():
        raise ValueError("mes fora de 1..12")
    estranhos = set(t["validade_espacial"].unique()) - VALIDADES
    if estranhos:
        raise ValueError(f"validade_espacial desconhecida: {estranhos}")
    contrato = contrato_13()
    for f in features:
        col = coluna_arquivo(f)
        un_meta = meta_campos.get(col, {}).get("unidade")
        if un_meta and un_meta != contrato[f]["unidade"]:
            raise ValueError(f"unidade de {col}: {un_meta} ≠ contrato {contrato[f]['unidade']}")
        lo, hi = _faixa(f)
        fora = int((t[col].notna() & ~t[col].between(lo, hi)).sum())
        if fora:
            raise ValueError(f"faixa física: {fora} valores de {col} fora de [{lo}, {hi}]")
        if f in PROFUNDIDADE and "h_d2_m" in t:
            imputado = int((t["h_d2_m"] < PROFUNDIDADE[f] - 1e-6).sum() - t.loc[t["h_d2_m"] < PROFUNDIDADE[f] - 1e-6, col].isna().sum())
            if imputado:
                raise ValueError(f"NaN físico: {imputado} valores de {col} onde h < {PROFUNDIDADE[f]} m (deveriam ser NaN)")
    if pontos_ml is not None:
        pm = set(zip(pontos_ml["latitude"].round(CASAS), pontos_ml["longitude"].round(CASAS)))
        pc = set(zip(t["latitude"], t["longitude"]))
        rel["pontos_ml_sem_croco"] = len(pm - pc)
        rel["pontos_croco_fora_do_ml"] = len(pc - pm)
    meses = (t["ano"] * 12 + t["mes"] - 1)
    todos = np.arange(meses.min(), meses.max() + 1)
    lacunas = sorted(set(todos) - set(meses.unique()))
    rel["cobertura"] = {"de": f"{meses.min() // 12}-{meses.min() % 12 + 1:02d}", "ate": f"{meses.max() // 12}-{meses.max() % 12 + 1:02d}",
                        "meses": int(len(set(meses))), "meses_faltando": [f"{m // 12}-{m % 12 + 1:02d}" for m in lacunas]}
    if lacunas:
        rel["avisos"].append(f"{len(lacunas)} meses sem CROCO (viram NaN, não são preenchidos)")
    if "mes_completo" in t and (~t["mes_completo"].astype(bool)).any():
        rel["avisos"].append("há meses incompletos (mes_completo = False): mantidos com a flag")
    rel["validade"] = t.groupby("validade_espacial")[["latitude", "longitude"]].apply(lambda g: len(g.drop_duplicates())).to_dict()
    rel["nan_por_feature"] = {f: int(t[coluna_arquivo(f)].isna().sum()) for f in features}
    prov_base = proveniencia or meta_arq.get("proveniencia", {})
    rel["proveniencia"] = {nomes_no_modelo([f])[0]: {
        "source": "CROCO", "coluna_contrato": coluna_arquivo(f), "feature_version": meta_arq.get("versao_contrato", "13.0.0"),
        "unit": contrato[f]["unidade"], "status_validation": contrato[f]["status"],
        "timestamp": "média do mês civil UTC (ano, mes); valor do mês de emissão t0",
        "grid_version": {k: prov_base.get(k) for k in ("build_croco", "commit_croco", "sha256_executavel", "grade", "sha256_grade") if k in prov_base},
        "periodo": rel["cobertura"]["de"] + " → " + rel["cobertura"]["ate"]} for f in features}
    return t, rel


def para_cubo(tabela, cubo, features):
    """Matrizes [mês, ponto] `croco_<f>_t0` alinhadas ao cubo do ML; pontos NAO_USAR → NaN em todos os meses;
    meses sem CROCO → NaN (nada é preenchido)."""
    t = tabela.copy()
    nao_usar_pts = set(map(tuple, t.loc[t["validade_espacial"] == NAO_USAR, ["latitude", "longitude"]].to_numpy()))
    pontos = cubo.pontos.assign(latitude=cubo.pontos["latitude"].round(CASAS), longitude=cubo.pontos["longitude"].round(CASAS))
    j = t.merge(pontos, on=["latitude", "longitude"], how="inner", validate="many_to_one")
    ti = (j["ano"] * 12 + j["mes"] - 1).to_numpy() - cubo.t_inicio
    ok = (ti >= 0) & (ti < cubo.T)
    pid = j["point_id"].to_numpy()
    mascara = np.array([(la, lo) in nao_usar_pts for la, lo in zip(pontos["latitude"], pontos["longitude"])])
    saida = {}
    for f in features:
        a = np.full((cubo.T, len(cubo.pontos)), np.nan, dtype=np.float32)
        a[ti[ok], pid[ok]] = j[coluna_arquivo(f)].to_numpy(dtype=np.float32)[ok]
        a[:, mascara] = np.nan
        saida[nomes_no_modelo([f])[0]] = a
    return saida, {"pontos_nao_usar": int(mascara.sum()), "linhas_casadas": int(ok.sum())}


def validade_por_ponto(tabela, cubo):
    """validade_espacial por point_id do cubo (para filtrar/estratificar métricas)."""
    v = tabela.drop_duplicates(["latitude", "longitude"])[["latitude", "longitude", "validade_espacial"]]
    p = cubo.pontos.assign(latitude=cubo.pontos["latitude"].round(CASAS), longitude=cubo.pontos["longitude"].round(CASAS))
    return p.merge(v, on=["latitude", "longitude"], how="left")["validade_espacial"].fillna("SEM_CROCO").to_numpy()
