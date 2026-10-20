"""Dataset de previsão multi-horizonte, sem vazamento.

Cada linha é (ponto, mês de emissão t0, horizonte h), com h = 1..12:
    features  = só informação observada até o fim do mês t0
    alvo      = floração no mês t0 + h

Grupos de features (ligados/desligados pelas flags de config/ml.toml):
- base (sempre): h, mês de emissão, mês-alvo, latitude, longitude;
- api (use_api_features): estado ambiental no mês de emissão — CHL, SST, SSS em t0,
  t0-1, t0-2, médias de 3 e 36 meses e anomalias em relação à climatologia passada;
- historico (use_historical_features): padrão histórico do mês-alvo — climatologia do
  mês-alvo, frequência histórica de floração no mês-alvo e o mesmo mês do ano anterior
  (t0+h-12 <= t0, por isso h <= 12);
- fisica (use_navier_stokes_features): features da feature store, também no mês t0
  (ver algamar_ml/fisica).

Climatologias e frequências usam só anos anteriores (expansivas); não há estatística
calculada sobre o dataset inteiro.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

H_MAX = 12
VARS_PAINEL = ["clorofila_media", "clorofila_maxima", "temperatura_media", "salinidade_media"]


def indice_mes(ano, mes):
    return np.asarray(ano, dtype=int) * 12 + np.asarray(mes, dtype=int) - 1


def texto_mes(indice):
    indice = np.asarray(indice, dtype=int)
    return np.char.add(np.char.add((indice // 12).astype(str), "-"), np.char.zfill((indice % 12 + 1).astype(str), 2))


@dataclass
class Cubo:
    """Painel em forma de matriz [mês, ponto] (NaN onde o mês não existe)."""
    pontos: pd.DataFrame          # point_id, latitude, longitude
    t_inicio: int                 # índice do primeiro mês (ano*12 + mes-1)
    dados: dict                   # nome -> float32 [T, P]

    @property
    def T(self):
        return next(iter(self.dados.values())).shape[0]

    @property
    def t_fim(self):
        return self.t_inicio + self.T - 1


def cubo_do_painel(painel, colunas=VARS_PAINEL):
    pontos = (painel[["latitude", "longitude"]].drop_duplicates()
              .sort_values(["latitude", "longitude"]).reset_index(drop=True))
    pontos.insert(0, "point_id", np.arange(len(pontos)))
    t = indice_mes(painel["ano"], painel["mes"])
    t0 = int(t.min())
    p = painel[["latitude", "longitude"]].merge(pontos, on=["latitude", "longitude"], how="left")["point_id"].to_numpy()
    dados = {}
    for c in colunas:
        a = np.full((int(t.max()) - t0 + 1, len(pontos)), np.nan, dtype=np.float32)
        a[t - t0, p] = painel[c].to_numpy(dtype=np.float32)
        dados[c] = a
    return Cubo(pontos, t0, dados)


def defasa(a, k):
    """b[t] = a[t-k] (NaN no início)."""
    b = np.full_like(a, np.nan)
    if k < a.shape[0]:
        b[k:] = a[: a.shape[0] - k]
    return b


def climatologia_passada(a):
    """c[t] = média de a[t-12], a[t-24], ... (mesmo mês, anos anteriores; NaN ignorado)."""
    c = np.full_like(a, np.nan)
    for r in range(12):
        sub = a[r::12]
        valido = ~np.isnan(sub)
        soma = np.cumsum(np.where(valido, sub, 0.0), axis=0, dtype=np.float64)
        n = np.cumsum(valido, axis=0)
        res = np.full(sub.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            res[1:] = np.where(n[:-1] > 0, soma[:-1] / np.maximum(n[:-1], 1), np.nan)
        c[r::12] = res.astype(np.float32)
    return c


def contagem_passada(a):
    """n[t] = quantos de a[t-12], a[t-24], ... existem (anos anteriores com dado naquele mês)."""
    n = np.zeros(a.shape, dtype=np.float32)
    for r in range(12):
        sub = ~np.isnan(a[r::12])
        c = np.cumsum(sub, axis=0)
        res = np.zeros(sub.shape, dtype=np.float32)
        res[1:] = c[:-1]
        n[r::12] = res
    return n


def media_movel(a, janela, minimo):
    """m[t] = média de a[t-janela+1 .. t] (inclui t), exige `minimo` valores válidos."""
    valido = ~np.isnan(a)
    soma = np.cumsum(np.where(valido, a, 0.0), axis=0, dtype=np.float64)
    n = np.cumsum(valido, axis=0)
    soma_j = soma - defasa_zero(soma, janela)
    n_j = n - defasa_zero(n, janela)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_j >= minimo, soma_j / np.maximum(n_j, 1), np.nan).astype(np.float32)


def defasa_zero(a, k):
    b = np.zeros_like(a)
    if k < a.shape[0]:
        b[k:] = a[: a.shape[0] - k]
    return b


def alvo(cubo, limiar):
    chl_max = cubo.dados["clorofila_maxima"]
    return np.where(np.isnan(chl_max), np.nan, (chl_max > limiar).astype(np.float32)).astype(np.float32)


def features_emissao(cubo):
    """Features do estado em t0 (grupo api): matriz indexada pelo mês de emissão."""
    d = cubo.dados
    chl, chlmax, sst, sss = d["clorofila_media"], d["clorofila_maxima"], d["temperatura_media"], d["salinidade_media"]
    f = {
        "chl_media_t0": chl, "chl_max_t0": chlmax, "sst_t0": sst, "sss_t0": sss,
        "chl_media_t0m1": defasa(chl, 1), "chl_media_t0m2": defasa(chl, 2),
        "sst_t0m1": defasa(sst, 1), "sst_t0m2": defasa(sst, 2), "sss_t0m1": defasa(sss, 1),
        "chl_media_3m": media_movel(chl, 3, 2), "sst_3m": media_movel(sst, 3, 2),
        "chl_media_36m": media_movel(chl, 36, 12),
        "sss_36m": media_movel(sss, 36, 12),
        "chl_media_anom_t0": chl - climatologia_passada(chl),
        "chl_max_anom_t0": chlmax - climatologia_passada(chlmax),
        "sst_anom_t0": sst - climatologia_passada(sst),
        "sss_anom_t0": sss - climatologia_passada(sss),
    }
    return f


def features_alvo(cubo, limiar):
    """Features do padrão histórico do mês-alvo (grupo historico): matriz indexada pelo mês-alvo τ.
    Em τ só usam meses τ-12, τ-24, ... (todos <= t0 porque h <= 12)."""
    d = cubo.dados
    flor = alvo(cubo, limiar)
    return {
        "clim_chl_media_alvo": climatologia_passada(d["clorofila_media"]),
        "clim_chl_max_alvo": climatologia_passada(d["clorofila_maxima"]),
        "freq_floracao_alvo": climatologia_passada(flor),
        "n_anos_alvo": contagem_passada(flor),
        "clim_sst_alvo": climatologia_passada(d["temperatura_media"]),
        "chl_max_alvo_ano_anterior": defasa(d["clorofila_maxima"], 12),
        "floracao_alvo_ano_anterior": defasa(flor, 12),
        # 13.3: completam as features conceituais do E0 (mesmo mês-alvo no ano anterior e climatologia da salinidade)
        "chl_media_alvo_ano_anterior": defasa(d["clorofila_media"], 12),
        "sst_alvo_ano_anterior": defasa(d["temperatura_media"], 12),
        "sss_alvo_ano_anterior": defasa(d["salinidade_media"], 12),
        "clim_sss_alvo": climatologia_passada(d["salinidade_media"]),
    }


def nomes_features(flags, fisicas=()):
    """Features do modelo conforme as flags. `use_baseline_features` (13.5) é sinônimo de `use_historical_features`.
    As físicas (`fisicas`: croco_<f>_t0 ou ns_<f>_t0) entram se algum grupo use_croco_* ou use_navier_stokes_features
    estiver ligado; sem nenhuma flag física o modelo é exatamente o baseline."""
    nomes = ["h", "mes_emissao", "mes_alvo", "latitude", "longitude"]
    if flags.get("use_api_features", True):
        nomes += [n for n in _NOMES_API if flags.get("use_salinity", True) or not n.startswith("sss")]
        if not flags.get("use_features_redundantes", True):      # versão enxuta: tira as que repetem a informação
            nomes = [n for n in nomes if n not in REDUNDANTES]
    if flags.get("use_baseline_features", flags.get("use_historical_features", True)):
        nomes += list(_NOMES_HIST)
    if flags.get("use_navier_stokes_features", False) or any(flags.get(k, False) for k in
                                                              ("use_croco_ts", "use_croco_core", "use_croco_vertical",
                                                               "use_croco_current", "use_croco_dynamics", "use_glorys_ts", "use_glorys_mld")):
        nomes += list(fisicas)
    return nomes


# 13.3: as 12 features conceituais do E0_corrigido (13.2B) reancoradas no mês de emissão t0, mais h:
# mês-alvo, posição, dist_hotspot (H2, km), o mesmo mês-alvo no ano anterior (τ-12 <= t0), a climatologia do mês-alvo
# (anos anteriores) e as médias de 36 meses terminando em t0 (em h = 1 é exatamente a janela t-36..t-1 do E0_corrigido).
E0_FORECAST_FEATURES = ["h", "mes_alvo", "latitude", "longitude", "dist_hotspot",
                        "chl_media_alvo_ano_anterior", "chl_max_alvo_ano_anterior", "sst_alvo_ano_anterior",
                        "sss_alvo_ano_anterior", "clim_chl_media_alvo", "clim_sss_alvo", "chl_media_36m", "sss_36m"]

_NOMES_API = ["chl_media_t0", "chl_max_t0", "sst_t0", "sss_t0", "chl_media_t0m1", "chl_media_t0m2",
              "sst_t0m1", "sst_t0m2", "sss_t0m1", "chl_media_3m", "sst_3m", "chl_media_36m",
              "chl_media_anom_t0", "chl_max_anom_t0", "sst_anom_t0", "sss_anom_t0"]
# Fechamento (2026-10-09): estado em t0 que repete a informação das médias de 3 meses e das anomalias. Cada uma tem
# < 0,5 % do ganho (gain, no TREINO) do baseline-hist-api-v1. Saem da versão enxuta (use_features_redundantes = false).
REDUNDANTES = ["chl_max_t0", "chl_media_t0m1", "chl_media_t0m2", "sst_t0m1", "sst_t0m2", "sss_t0m1", "chl_max_anom_t0"]
_NOMES_HIST = ["clim_chl_media_alvo", "clim_chl_max_alvo", "freq_floracao_alvo", "n_anos_alvo", "clim_sst_alvo",
               "chl_max_alvo_ano_anterior", "floracao_alvo_ano_anterior"]


def constroi_linhas(cubo, limiar, emissoes, horizontes=range(1, H_MAX + 1), flags=None, fisica=None,
                    exige_estado=True, hotspots=None):
    """Linhas (ponto, t0, h). `emissoes`: índices de mês (ano*12+mes-1) de emissão.

    `fisica`: dict nome -> matriz [T, P] alinhada ao cubo (mês de emissão), opcional.
    O alvo é NaN quando o mês t0+h ainda não foi observado (previsão).
    """
    flags = flags or {}
    horizontes = list(horizontes)
    if max(horizontes) > H_MAX or min(horizontes) < 1:
        raise ValueError("horizontes devem estar entre 1 e 12 meses")
    emissoes = np.asarray([e for e in emissoes if cubo.t_inicio <= e <= cubo.t_fim], dtype=int)
    fe, fa, flor = features_emissao(cubo), features_alvo(cubo, limiar), alvo(cubo, limiar)
    fisica = fisica or {}
    P = len(cubo.pontos)
    dist_hs = None
    if hotspots is not None:   # 13.3: dist_hotspot (km) por ponto, hotspots fixos (H2 vem só do treino)
        from algamar_ml.features.sazonal import distancia_hotspot_km
        dist_hs = distancia_hotspot_km(cubo.pontos, hotspots).astype(np.float32)
    blocos = []
    for h in horizontes:
        i0 = emissoes - cubo.t_inicio
        tau = emissoes + h
        dentro = tau <= cubo.t_fim
        it = np.where(dentro, tau - cubo.t_inicio, 0)
        n = len(emissoes)
        col = {
            "point_id": np.tile(cubo.pontos["point_id"].to_numpy(dtype=np.int32), n),
            "t0": np.repeat(emissoes, P).astype(np.int32),
            "tau": np.repeat(tau, P).astype(np.int32),
            "h": np.full(n * P, h, dtype=np.int16),
            "mes_emissao": np.repeat(emissoes % 12 + 1, P).astype(np.int16),
            "mes_alvo": np.repeat(tau % 12 + 1, P).astype(np.int16),
            "latitude": np.tile(cubo.pontos["latitude"].to_numpy(dtype=np.float32), n),
            "longitude": np.tile(cubo.pontos["longitude"].to_numpy(dtype=np.float32), n),
        }
        if dist_hs is not None:
            col["dist_hotspot"] = np.tile(dist_hs, n)
        for nome, a in fe.items():
            col[nome] = a[i0].ravel()
        for nome, a in fisica.items():
            col[nome] = a[i0].ravel()
        # features do mês-alvo dependem só de meses <= t0, então existem mesmo se τ > último mês
        for nome, a in fa.items():
            col[nome] = _alvo_futuro(cubo, a, nome, tau, i0, limiar).ravel()
        y = np.where(dentro[:, None], flor[it], np.nan)
        col["floracao"] = y.ravel().astype(np.float32)
        blocos.append(pd.DataFrame(col))
    linhas = pd.concat(blocos, ignore_index=True)
    if exige_estado:
        linhas = linhas[linhas["chl_media_t0"].notna()].reset_index(drop=True)
    return linhas


def _alvo_futuro(cubo, a, nome, tau, i0, limiar):
    """Valor da feature do mês-alvo τ. Se τ está além do painel, recalcula a partir dos meses
    τ-12, τ-24, ... que já existem (todos <= t0)."""
    it = tau - cubo.t_inicio
    saida = np.full((len(tau), a.shape[1]), np.nan, dtype=np.float32)
    dentro = it < a.shape[0]
    saida[dentro] = a[it[dentro]]
    for k in np.where(~dentro)[0]:
        saida[k] = _feature_alvo_alem(cubo, nome, int(it[k]), limiar)
    return saida


def _feature_alvo_alem(cubo, nome, it, limiar):
    d = cubo.dados
    base = {"clim_chl_media_alvo": d["clorofila_media"], "clim_chl_max_alvo": d["clorofila_maxima"],
            "freq_floracao_alvo": alvo(cubo, limiar), "n_anos_alvo": alvo(cubo, limiar), "clim_sst_alvo": d["temperatura_media"],
            "chl_max_alvo_ano_anterior": d["clorofila_maxima"], "floracao_alvo_ano_anterior": alvo(cubo, limiar),
            "chl_media_alvo_ano_anterior": d["clorofila_media"], "sst_alvo_ano_anterior": d["temperatura_media"],
            "sss_alvo_ano_anterior": d["salinidade_media"], "clim_sss_alvo": d["salinidade_media"]}[nome]
    anteriores = np.arange(it - 12, -1, -12)
    anteriores = anteriores[anteriores < base.shape[0]]
    if nome == "n_anos_alvo":
        return np.sum(~np.isnan(base[anteriores]), axis=0).astype(np.float32)
    if nome.endswith("ano_anterior"):
        return base[it - 12] if 0 <= it - 12 < base.shape[0] else np.full(base.shape[1], np.nan, np.float32)
    with np.errstate(invalid="ignore"):
        return np.nanmean(base[anteriores], axis=0).astype(np.float32) if len(anteriores) else np.full(base.shape[1], np.nan, np.float32)
