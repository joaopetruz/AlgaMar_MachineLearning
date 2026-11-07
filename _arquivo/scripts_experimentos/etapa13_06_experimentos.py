"""Etapa 13.6 — experimento E0–E7: a física (CROCO; controle GLORYS) melhora o ML?

Referência: E0_corrigido (13.2B): features sem vazamento (hotspots H2), XGBoost 300/6/0,1, scale_pos_weight do treino,
split treino <= 2018 / validação 2019-2021 / teste 2022-2024, limiar de máximo F1 na validação.
O E0 prevê o mês T com dados até T-1 (FORECAST_1M, 13.3), então a física entra com DEFASAGEM DE 1 MÊS por data
(valor do mês T-1; `lag_por_data`), nunca do mês T.

Regras de justiça (todos os modelos): mesmas linhas (dataset do E0 sem os pontos NAO_USAR_ESTUARIO_CANAL e restrito aos
meses em que a fonte física existe em T-1), mesmo split, mesmas seeds, mesmos hiperparâmetros, mesma regra de limiar.
NaN físico (ex.: 50 m em água rasa, terra no GLORYS) fica NaN; nenhuma linha é descartada por isso.

Fontes:
  --fonte croco  : tabela mensal do contrato 13.0 (config caminhos.croco_mensal) → E1..E6
  --fonte glorys : GLORYS12 mensal (dados/brutos/glorys_mensal) → E7 e a escada análoga E1g/E2g/E4g (controle)
Seeds: o XGBoost aqui não tem subamostragem, então a seed sozinha não muda nada; mas o resultado depende da ORDEM das
linhas de treino (medido: E0 0,714 × 0,738 só embaralhando). Cada seed embaralha a ordem do treino (a MESMA permutação para
todos os modelos), e a variação entre seeds entra no relatório.
Controle observacional (13.6): E0 + SST/SSS de satélite do mês T-1 (as mesmas que o ML já baixa) — separa o ganho de
"estado recente do mar" do ganho específico da física de modelo.
Significância: diferença pareada de PR-AUC (mesmas linhas, probabilidades médias das seeds), IC 95 % por bootstrap em
blocos espaço-temporais (ano do alvo × região × trimestre), 500 reamostragens.
uso: python etapa13_06_experimentos.py --fonte glorys [--seeds 42,43,44]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score

from algamar_ml import config, croco_ml
from algamar_ml.avaliacao import divide, escolhe_limiar, novo_modelo
from algamar_ml.espacial import regiao
from algamar_ml.features import CHAVE, FEATURES_BASELINE, constroi_features, lag_por_data
from algamar_ml.metricas import ESTACAO

ap = argparse.ArgumentParser()
ap.add_argument("--fonte", choices=["croco", "glorys"], required=True)
ap.add_argument("--seeds", default="42,43,44")
ap.add_argument("--bootstrap", type=int, default=500)
args = ap.parse_args()
t00 = time.time()
cfg = config.carrega()
O = Path("resultados/etapa13_06") / args.fonte
O.mkdir(parents=True, exist_ok=True)
SEEDS = [int(s) for s in args.seeds.split(",")]
CFD = Path("../../cfd_navier_stokes/outputs/etapa13_01_piloto_producao/features_mensais/features_mensais_auditoria.parquet")

# ------------------------------------------------------------------ dados do E0 e metadados dos pontos
mensal = pd.read_csv(cfg["caminhos"]["historico_equipe"])
e0 = constroi_features(mensal)
e0["latitude"], e0["longitude"] = e0["latitude"].round(5), e0["longitude"].round(5)
meta = pd.read_parquet(CFD)[["latitude", "longitude", "validade_espacial", "h_d2_m"]].drop_duplicates(["latitude", "longitude"])
meta["latitude"], meta["longitude"] = meta["latitude"].round(5), meta["longitude"].round(5)
e0 = e0.merge(meta, on=["latitude", "longitude"], how="left")
e0["validade_espacial"] = e0["validade_espacial"].fillna("SEM_FLAG")
e0 = e0[e0["validade_espacial"] != croco_ml.NAO_USAR].reset_index(drop=True)


# ------------------------------------------------------------------ fonte física → tabela mensal nos pontos do ML
def glorys_mensal():
    ds = xr.open_dataset("dados/brutos/glorys_mensal/glorys_mensal_2000_2024.nc").isel(depth=0)
    pts = e0[["latitude", "longitude"]].drop_duplicates()
    la, lo = xr.DataArray(pts.latitude.to_numpy(), dims="p"), xr.DataArray(pts.longitude.to_numpy(), dims="p")
    it = ds.interp(latitude=la, longitude=lo)          # bilinear; vértice em terra → NaN (não preenchido)
    tempo = pd.to_datetime(it.time.values)
    linhas = []
    for k, t in enumerate(tempo):
        u, v = it.uo.values[k], it.vo.values[k]
        linhas.append(pd.DataFrame({"latitude": pts.latitude.to_numpy(), "longitude": pts.longitude.to_numpy(), "ano": t.year,
                                    "mes": t.month, "temperature_surface_mean": it.thetao.values[k],
                                    "salinity_surface_mean": it.so.values[k], "mld_mean": it.mlotst.values[k],
                                    "u_east_surface_mean": u, "v_north_surface_mean": v, "speed_of_mean_current": np.hypot(u, v)}))
    return pd.concat(linhas, ignore_index=True)


if args.fonte == "glorys":
    fis = glorys_mensal()
    GR = {"TS": ["temperature_surface_mean", "salinity_surface_mean"], "MLD": ["mld_mean"],
          "CORRENTE": ["u_east_surface_mean", "v_north_surface_mean", "speed_of_mean_current"]}
    ESCADA = {"E1g_E0+TS": ["TS"], "E2g_E1g+MLD": ["TS", "MLD"], "E7_E0+GLORYS(TS+MLD+corrente)": ["TS", "MLD", "CORRENTE"]}
    ABLACAO = {"E7-TS": ["MLD", "CORRENTE"], "E7-MLD": ["TS", "CORRENTE"], "E7-corrente": ["TS", "MLD"]}
    SO_FISICA = ["TS", "MLD", "CORRENTE"]
else:
    fis, _ = croco_ml.carrega(cfg["caminhos"]["croco_mensal"], [f for g in croco_ml.GRUPOS.values() for f in g])
    c = croco_ml.coluna_arquivo
    GR = {"TS": [c(f) for f in croco_ml.GRUPOS["CROCO_TS"]], "MLD_N2": [c("mld"), c("n2_max")],
          "VERTICAL": [c(f) for f in croco_ml.GRUPOS["CROCO_VERTICAL"]], "CORRENTE": [c(f) for f in croco_ml.GRUPOS["CROCO_CURRENT"]],
          "SSH": [c("ssh")], "W": [c("w_10m"), c("w_50m")],
          "VORT_DIV": [c("relative_vorticity_surface"), c("normalized_vorticity_surface"), c("horizontal_divergence_surface")],
          "HBL": [c("hbl")], "TRANSPORTES": [c("transport_east"), c("transport_north")]}
    DYN = ["SSH", "W", "VORT_DIV", "HBL", "TRANSPORTES"]
    ESCADA = {"E1_E0+TS": ["TS"], "E2_E1+MLD+N2": ["TS", "MLD_N2"], "E3_E2+vertical": ["TS", "MLD_N2", "VERTICAL"],
              "E4_E3+corrente": ["TS", "MLD_N2", "VERTICAL", "CORRENTE"],
              "E5_E4+dinamica": ["TS", "MLD_N2", "VERTICAL", "CORRENTE"] + DYN}
    E5 = ESCADA["E5_E4+dinamica"]
    ABLACAO = {f"E5-{g}": [x for x in E5 if x != g] for g in E5}
    SO_FISICA = E5

# defasagem de 1 mês POR DATA (o E0 prevê T com dados até T-1)
fis["latitude"], fis["longitude"] = fis["latitude"].round(5), fis["longitude"].round(5)
fis = fis.sort_values(CHAVE).reset_index(drop=True)
colunas_fis = sorted({c for g in GR.values() for c in g})
lagged = fis[CHAVE].copy()
for c in colunas_fis:
    lagged[f"{c}__lag1"] = lag_por_data(fis, c, 1)
dados = e0.merge(lagged, on=CHAVE, how="left", validate="one_to_one")
# mesmas linhas para todos: meses em que a fonte existe em T-1 (algum ponto com valor)
idx_fonte = set((fis["ano"] * 12 + fis["mes"] - 1 + 1).unique())     # T tal que T-1 existe na fonte
dados = dados[(dados["ano"] * 12 + dados["mes"] - 1).isin(idx_fonte)].reset_index(drop=True)
# controle observacional: SST, SSS e CHL de satélite (painel do ML) no mês T-1, por data
obs = mensal.copy()
obs["latitude"], obs["longitude"] = obs["latitude"].round(5), obs["longitude"].round(5)
obs = obs.sort_values(CHAVE).reset_index(drop=True)
lag_obs = obs[CHAVE].copy()
for c in ("temperatura_media", "salinidade_media", "clorofila_media"):
    lag_obs[f"obs_{c}__lag1"] = lag_por_data(obs, c, 1)
dados = dados.merge(lag_obs, on=CHAVE, how="left", validate="one_to_one")
nomes = lambda grupos: [f"{c}__lag1" for g in grupos for c in GR[g]]  # noqa: E731

MINIMAS = ["mes", "latitude", "longitude"]
OBS_TS = ["obs_temperatura_media__lag1", "obs_salinidade_media__lag1"]
MODELOS = {"E0_corrigido": FEATURES_BASELINE, "C1_E0+satelite_TS(T-1)": FEATURES_BASELINE + OBS_TS,
           "C2_E0+satelite_TS+CHL(T-1)": FEATURES_BASELINE + OBS_TS + ["obs_clorofila_media__lag1"]}
MODELOS.update({k: FEATURES_BASELINE + nomes(v) for k, v in ESCADA.items()})
MODELOS.update({k: FEATURES_BASELINE + nomes(v) for k, v in ABLACAO.items()})
MODELOS["E6_so_fisica+tempo_minimo" if args.fonte == "croco" else "E6g_so_GLORYS+tempo_minimo"] = MINIMAS + nomes(SO_FISICA)
_completo = list(ESCADA)[-1]
MODELOS[f"C3_{_completo}+satelite_TS+CHL(T-1)"] = MODELOS[_completo] + OBS_TS + ["obs_clorofila_media__lag1"]
# C4: o que a física acrescenta ALÉM do estado observado por satélite (TS de satélite + grupos não-TS da física)
MODELOS["C4_C1+fisica_sem_TS"] = FEATURES_BASELINE + OBS_TS + nomes([g for g in SO_FISICA if g != "TS"])
if args.fonte == "glorys":
    MODELOS["C5_C1+GLORYS_MLD"] = FEATURES_BASELINE + OBS_TS + nomes(["MLD"])

tr, va, te = divide(dados)
y = dados["floracao"].to_numpy(int)
idx_tr = np.where(tr)[0]
print(f"linhas {len(dados):,} | treino {tr.sum():,} validação {va.sum():,} teste {te.sum():,} | modelos {len(MODELOS)} | {time.time()-t00:.0f}s", flush=True)


def met(yy, p, lim):
    pred = (p >= lim).astype(int)
    return {"pr_auc": average_precision_score(yy, p), "roc_auc": roc_auc_score(yy, p), "f1": f1_score(yy, pred, zero_division=0),
            "recall": recall_score(yy, pred, zero_division=0), "precision": precision_score(yy, pred, zero_division=0),
            "brier": brier_score_loss(yy, p), "log_loss": log_loss(yy, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])}


prob, lim, por_seed = {}, {}, []
for nome, feats in MODELOS.items():
    ps, ls = [], []
    for s in SEEDS:
        ordem = np.random.default_rng(s).permutation(idx_tr)      # mesma permutação para todos os modelos
        m = novo_modelo(dados.loc[tr, "floracao"], seed=s)
        m.fit(dados.loc[ordem, feats], y[ordem])
        pv = m.predict_proba(dados.loc[va, feats])[:, 1]
        l = escolhe_limiar(y[va], pv)
        pt = m.predict_proba(dados.loc[te, feats])[:, 1]
        ps.append(pt); ls.append(l)
        por_seed.append({"modelo": nome, "seed": s, "limiar": l, "pr_auc_validacao": average_precision_score(y[va], pv), **met(y[te], pt, l)})
    prob[nome], lim[nome] = np.mean(ps, axis=0), float(np.mean(ls))
    print(f"{nome:<40} PR-AUC teste {np.mean([r['pr_auc'] for r in por_seed if r['modelo'] == nome]):.4f} | {time.time()-t00:.0f}s", flush=True)
ps_df = pd.DataFrame(por_seed)
ps_df.to_csv(O / "metricas_por_seed.csv", index=False, float_format="%.5f")

# ------------------------------------------------------------------ tabela principal + bootstrap pareado em blocos
T = dados[te].reset_index(drop=True)
yt = y[te]
reg = regiao(T["longitude"], cfg["regioes"])
blocos = T["ano"].astype(str).to_numpy() + "|" + reg.astype(str) + "|" + ((T["mes"] - 1) // 3).astype(str).to_numpy()
ub = np.unique(blocos); idxb = {b: np.where(blocos == b)[0] for b in ub}
rng = np.random.default_rng(0)
amostras = [np.concatenate([idxb[b] for b in rng.choice(ub, len(ub), replace=True)]) for _ in range(args.bootstrap)]
ref = prob["E0_corrigido"]
tab = []
for nome, feats in MODELOS.items():
    agg = ps_df[ps_df.modelo == nome].mean(numeric_only=True)
    p = prob[nome]
    d = [average_precision_score(yt[a], p[a]) - average_precision_score(yt[a], ref[a]) for a in amostras]
    dl = [log_loss(yt[a], np.clip(p[a], 1e-15, 1 - 1e-15), labels=[0, 1]) - log_loss(yt[a], np.clip(ref[a], 1e-15, 1 - 1e-15), labels=[0, 1]) for a in amostras]
    tab.append({"modelo": nome, "n_features": len(feats), "features_fisicas": ";".join(f for f in feats if f.endswith("__lag1")),
                **{k: agg[k] for k in ("pr_auc", "roc_auc", "f1", "recall", "precision", "brier", "log_loss")},
                "pr_auc_desvio_seeds": float(ps_df[ps_df.modelo == nome].pr_auc.std()),
                "delta_pr_auc_vs_E0": float(agg["pr_auc"] - ps_df[ps_df.modelo == "E0_corrigido"].pr_auc.mean()),
                "delta_pr_auc_media_seeds_pareado": float(average_precision_score(yt, p) - average_precision_score(yt, ref)),
                "ic95_delta_pr_auc": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "frac_bootstrap_delta_positivo": float(np.mean(np.array(d) > 0)),
                "ic95_delta_log_loss": [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))]})
tabela = pd.DataFrame(tab)
# comparações pareadas extras (mesmas reamostragens): o que a física acrescenta ALÉM do satélite
pares = [("C5_C1+GLORYS_MLD", "C1_E0+satelite_TS(T-1)"), ("C4_C1+fisica_sem_TS", "C1_E0+satelite_TS(T-1)"),
         (list(ESCADA)[0], "C1_E0+satelite_TS(T-1)"), (list(ESCADA)[-1], list(ESCADA)[1] if len(ESCADA) > 1 else list(ESCADA)[0])]
extra = []
for a_, b_ in pares:
    if a_ in prob and b_ in prob and a_ != b_:
        d = [average_precision_score(yt[x], prob[a_][x]) - average_precision_score(yt[x], prob[b_][x]) for x in amostras]
        extra.append({"modelo": a_, "referencia": b_, "delta_pr_auc": float(average_precision_score(yt, prob[a_]) - average_precision_score(yt, prob[b_])),
                      "ic95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "frac_positivo": float(np.mean(np.array(d) > 0))})
pd.DataFrame(extra).to_csv(O / "comparacoes_pareadas_extras.csv", index=False, float_format="%.5f")
print(pd.DataFrame(extra).to_string(index=False), flush=True)
tabela.to_csv(O / "tabela_experimentos.csv", index=False, float_format="%.5f")
print(tabela[["modelo", "pr_auc", "roc_auc", "f1", "brier", "log_loss", "delta_pr_auc_vs_E0", "ic95_delta_pr_auc"]].to_string(index=False), flush=True)

# ------------------------------------------------------------------ estratificações (teste; probabilidades médias das seeds)
faixa_h = pd.cut(T["h_d2_m"], [0, 20, 50, 100, 1e5], labels=["<20 m", "20–50 m", "50–100 m", ">100 m"]).astype(str)
estr = {"ano": T["ano"].astype(str), "mes": T["mes"].astype(str), "estacao": T["mes"].map(ESTACAO), "regiao": pd.Series(reg),
        "faixa_profundidade": faixa_h, "validade_espacial": T["validade_espacial"]}
melhor = tabela[tabela.modelo != "E0_corrigido"].sort_values("pr_auc", ascending=False).modelo.iloc[0]
lin = []
for nome_e, g in estr.items():
    for val in sorted(pd.unique(g)):
        s = (g == val).to_numpy()
        if s.sum() < 200 or len(np.unique(yt[s])) < 2:
            continue
        for nome in ["E0_corrigido", melhor] + [k for k in ESCADA if k != melhor]:
            lin.append({"estrato": nome_e, "valor": val, "modelo": nome, "n": int(s.sum()), "positivos": int(yt[s].sum()),
                        "pr_auc": average_precision_score(yt[s], prob[nome][s]), "roc_auc": roc_auc_score(yt[s], prob[nome][s]),
                        "delta_pr_auc_vs_E0": average_precision_score(yt[s], prob[nome][s]) - average_precision_score(yt[s], ref[s])})
pd.DataFrame(lin).to_csv(O / "estratificacoes.csv", index=False, float_format="%.5f")
json.dump({"fonte": args.fonte, "seeds": SEEDS, "linhas": {"treino": int(tr.sum()), "validacao": int(va.sum()), "teste": int(te.sum())},
           "modelos": MODELOS, "grupos": GR, "melhor_por_pr_auc": melhor, "tempo_s": round(time.time() - t00, 1),
           "defasagem_fisica": "1 mês por data (valor de T-1)", "bootstrap": {"n": args.bootstrap, "blocos": int(len(ub)),
                                                                              "definicao": "ano do alvo × região × trimestre"}},
          open(O / "resultados.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok", round(time.time() - t00), "s")
