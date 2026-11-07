"""Piloto EXPLORATÓRIO CROCO → ML: há sinal nas features físicas já produzidas?

Este é um teste piloto exploratório. A série CROCO disponível é insuficiente para concluir que as features físicas
melhoram a capacidade de previsão futura do modelo.

Entradas (só dados existentes):
- resultados/piloto_croco_ml/croco_mensal_piloto.parquet, gerado por cfd_navier_stokes/scripts/piloto_croco_ml_extrai.py
  com as funções 13.0/13.1. Cobre 2017-11 e 2017-12 (piloto 13.1), 2014-01 parcial (produção interrompida) e 2024-01
  (rodada 12.13 P4, pré-oficial, só como sensibilidade).
- painel do ML (alvo e features oficiais), o modelo oficial salvo (só como referência; não é alterado) e GLORYS mensal.

Desenho, sem vazamento:
- linha = (ponto, t0); a física é a média CROCO do mês t0 (parcial) e as features do ML usam dados ≤ t0;
- alvo FORECAST = floração em t0 + 1 (h = 1). Para diagnóstico também o alvo CONTEMPORÂNEO (t0), que NÃO é previsão;
- modelos P0–P4: XGBoost com os MESMOS hiperparâmetros do oficial (config/ml.toml) e a mesma base climatológica.
  Avaliação deixando um mês de fora: treina em 2 meses e testa no 3º. Nada de CV aleatória;
- controle: as colunas CROCO são embaralhadas entre linhas do mesmo mês (20 vezes).
Saídas: resultados/piloto_croco_ml/
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)
import xgboost as xgb

from algamar_ml import config, croco_ml, glorys_ml, pipeline
from algamar_ml.espacial import regiao
from algamar_ml.features.horizonte import constroi_linhas, indice_mes, texto_mes
from algamar_ml.modelos import treino as tr

t00 = time.time()
O = Path("resultados/piloto_croco_ml"); O.mkdir(parents=True, exist_ok=True)
AVISO = ("Este é um teste piloto exploratório. A série CROCO disponível é insuficiente para concluir que as features "
         "físicas melhoram a capacidade de previsão futura do modelo.")
cfg = config.carrega()
GR = {g: list(f) for g, f in croco_ml.GRUPOS.items()}
TODAS = list(dict.fromkeys(f for g in GR.values() for f in g))
COL = {f: croco_ml.coluna_arquivo(f) for f in TODAS}
STATUS = {f: croco_ml.contrato_13().get(f, {}).get("status", "?") for f in TODAS}

# ------------------------------------------------------------------ 1. auditoria dos dados CROCO existentes
cr = pd.read_parquet(O / "croco_mensal_piloto.parquet")
aud = []
for (fonte, ano, mes), g in cr.groupby(["fonte_croco", "ano", "mes"]):
    usa = g[g.validade_espacial != croco_ml.NAO_USAR]
    for f in TODAS:
        x = usa[COL[f]] if COL[f] in usa else pd.Series(np.nan, index=usa.index)
        aud.append({"fonte_croco": fonte, "ano": ano, "mes": mes, "dias": int(g.dias_cobertos.iloc[0]), "pontos": int(len(usa)),
                    "feature": f, "grupo": next(k for k, v in GR.items() if f in v), "status_validacao": STATUS[f],
                    "existe_no_arquivo": COL[f] in usa, "n_validos": int(x.notna().sum()), "pct_nan": round(100 * float(x.isna().mean()), 2),
                    "min": x.min(), "p10": x.quantile(.1), "media": x.mean(), "p50": x.median(), "p90": x.quantile(.9), "max": x.max(), "dp": x.std()})
aud = pd.DataFrame(aud); aud.to_csv(O / "auditoria_dados_croco.csv", index=False, float_format="%.6g")

# ------------------------------------------------------------------ 2–3. join CROCO × ML (chave lat, lon, ano, mes; round 5)
cubo, painel = pipeline.carrega_cubo(cfg)
_, cartao = tr.carrega(Path(cfg["caminhos"]["modelos"]) / cfg["modelo"]["oficial"])
BASE = list(cartao["features"])
pts = cubo.pontos.assign(la=cubo.pontos.latitude.round(5), lo=cubo.pontos.longitude.round(5))
fis = {f"croco_{f}": np.full((cubo.T, len(pts)), np.nan, np.float32) for f in TODAS}
valid = {}
cr["la"], cr["lo"] = cr.latitude.round(5), cr.longitude.round(5)
m = cr.merge(pts[["point_id", "la", "lo"]], on=["la", "lo"], how="inner")
for _, g in m.groupby(["ano", "mes"]):
    ti = int(indice_mes(int(g.ano.iloc[0]), int(g.mes.iloc[0]))) - cubo.t_inicio
    for f in TODAS:
        if COL[f] in g:
            fis[f"croco_{f}"][ti, g.point_id.to_numpy()] = g[COL[f]].to_numpy(np.float32)
for pid, v in zip(m.point_id, m.validade_espacial):
    valid[pid] = v
# GLORYS (mesmo mês t0) para a comparação qualitativa
gl = glorys_ml.para_cubo(cfg["caminhos"]["glorys_mensal"], cubo, {"use_glorys_ts": True, "use_glorys_mld": True})
fonte_mes = {int(indice_mes(int(a), int(b))): f for (f, a, b) in cr[["fonte_croco", "ano", "mes"]].drop_duplicates().itertuples(index=False)}
T0 = sorted(fonte_mes)
L = constroi_linhas(cubo, cfg["alvo"]["limiar"], T0, horizontes=[1], fisica={**fis, **gl}, exige_estado=False)
L["validade_espacial"] = L.point_id.map(valid)
L["fonte_croco"] = L.t0.map(fonte_mes)
L = L[(L.validade_espacial.notna()) & (L.validade_espacial != croco_ml.NAO_USAR)].reset_index(drop=True)
# alvo contemporâneo (diagnóstico): floração no próprio mês t0
cmax = cubo.dados["clorofila_maxima"]
L["floracao_t0"] = np.where(np.isnan(cmax[L.t0 - cubo.t_inicio, L.point_id]), np.nan, (cmax[L.t0 - cubo.t_inicio, L.point_id] > cfg["alvo"]["limiar"]).astype(float))
L["regiao"] = regiao(L.longitude, cfg["regioes"])
L["mes_t0"] = texto_mes(L.t0.to_numpy()); L["mes_alvo_txt"] = texto_mes(L.tau.to_numpy())
L = L[L.floracao.notna()].reset_index(drop=True)
CF = [f"croco_{f}" for f in TODAS]
L[list(dict.fromkeys(["point_id", "latitude", "longitude", "regiao", "validade_espacial", "fonte_croco", "mes_t0", "mes_alvo_txt", "t0", "tau", "h", "floracao", "floracao_t0"]
  + BASE + CF + list(gl)))].to_parquet(O / "dataset_piloto.parquet", index=False)
cob = {"pontos_ml": int(len(cubo.pontos)), "pontos_croco_casados": int(m.point_id.nunique()),
       "pontos_ml_sem_croco": int(len(cubo.pontos) - m.point_id.nunique()),
       "pontos_nao_usar_excluidos": int(sum(v == croco_ml.NAO_USAR for v in valid.values())),
       "meses_t0": {texto_mes(t): fonte_mes[t] for t in T0},
       "linhas_por_t0": L.groupby("mes_t0").size().to_dict(), "linhas_total": int(len(L)),
       "positivos_por_alvo": L.groupby("mes_alvo_txt").floracao.agg(["sum", "mean"]).round(3).to_dict("index"),
       "dias_croco_por_mes": {f"{a}-{b:02d} {f}": int(v) for (f, a, b), v in cr.groupby(["fonte_croco", "ano", "mes"]).dias_cobertos.first().items()}}
json.dump(cob, open(O / "cobertura_croco.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(cob, ensure_ascii=False, default=str), flush=True)

# ------------------------------------------------------------------ 5. associação feature × floração
oficiais = sorted(L[L.fonte_croco != "REST_JAN2024_PRE_OFICIAL"].t0.unique())
A = L[L.t0.isin(oficiais)]
clim = A.freq_floracao_alvo.fillna(A.freq_floracao_alvo.mean()).to_numpy()
assoc = []
for f in CF + list(gl):
    for alvo in ("floracao", "floracao_t0"):
        d = A[[f, alvo, "t0", "freq_floracao_alvo"]].dropna()
        if len(d) < 50 or d[alvo].nunique() < 2:
            assoc.append({"feature": f, "alvo": alvo, "n": len(d), "nota": "dados insuficientes"}); continue
        x, y = d[f].to_numpy(), d[alvo].to_numpy()
        m1, m0, sd = x[y == 1].mean(), x[y == 0].mean(), x.std()
        rho_m = [spearmanr(g[f], g[alvo]).correlation for _, g in d.groupby("t0") if g[alvo].nunique() > 1 and g[f].nunique() > 1]
        res = y - d.freq_floracao_alvo.fillna(d.freq_floracao_alvo.mean()).to_numpy()
        assoc.append({"feature": f, "grupo": next((k for k, v in GR.items() if f.replace("croco_", "") in v), "GLORYS"),
                      "status_validacao": STATUS.get(f.replace("croco_", ""), "REANALISE"), "alvo": alvo, "n": len(d),
                      "media_floracao": m1, "media_sem_floracao": m0, "diferenca": m1 - m0, "diferenca_padronizada": (m1 - m0) / sd if sd else np.nan,
                      "spearman": spearmanr(x, y).correlation, "spearman_dentro_do_mes": float(np.nanmean(rho_m)) if rho_m else np.nan,
                      "spearman_residuo_climatologia": spearmanr(x, res).correlation,
                      "mutual_info": float(mutual_info_classif(x.reshape(-1, 1), y.astype(int), random_state=0)[0])})
assoc = pd.DataFrame(assoc); assoc.to_csv(O / "associacao_features.csv", index=False, float_format="%.5g")
A[CF].corr(method="spearman").to_csv(O / "correlacao_features_croco.csv", float_format="%.3f")

# ------------------------------------------------------------------ 6–9. modelos piloto (deixa um mês de fora)
def met(y, p, lim):
    c = p >= lim
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p), "f1": f1_score(y, c),
            "precision": precision_score(y, c, zero_division=0), "recall": recall_score(y, c), "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}


def margem(d):
    return tr.margem_base(d, P0_TREINO) if cfg["modelo"].get("base_climatologica") else None


def ajusta(d, feats):
    mod = tr.novo_modelo(cfg["modelo"]); mod.set_params(early_stopping_rounds=None)   # sem conjunto de validação no piloto
    mod.fit(d[feats], d.floracao.astype(int), base_margin=margem(d))
    return mod


def prob(mod, d, feats):
    return mod.predict_proba(d[feats], base_margin=margem(d))[:, 1]


def avalia(D, feats, embaralha=None, seed=0):
    rng = np.random.default_rng(seed)
    D = D.copy()
    if embaralha:                                             # controle: CROCO embaralhado entre linhas do MESMO mês
        for _, ix in D.groupby("t0").groups.items():
            perm = rng.permutation(len(ix))
            D.loc[ix, embaralha] = D.loc[ix, embaralha].to_numpy()[perm]
    oof = np.full(len(D), np.nan)
    for t in sorted(D.t0.unique()):
        te = (D.t0 == t).to_numpy()
        mod = ajusta(D[~te], feats); oof[te] = prob(mod, D[te], feats)
    return oof


GRUPOS_P = {"P0_baseline": [], "P1_+CROCO_CORE": GR["CROCO_CORE"], "P2_+CROCO_CURRENT": GR["CROCO_CURRENT"],
            "P3_+CROCO_DYNAMICS": GR["CROCO_DYNAMICS"], "P4_+CROCO_TODAS": TODAS,
            "Pv_+CROCO_VERTICAL": GR["CROCO_VERTICAL"], "Pt_+CROCO_TS": GR["CROCO_TS"],
            "Pg_+GLORYS_TS_MLD": ["__glorys"]}
metr, comp, emb, imp = [], [], [], []
for cenario, meses in {"oficiais_3_meses": oficiais, "sensibilidade_4_meses_com_pre_oficial": sorted(L.t0.unique())}.items():
    D = L[L.t0.isin(meses)].reset_index(drop=True)
    P0_TREINO = float(D.floracao.mean())
    y = D.floracao.astype(int).to_numpy()
    lim = cartao["limiar_alto"]
    probs = {}
    for nome, gfe in GRUPOS_P.items():
        feats = BASE + (list(gl) if gfe == ["__glorys"] else [f"croco_{f}" for f in gfe])
        insuf = [f for f in feats if D[f].notna().mean() < 0.5]
        if insuf:
            metr.append({"cenario": cenario, "modelo": nome, "nota": f"dados insuficientes: {insuf}"}); continue
        probs[nome] = avalia(D, feats)
        metr.append({"cenario": cenario, "modelo": nome, "n": len(D), "positivos": int(y.sum()), **met(y, probs[nome], lim),
                     **{f"pr_auc_mes_{texto_mes(t)}": average_precision_score(y[D.t0 == t], probs[nome][(D.t0 == t).to_numpy()])
                        for t in meses if y[D.t0 == t].sum() > 0}})
    # comparação pareada com P0 (bootstrap por blocos mês × região)
    bl = D.mes_t0 + "|" + D.regiao; ub = bl.unique(); ix = {b: np.where(bl == b)[0] for b in ub}
    rng = np.random.default_rng(0)
    am = [np.concatenate([ix[b] for b in rng.choice(ub, len(ub))]) for _ in range(500)]
    for nome, pp in probs.items():
        d = [average_precision_score(y[a], pp[a]) - average_precision_score(y[a], probs["P0_baseline"][a]) for a in am]
        comp.append({"cenario": cenario, "modelo": nome, "delta_pr_auc_vs_P0": average_precision_score(y, pp) - average_precision_score(y, probs["P0_baseline"]),
                     "ic95_lo": np.percentile(d, 2.5), "ic95_hi": np.percentile(d, 97.5), "frac_boot_pos": float(np.mean(np.array(d) > 0)),
                     "delta_brier_vs_P0": brier_score_loss(y, pp) - brier_score_loss(y, probs["P0_baseline"]),
                     "mudanca_media_abs_prob": float(np.mean(np.abs(pp - probs["P0_baseline"]))),
                     "mudanca_p95_abs_prob": float(np.percentile(np.abs(pp - probs["P0_baseline"]), 95))})
    # controle embaralhado (20 permutações) para os grupos CROCO
    for nome in ("P1_+CROCO_CORE", "P2_+CROCO_CURRENT", "P3_+CROCO_DYNAMICS", "P4_+CROCO_TODAS"):
        if nome not in probs:
            continue
        cols = [f"croco_{f}" for f in GRUPOS_P[nome]]
        real = average_precision_score(y, probs[nome])
        perms = [average_precision_score(y, avalia(D, BASE + cols, embaralha=cols, seed=s)) for s in range(20)]
        emb.append({"cenario": cenario, "modelo": nome, "pr_auc_real": real, "pr_auc_embaralhado_media": np.mean(perms),
                    "pr_auc_embaralhado_dp": np.std(perms), "pr_auc_embaralhado_max": np.max(perms),
                    "pr_auc_P0": average_precision_score(y, probs["P0_baseline"]),
                    "p_valor_empirico": (1 + sum(p >= real for p in perms)) / (1 + len(perms))})
        print(cenario, nome, f"real {real:.4f} emb {np.mean(perms):.4f}±{np.std(perms):.4f}", f"{time.time() - t00:.0f}s", flush=True)
    # importância (P4 treinado em todos os meses do cenário): ganho e SHAP médio |contrib|
    feats = BASE + CF
    mod = ajusta(D, feats)
    ganho = mod.get_booster().get_score(importance_type="total_gain")
    contrib = mod.get_booster().predict(xgb.DMatrix(D[feats], base_margin=margem(D)), pred_contribs=True)[:, :-1]
    shap = np.abs(contrib).mean(axis=0); tot = shap.sum()
    for f, s in zip(feats, shap):
        imp.append({"cenario": cenario, "feature": f, "tipo": "CROCO" if f.startswith("croco_") else "ML", "shap_medio_abs": s,
                    "frac_shap": s / tot, "ganho_total": ganho.get(f, 0.0)})
pd.DataFrame(metr).to_csv(O / "metricas_modelos.csv", index=False, float_format="%.5f")
pd.DataFrame(comp).to_csv(O / "comparacao_modelos.csv", index=False, float_format="%.5f")
pd.DataFrame(emb).to_csv(O / "controle_embaralhado.csv", index=False, float_format="%.5f")
imp = pd.DataFrame(imp).sort_values(["cenario", "shap_medio_abs"], ascending=[True, False]); imp.to_csv(O / "feature_importance.csv", index=False, float_format="%.6g")
res = {"aviso": AVISO, "cobertura": cob, "segundos": round(time.time() - t00),
       "metricas": pd.DataFrame(metr).round(4).to_dict("records"), "comparacao": pd.DataFrame(comp).round(4).to_dict("records"),
       "controle_embaralhado": pd.DataFrame(emb).round(4).to_dict("records"),
       "importancia_croco_frac_total": imp[imp.tipo == "CROCO"].groupby("cenario").frac_shap.sum().round(4).to_dict()}
json.dump(res, open(O / "resultados.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(pd.DataFrame(metr).round(3).to_string()); print(pd.DataFrame(comp).round(4).to_string()); print(pd.DataFrame(emb).round(4).to_string())
print(imp.groupby("cenario").head(15).round(4).to_string())
print("ok", round(time.time() - t00), "s")
