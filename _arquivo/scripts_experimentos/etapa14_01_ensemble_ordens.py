"""Etapa 14.1 — ensemble de ordens (K = 8) + recalibração isotônica, comparado com o oficial no MESMO teste.

1. Registra a versão NOVA risco-sat-clim-sp-v2-ens: 8 XGBoost com os hiperparâmetros de [modelo], cada um em uma ordem
   das linhas de treino, seeds de SeedSequence(42).spawn(8) gravadas no cartão; isotônica e limiar escolhidos só na
   validação. Se a versão já existir, é lida do registro (nada é sobrescrito).
2. Variância entre ordens: mais 4 ensembles independentes (seeds mestras 1–4, 32 membros). "Antes" = PR-AUC de teste de
   cada um dos 40 modelos simples (uma ordem cada); "depois" = PR-AUC dos 5 ensembles de 8.
3. Comparação no teste do config (emissões ≥ 2022-01, h = 1..12, painel SP total): oficial (registro, sem retreino),
   v2 calibrada, v2 bruta e climatologia (freq_floracao_alvo, como na 13.7). PR-AUC, ROC-AUC, Brier, log loss, F1 por
   horizonte; IC 95 % por bootstrap em blocos ano do alvo × região × trimestre (como na 13.7), 200 reamostragens.
4. Calibração: confiabilidade por faixa de 10 % e ECE, validação e teste.
Saídas: resultados/etapa14_01_ensemble/; probabilidades intermediárias em dados/etapa14_01/ (cache, gitignored).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score

from algamar_ml import config, metricas, pipeline, registro
from algamar_ml.espacial import regiao
from algamar_ml.modelos import ensemble as ens
from algamar_ml.modelos import treino as tr

t00 = time.time()
O = Path("resultados/etapa14_01_ensemble"); O.mkdir(parents=True, exist_ok=True)
C = Path("dados/etapa14_01"); C.mkdir(parents=True, exist_ok=True)
cfg = config.carrega()
OFICIAL, NOVA = cfg["modelo"]["oficial"], "risco-sat-clim-sp-v2-ens"
ENS = {"k": int(cfg["ensemble"]["k"]), "seed_mestra": int(cfg["ensemble"]["seed_mestra"]), "calibracao": cfg["ensemble"]["calibracao"]}
SEEDS_EXTRA = [1, 2, 3, 4]
B = 200


def log(msg):
    print(f"[{time.time() - t00:6.0f}s] {msg}", flush=True)


m_of, c_of, _ = registro.carrega(cfg["caminhos"]["modelos"], OFICIAL)
painel = pd.read_parquet(cfg["caminhos"]["painel"])
cubo = pipeline.cubo_do_painel(painel)
linhas = pipeline.dataset(cfg, cubo, {})
feats = pipeline.features_de(cfg, [])
assert feats == c_of["features"], "features do config diferem das do oficial; a comparação não seria justa"
tr_, va, te = tr.divide(linhas, cfg["split"])
log(f"linhas {len(linhas):,} | treino {tr_.sum():,} | validação {va.sum():,} | teste {te.sum():,}")
uso = {"features": feats, "base_climatologica": True}
LV, LT = linhas[va], linhas[te]
y_va, y = LV["floracao"].to_numpy(int), LT["floracao"].to_numpy(int)
bm_va, bm_te = tr.margem_base(LV, c_of["p0"]), tr.margem_base(LT, c_of["p0"])   # p0 é o mesmo (mesmo treino)

# ------------------------------------------------------------------ 1. versão nova (registrada uma vez só)
pasta_nova = Path(cfg["caminhos"]["modelos"]) / NOVA
if not pasta_nova.exists():
    log(f"treinando {NOVA} (k = {ENS['k']}, seed_mestra = {ENS['seed_mestra']})")
    m2, c2, _ = pipeline.treina_e_avalia(cfg, linhas, feats, NOVA, pipeline.flags_de(cfg),
                                         {"painel": {"arquivo": str(cfg["caminhos"]["painel"]), "sha256": registro.sha256(cfg["caminhos"]["painel"])},
                                          "seed": ENS["seed_mestra"], "origem": "etapa14_01_ensemble_ordens.py"}, ensemble=ENS)
    man = registro.registra(m2, c2, cfg["caminhos"]["modelos"], NOVA)
    log(f"registrada {NOVA}: {sorted(man['arquivos'])}")
m2, c2, _ = registro.carrega(cfg["caminhos"]["modelos"], NOVA)
assert abs(c2["p0"] - c_of["p0"]) < 1e-12
assert c_of.get("ensemble") is None


# ------------------------------------------------------------------ 2. membros: probabilidades por ordem (cache)
def probs_membros(seed_mestra, modelo=None):
    arq = C / f"membros_seed{seed_mestra}.npz"
    if arq.exists():
        z = np.load(arq); return z["va"], z["te"]
    if modelo is None:
        log(f"treinando ensemble extra seed_mestra = {seed_mestra}")
        modelo = ens.treina_ensemble(linhas, feats, cfg, ENS["k"], seed_mestra)
    pv = modelo.prob_membros(LV[feats], bm_va).astype(np.float32)
    pt = modelo.prob_membros(LT[feats], bm_te).astype(np.float32)
    np.savez(arq, va=pv, te=pt)
    return pv, pt


membros = {ENS["seed_mestra"]: probs_membros(ENS["seed_mestra"], m2)}
for s in SEEDS_EXTRA:
    membros[s] = probs_membros(s)
log("membros prontos")

p_of = tr.probabilidade(m_of, LT, c_of)
p_of_va = tr.probabilidade(m_of, LV, c_of)
p2 = tr.probabilidade(m2, LT, c2)                                      # calibrada (isotônica da validação)
p2b = membros[ENS["seed_mestra"]][1].mean(axis=1).astype(float)        # bruta (média simples)
p2b_va = membros[ENS["seed_mestra"]][0].mean(axis=1).astype(float)
np.testing.assert_allclose(p2b, m2.predict_proba(LT[feats], base_margin=bm_te, calibrada=False)[:, 1], rtol=1e-5)
lim_bruto = c2["ensemble"]["validacao_bruta"]["limiar"]
clim = np.clip(LT["freq_floracao_alvo"].fillna(c_of["p0"]).to_numpy(float), 1e-4, 1 - 1e-4)
clim_va = np.clip(LV["freq_floracao_alvo"].fillna(c_of["p0"]).to_numpy(float), 1e-4, 1 - 1e-4)
h = LT["h"].to_numpy()

# ------------------------------------------------------------------ 3. variância entre ordens (antes × depois)
ap = average_precision_score
var = []
for s, (pv, pt) in membros.items():
    for j in range(pt.shape[1]):
        var.append({"tipo": "modelo simples (1 ordem)", "seed_mestra": s, "membro": j, "pr_auc": ap(y, pt[:, j]),
                    "brier": brier_score_loss(y, pt[:, j])})
    pe = pt.mean(axis=1)
    cal = ens.ajusta_isotonica(y_va, pv.mean(axis=1))
    pc = ens.aplica_isotonica(cal, pe)
    var.append({"tipo": "ensemble de 8 (bruto)", "seed_mestra": s, "membro": -1, "pr_auc": ap(y, pe), "brier": brier_score_loss(y, pe)})
    var.append({"tipo": "ensemble de 8 (isotônica)", "seed_mestra": s, "membro": -1, "pr_auc": ap(y, pc), "brier": brier_score_loss(y, pc)})
var.append({"tipo": "oficial (ordem original)", "seed_mestra": 42, "membro": -1, "pr_auc": ap(y, p_of), "brier": brier_score_loss(y, p_of)})
var = pd.DataFrame(var); var.to_csv(O / "variancia_entre_ordens.csv", index=False, float_format="%.5f")
resumo_var = var.groupby("tipo").agg(n=("pr_auc", "size"), pr_auc_media=("pr_auc", "mean"), pr_auc_dp=("pr_auc", "std"),
                                     pr_auc_min=("pr_auc", "min"), pr_auc_max=("pr_auc", "max"), brier_media=("brier", "mean"),
                                     brier_dp=("brier", "std"))
resumo_var["pr_auc_amplitude"] = resumo_var.pr_auc_max - resumo_var.pr_auc_min
resumo_var.to_csv(O / "variancia_resumo.csv", float_format="%.5f")
log("variância entre ordens\n" + resumo_var.round(4).to_string())
# por horizonte: desvio-padrão da PR-AUC entre os 40 simples e entre os 5 ensembles brutos
todos_pt = np.concatenate([pt for _, pt in membros.values()], axis=1)
var_h = []
for k in range(1, 13):
    m = h == k
    simples = [ap(y[m], todos_pt[m, j]) for j in range(todos_pt.shape[1])]
    ensembles = [ap(y[m], pt[m].mean(axis=1)) for _, pt in membros.values()]
    var_h.append({"h": k, "dp_simples": np.std(simples, ddof=1), "amplitude_simples": np.ptp(simples),
                  "dp_ensemble8": np.std(ensembles, ddof=1), "amplitude_ensemble8": np.ptp(ensembles)})
var_h = pd.DataFrame(var_h); var_h.to_csv(O / "variancia_por_horizonte.csv", index=False, float_format="%.5f")
log("variância por horizonte\n" + var_h.round(4).to_string())

# ------------------------------------------------------------------ 4. comparação no mesmo teste (por horizonte + IC)
reg = np.asarray(regiao(LT["longitude"], cfg["regioes"])).astype(str)
blocos = (LT["tau"] // 12).astype(str).to_numpy() + "|" + reg + "|" + ((LT["tau"] % 12) // 3).astype(str).to_numpy()
rng = np.random.default_rng(0)
MOD = {"oficial": (p_of, c_of["limiar_alto"]), "v2_ens_iso": (p2, c2["limiar_alto"]), "v2_ens_bruto": (p2b, lim_bruto),
       "climatologia": (clim, None)}


def met(yy, pp, lim):
    r = {"pr_auc": ap(yy, pp), "roc_auc": roc_auc_score(yy, pp), "brier": brier_score_loss(yy, pp),
         "log_loss": log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6))}
    if lim is not None:
        r["f1"] = f1_score(yy, pp >= lim)
    return r


lin = []
for k in list(range(1, 13)) + ["todos"]:
    m = np.ones(len(y), bool) if k == "todos" else h == k
    yy = y[m]
    r = {"h": k, "n": int(m.sum()), "taxa_obs": float(yy.mean())}
    for nome, (pp, lim) in MOD.items():
        r.update({f"{q}_{nome}": v for q, v in met(yy, pp[m], lim).items()})
    bb = blocos[m]; ub = np.unique(bb); ix = {b: np.where(bb == b)[0] for b in ub}
    d = {"pr_v2_of": [], "brier_v2_of": [], "pr_v2_clim": [], "pr_of_clim": [], "pr_v2b_of": []}
    po, pv2, pv2b, pc = p_of[m], p2[m], p2b[m], clim[m]
    for _ in range(B):
        a = np.concatenate([ix[b] for b in rng.choice(ub, len(ub))])
        ya = yy[a]; a_of, a_v2, a_c = ap(ya, po[a]), ap(ya, pv2[a]), ap(ya, pc[a])
        d["pr_v2_of"].append(a_v2 - a_of); d["pr_v2_clim"].append(a_v2 - a_c); d["pr_of_clim"].append(a_of - a_c)
        d["pr_v2b_of"].append(ap(ya, pv2b[a]) - a_of)
        d["brier_v2_of"].append(brier_score_loss(ya, pv2[a]) - brier_score_loss(ya, po[a]))
    r["delta_pr_v2_of"] = r["pr_auc_v2_ens_iso"] - r["pr_auc_oficial"]
    r["delta_pr_v2b_of"] = r["pr_auc_v2_ens_bruto"] - r["pr_auc_oficial"]
    r["delta_brier_v2_of"] = r["brier_v2_ens_iso"] - r["brier_oficial"]
    r["delta_pr_v2_clim"] = r["pr_auc_v2_ens_iso"] - r["pr_auc_climatologia"]
    r["delta_pr_of_clim"] = r["pr_auc_oficial"] - r["pr_auc_climatologia"]
    for q, v in d.items():
        r[f"ic95_{q}"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    lin.append(r)
    log(f"h = {k}: PR-AUC of {r['pr_auc_oficial']:.4f} v2 {r['pr_auc_v2_ens_iso']:.4f} clim {r['pr_auc_climatologia']:.4f} | "
        f"Δv2−of {r['delta_pr_v2_of']:+.4f} {np.round(r['ic95_pr_v2_of'], 4)} | Δv2−clim {r['delta_pr_v2_clim']:+.4f} "
        f"{np.round(r['ic95_pr_v2_clim'], 4)} | ΔBrier {r['delta_brier_v2_of']:+.5f} {np.round(r['ic95_brier_v2_of'], 5)}")
comp = pd.DataFrame(lin)
comp.to_json(O / "comparacao_por_horizonte.json", orient="records", indent=1, force_ascii=False)
plano = comp.copy()
for c in [c for c in plano if c.startswith("ic95_")]:
    plano[c + "_lo"] = plano[c].str[0]; plano[c + "_hi"] = plano[c].str[1]; plano = plano.drop(columns=c)
plano.to_csv(O / "comparacao_por_horizonte.csv", index=False, float_format="%.5f")


# ------------------------------------------------------------------ 5. calibração
def ece(yy, pp, bins=10):
    f = np.minimum((pp * bins).astype(int), bins - 1)
    return float(sum(abs(pp[f == b].mean() - yy[f == b].mean()) * (f == b).mean() for b in range(bins) if (f == b).any()))


cal = []
for conj, yy, probs in (("validacao", y_va, {"oficial": p_of_va, "v2_ens_bruto": p2b_va, "v2_ens_iso": tr.probabilidade(m2, LV, c2),
                                             "climatologia": clim_va}),
                        ("teste", y, {"oficial": p_of, "v2_ens_bruto": p2b, "v2_ens_iso": p2, "climatologia": clim})):
    for nome, pp in probs.items():
        for r in metricas.calibracao(yy, pp):
            cal.append({"conjunto": conj, "modelo": nome, **r})
        cal.append({"conjunto": conj, "modelo": nome, "faixa": "ECE", "n": int(len(yy)), "prob_prevista": float(pp.mean()),
                    "freq_observada": float(yy.mean()), "ece": ece(yy, pp)})
cal = pd.DataFrame(cal); cal.to_csv(O / "calibracao.csv", index=False, float_format="%.4f")
log("calibração (ECE)\n" + cal[cal.faixa == "ECE"][["conjunto", "modelo", "prob_prevista", "freq_observada", "ece"]].round(4).to_string())
log("confiabilidade no teste\n" + cal[(cal.conjunto == "teste") & (cal.faixa != "ECE")]
    .pivot(index="faixa", columns="modelo", values="freq_observada").round(3).to_string())

g = comp[comp.h == "todos"].iloc[0]
json.dump({"oficial": OFICIAL, "nova": NOVA, "ensemble": {**ENS, "seeds": c2["ensemble"]["seeds"]},
           "limiar_oficial": c_of["limiar_alto"], "limiar_v2": c2["limiar_alto"], "limiar_v2_bruto": lim_bruto,
           "isotonica_nos": len(m2.isotonica["x"]), "melhores_iteracoes_v2": c2["ensemble"]["melhores_iteracoes"],
           "seeds_extra_variancia": SEEDS_EXTRA, "reamostragens_bootstrap": B,
           "geral": {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in g.items()},
           "criterio_promocao": "IC 95 % do Δ PR-AUC (v2 − oficial) acima de zero e Δ Brier ≤ 0",
           "passa_criterio": bool(g["ic95_pr_v2_of"][0] > 0 and g["delta_brier_v2_of"] <= 0),
           "segundos": round(time.time() - t00)},
          open(O / "resumo.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
log("ok")
