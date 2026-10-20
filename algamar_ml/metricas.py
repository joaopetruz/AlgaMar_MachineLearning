"""Métricas da previsão: geral, por horizonte, por região, por estação e calibração.

O alvo é raro (~19 %), então a métrica principal é a PR-AUC (average precision); acurácia
não é usada. As probabilidades são comparadas com duas referências sem ML:
- climatologia: frequência histórica de floração no mês-alvo naquele ponto (anos anteriores);
- persistência: houve floração no mesmo mês do ano anterior (0/1).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score,
)

ESTACAO = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def escolhe_limiar(y, p):
    """Limiar de máximo F1 (regra fixa, aplicada na validação)."""
    candidatos = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [f1_score(y, p >= c, zero_division=0) for c in candidatos]
    return float(candidatos[int(np.argmax(f1s))])


def basicas(y, p, limiar=None):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "positivos": int(y.sum()), "nota": "uma só classe: métricas de ranking indefinidas"}
    m = {"n": int(len(y)), "positivos": int(y.sum()), "taxa_positivos": float(y.mean()),
         "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
         "brier": float(brier_score_loss(y, np.clip(p, 0, 1))), "prob_media": float(p.mean())}
    if limiar is not None:
        pred = p >= limiar
        m.update({"limiar": limiar, "f1": float(f1_score(y, pred, zero_division=0)),
                  "recall": float(recall_score(y, pred, zero_division=0)),
                  "precision": float(precision_score(y, pred, zero_division=0))})
    return m


def calibracao(y, p, bins=10):
    """Curva de confiabilidade: prob. média prevista × frequência observada por faixa."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    faixa = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        sel = faixa == b
        if sel.any():
            out.append({"faixa": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int(sel.sum()),
                        "prob_prevista": float(p[sel].mean()), "freq_observada": float(y[sel].mean())})
    return out


def referencias(linhas):
    return {"climatologia": linhas["freq_floracao_alvo"].to_numpy(dtype=float),
            "persistencia": linhas["floracao_alvo_ano_anterior"].to_numpy(dtype=float)}


def relatorio(linhas, prob, limiar, regioes=None):
    """Métricas do modelo e das referências: geral, por horizonte, região e estação."""
    y = linhas["floracao"].to_numpy(dtype=int)
    refs = referencias(linhas)
    grupos = {"h": linhas["h"].to_numpy(), "estacao_alvo": linhas["mes_alvo"].map(ESTACAO).to_numpy()}
    if regioes is not None:
        grupos["regiao"] = np.asarray(regioes)
    rel = {"geral": {"modelo": basicas(y, prob, limiar), **{k: basicas(y, v) for k, v in refs.items()}},
           "calibracao": calibracao(y, prob)}
    for nome, g in grupos.items():
        rel[f"por_{nome}"] = {}
        for valor in pd.unique(g):
            sel = g == valor
            rel[f"por_{nome}"][str(valor)] = {"modelo": basicas(y[sel], prob[sel], limiar),
                                              **{k: basicas(y[sel], v[sel]) for k, v in refs.items()}}
    return rel


def bootstrap_delta(y, p_a, p_b, blocos, n=200, seed=0, metrica=average_precision_score):
    """IC 95 % de metrica(p_b) - metrica(p_a), reamostrando blocos inteiros (ex.: ano × região)."""
    y, p_a, p_b, blocos = map(np.asarray, (y, p_a, p_b, blocos))
    rng = np.random.default_rng(seed)
    unicos = np.unique(blocos)
    indices = {b: np.where(blocos == b)[0] for b in unicos}
    deltas = []
    for _ in range(n):
        amostra = np.concatenate([indices[b] for b in rng.choice(unicos, len(unicos), replace=True)])
        if len(np.unique(y[amostra])) < 2:
            continue
        deltas.append(metrica(y[amostra], p_b[amostra]) - metrica(y[amostra], p_a[amostra]))
    deltas = np.array(deltas)
    return {"delta": float(metrica(y, p_b) - metrica(y, p_a)), "ic95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "reamostragens": int(len(deltas)), "blocos": int(len(unicos))}
