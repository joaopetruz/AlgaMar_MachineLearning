"""Split temporal, escolha do limiar na validação e métricas.

Split padrão (13.2A, proposta A): treino <= 2018, validação 2019-2021, teste 2022-2024.
O teste é o mesmo do baseline antigo; ele só é olhado DEPOIS de o limiar estar fixado.
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

ANO_FIM_TREINO = 2018
ANO_FIM_VALIDACAO = 2021


def divide(tabela, ano_fim_treino=ANO_FIM_TREINO, ano_fim_validacao=ANO_FIM_VALIDACAO):
    treino = tabela["ano"] <= ano_fim_treino
    validacao = (tabela["ano"] > ano_fim_treino) & (tabela["ano"] <= ano_fim_validacao)
    teste = tabela["ano"] > ano_fim_validacao
    return treino, validacao, teste


def novo_modelo(y_treino, seed=42):
    peso = (y_treino == 0).sum() / (y_treino == 1).sum()
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=peso, random_state=seed, n_jobs=-1,
        eval_metric="logloss",
    )


def escolhe_limiar(y, probabilidades):
    """Limiar de máximo F1 (regra fixa, igual para todos os modelos)."""
    candidatos = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [f1_score(y, (probabilidades >= l).astype(int), zero_division=0) for l in candidatos]
    return float(candidatos[int(np.argmax(f1s))])


def metricas(y, probabilidades, limiar):
    y_pred = (probabilidades >= limiar).astype(int)
    return {
        "n": int(len(y)),
        "positivos": int(y.sum()),
        "pr_auc": float(average_precision_score(y, probabilidades)),
        "roc_auc": float(roc_auc_score(y, probabilidades)),
        "brier": float(brier_score_loss(y, probabilidades)),
        "log_loss": float(log_loss(y, np.clip(probabilidades, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "limiar": limiar,
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "matriz_confusao": confusion_matrix(y, y_pred).tolist(),
    }


def treina_e_avalia(tabela, features, alvo="floracao", seed=42):
    """Treina no treino, fixa o limiar na validação e só então mede no teste."""
    treino, validacao, teste = divide(tabela)
    X, y = tabela[features], tabela[alvo]

    modelo = novo_modelo(y[treino], seed)
    modelo.fit(X[treino], y[treino])

    prob_val = modelo.predict_proba(X[validacao])[:, 1]
    limiar = escolhe_limiar(y[validacao], prob_val)
    prob_teste = modelo.predict_proba(X[teste])[:, 1]

    return modelo, {
        "split": {"treino": f"<= {ANO_FIM_TREINO}", "validacao": f"{ANO_FIM_TREINO + 1}-{ANO_FIM_VALIDACAO}", "teste": f"> {ANO_FIM_VALIDACAO}"},
        "linhas": {"treino": int(treino.sum()), "validacao": int(validacao.sum()), "teste": int(teste.sum())},
        "limiar_escolhido_na_validacao": limiar,
        "validacao": metricas(y[validacao], prob_val, limiar),
        "teste": metricas(y[teste], prob_teste, limiar),
    }, prob_teste
