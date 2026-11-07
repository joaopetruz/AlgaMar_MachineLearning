"""Mede quanto o vazamento inflava as métricas: features antigas × corrigidas.

Três números no teste 2022-2024:
1. protocolo antigo: features antigas, treino <= 2021, limiar 0,5 (o que o projeto reportava);
2. features antigas no protocolo novo (treino <= 2018, limiar escolhido na validação);
3. features corrigidas no protocolo novo (o baseline E0 a partir de agora).
2 × 3 isola o efeito das features; 1 × 2 isola o efeito do protocolo.
"""
import json
from pathlib import Path

import pandas as pd

from algamar_ml.avaliacao import metricas, novo_modelo, treina_e_avalia
from algamar_ml.features import CHAVE, FEATURES_BASELINE, constroi_features, constroi_features_original

mensal = pd.read_csv("dados/dados_mensais_2000_2024_sp_com_salinidade.csv")
original = constroi_features_original(mensal)
corrigido = constroi_features(mensal)

# Mesmas linhas nas duas versões (as 900 linhas com "ano anterior" errado saem da original)
comuns = original[CHAVE].merge(corrigido[CHAVE], on=CHAVE)
original = original.merge(comuns, on=CHAVE)
corrigido = corrigido.merge(comuns, on=CHAVE)
print(f"Linhas comuns: {len(comuns)}")

# 1. protocolo antigo
treino, teste = original["ano"] <= 2021, original["ano"] > 2021
modelo = novo_modelo(original.loc[treino, "floracao"])
modelo.fit(original.loc[treino, FEATURES_BASELINE], original.loc[treino, "floracao"])
prob = modelo.predict_proba(original.loc[teste, FEATURES_BASELINE])[:, 1]
antigo = metricas(original.loc[teste, "floracao"], prob, 0.5)

# 2 e 3. protocolo novo
_, orig_novo, _ = treina_e_avalia(original, FEATURES_BASELINE)
_, corr_novo, _ = treina_e_avalia(corrigido, FEATURES_BASELINE)

resultado = {
    "linhas_comuns": int(len(comuns)),
    "protocolo_antigo_features_antigas": antigo,
    "protocolo_novo_features_antigas": orig_novo,
    "protocolo_novo_features_corrigidas_E0": corr_novo,
}
print(f"\n{'teste 2022-2024':<38}{'PR-AUC':>8}{'ROC-AUC':>9}{'F1':>8}{'Recall':>8}{'Prec':>8}{'limiar':>8}")
for nome, m in [("antigo (features+protocolo antigos)", antigo),
                ("features antigas, protocolo novo", orig_novo["teste"]),
                ("E0 corrigido", corr_novo["teste"])]:
    print(f"{nome:<38}{m['pr_auc']:>8.4f}{m['roc_auc']:>9.4f}{m['f1']:>8.4f}{m['recall']:>8.4f}{m['precision']:>8.4f}{m['limiar']:>8.2f}")

Path("resultados").mkdir(exist_ok=True)
with open("resultados/comparacao_vazamento.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=1)
print("\nSalvo: resultados/comparacao_vazamento.json")
