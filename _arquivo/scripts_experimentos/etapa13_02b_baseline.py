"""Etapa 13.2B — E0_original × E0_corrigido no mesmo teste 2022-2024, com manifesto de reprodutibilidade.

E0_original: features antigas (constroi_features_original, idênticas a feature_engineering_sazonal.py de cc3145c)
             e protocolo antigo (treino <= 2021, teste 2022-2024, limiar fixo 0,5). Preservado só como referência.
E0_corrigido: features sem vazamento (constroi_features) e protocolo novo (treino 2001-2018, validação 2019-2021,
              teste 2022-2024; limiar de máximo F1 escolhido só na validação e congelado antes do teste).
Mesmo algoritmo e hiperparâmetros (XGBoost 300/6/0,1, scale_pos_weight do treino, seed 42), mesmo alvo
(clorofila_maxima > 1,5) e as mesmas 12 features conceituais. Diagnóstico extra: features antigas no protocolo novo.

uso: python etapa13_02b_baseline.py [--hotspot H0|H1|H2|H3]
  H0 coordenadas fixas originais, distância em graus (como no original)
  H1 coordenadas fixas originais, distância em km
  H2 hotspots recalculados só com os anos de treino (2000-2018), distância em km
  H3 sem dist_hotspot
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from algamar_ml.avaliacao import ANO_FIM_TREINO, ANO_FIM_VALIDACAO, metricas, novo_modelo, treina_e_avalia
from algamar_ml.features import CHAVE, FEATURES_BASELINE, HOTSPOTS, constroi_features, constroi_features_original
from algamar_ml.features.sazonal import distancia_hotspot_km, hotspots_do_treino

ap = argparse.ArgumentParser()
ap.add_argument("--hotspot", default="H0", choices=["H0", "H1", "H2", "H3"])
ap.add_argument("--saida", default="resultados/etapa13_02b")
args = ap.parse_args()
O = Path(args.saida) / args.hotspot
O.mkdir(parents=True, exist_ok=True)
ENTRADA = Path("dados/dados_mensais_2000_2024_sp_com_salinidade.csv")
SEED = 42


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_df(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


mensal = pd.read_csv(ENTRADA)
original = constroi_features_original(mensal)
corrigido = constroi_features(mensal)
features_corr = list(FEATURES_BASELINE)
hs_usados = HOTSPOTS
if args.hotspot == "H1":
    corrigido["dist_hotspot"] = distancia_hotspot_km(corrigido, HOTSPOTS)
elif args.hotspot == "H2":
    hs_usados = hotspots_do_treino(mensal, ano_fim=ANO_FIM_TREINO)
    corrigido["dist_hotspot"] = distancia_hotspot_km(corrigido, hs_usados)
elif args.hotspot == "H3":
    features_corr = [f for f in FEATURES_BASELINE if f != "dist_hotspot"]
    hs_usados = None

# mesmas linhas nas duas versões (o shift por posição da original pega mês errado em 900 linhas)
comuns = original[CHAVE].merge(corrigido[CHAVE], on=CHAVE)
original = original.merge(comuns, on=CHAVE)
corrigido = corrigido.merge(comuns, on=CHAVE)

# ---------------------------------------------------------------- E0_original (protocolo antigo, preservado)
treino, teste = original["ano"] <= 2021, original["ano"] > 2021
mod_o = novo_modelo(original.loc[treino, "floracao"], SEED)
mod_o.fit(original.loc[treino, FEATURES_BASELINE], original.loc[treino, "floracao"])
p_o = mod_o.predict_proba(original.loc[teste, FEATURES_BASELINE])[:, 1]
E0_original = {"protocolo": "treino <= 2021, sem validação, limiar fixo 0,5 (como no repositório original)",
               "features": "constroi_features_original (com vazamento)", "linhas": {"treino": int(treino.sum()), "teste": int(teste.sum())},
               "teste": metricas(original.loc[teste, "floracao"], p_o, 0.5)}

# ---------------------------------------------------------------- diagnóstico e E0_corrigido (protocolo novo)
_, diag, _ = treina_e_avalia(original, FEATURES_BASELINE, seed=SEED)
mod_c, E0_corrigido, p_c = treina_e_avalia(corrigido, features_corr, seed=SEED)
E0_corrigido.update(features="constroi_features (sem vazamento)", hotspot=args.hotspot, hotspots=hs_usados)

# ---------------------------------------------------------------- saídas
config = {
    "algoritmo": "xgboost.XGBClassifier",
    "hiperparametros": {k: v for k, v in mod_c.get_params().items() if v is not None and k in
                        ("n_estimators", "max_depth", "learning_rate", "random_state", "n_jobs", "eval_metric")},
    "scale_pos_weight": {"E0_original": float(mod_o.get_params()["scale_pos_weight"]),
                         "E0_corrigido": float(mod_c.get_params()["scale_pos_weight"]), "regra": "negativos/positivos do treino"},
    "alvo": "floracao = clorofila_maxima > 1,5 mg/m³ (inalterado)",
    "features_E0_original": FEATURES_BASELINE, "features_E0_corrigido": features_corr,
    "split_E0_corrigido": {"treino": f"<= {ANO_FIM_TREINO}", "validacao": f"{ANO_FIM_TREINO + 1}-{ANO_FIM_VALIDACAO}", "teste": f"> {ANO_FIM_VALIDACAO}"},
    "limiar_E0_corrigido": {"valor": E0_corrigido["limiar_escolhido_na_validacao"], "regra": "máximo F1 na validação, grade 0,05–0,95 passo 0,01; congelado antes do teste"},
    "limiar_E0_original": {"valor": 0.5, "regra": "fixo, como no original"},
}
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
sujo = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
import sklearn, xgboost, scipy  # noqa: E402
manifest = {
    "etapa": "13.2B", "data_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
    "xgboost": xgboost.__version__, "scipy": scipy.__version__, "seed": SEED,
    "commit": commit, "arquivos_modificados_nao_commitados": sujo.splitlines(),
    "dataset_mensal": {"arquivo": str(ENTRADA), "sha256": sha(ENTRADA), "linhas": int(len(mensal))},
    "features_E0_original_sha256_conteudo": sha_df(original[CHAVE + FEATURES_BASELINE + ["floracao"]]),
    "features_E0_corrigido_sha256_conteudo": sha_df(corrigido[CHAVE + features_corr + ["floracao"]]),
    "linhas_comuns": int(len(comuns)),
    "comando": " ".join([Path(sys.executable).name] + sys.argv),
}
json.dump(E0_original, open(O / "metricas_E0_original.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(E0_corrigido, open(O / "metricas_E0_corrigido.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
json.dump(diag, open(O / "metricas_diagnostico_features_antigas_protocolo_novo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(config, open(O / "config_modelos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
json.dump({"limiar": E0_corrigido["limiar_escolhido_na_validacao"], "fonte": "validação 2019-2021", "metrica_validacao": E0_corrigido["validacao"]},
          open(O / "limiar.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(manifest, open(O / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
lin = []
for nome, m in [("E0_original", E0_original["teste"]), ("features_antigas_protocolo_novo", diag["teste"]), ("E0_corrigido", E0_corrigido["teste"])]:
    tn, fp, fn, tp = np.ravel(m["matriz_confusao"])
    lin.append({"modelo": nome, **{k: m[k] for k in ("n", "positivos", "pr_auc", "roc_auc", "f1", "recall", "precision", "brier", "log_loss", "limiar")},
                "TN": tn, "FP": fp, "FN": fn, "TP": tp})
c = pd.DataFrame(lin)
c.to_csv(O / "comparacao_teste_2022_2024.csv", index=False, float_format="%.4f")
print(c.to_string(index=False))
