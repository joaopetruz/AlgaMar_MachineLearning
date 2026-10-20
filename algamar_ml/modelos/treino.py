"""Split temporal, treino e registro do modelo de previsão multi-horizonte.

Split (config [split]), aplicado ao par (mês de emissão t0, mês-alvo τ = t0+h):
- treino:    τ <= fim_treino
- validação: t0 >= inicio_validacao e τ <= fim_validacao
- teste:     t0 >= inicio_teste (alvos até o último mês observado)
Nenhum alvo aparece em dois conjuntos. A validação escolhe o limiar (máximo F1) e a parada
do treino (early stopping); o teste só é usado para relatar.
"""
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from algamar_ml.features.horizonte import indice_mes


def mes_cfg(texto):
    a, m = texto.split("-")
    return int(indice_mes(int(a), int(m)))


def divide(linhas, cfg_split):
    t0, tau = linhas["t0"].to_numpy(), linhas["tau"].to_numpy()
    observado = linhas["floracao"].notna().to_numpy()
    treino = observado & (tau <= mes_cfg(cfg_split["fim_treino"]))
    validacao = observado & (t0 >= mes_cfg(cfg_split["inicio_validacao"])) & (tau <= mes_cfg(cfg_split["fim_validacao"]))
    teste = observado & (t0 >= mes_cfg(cfg_split["inicio_teste"]))
    return treino, validacao, teste


def novo_modelo(cfg_modelo):
    return XGBClassifier(
        n_estimators=cfg_modelo["n_estimators"], max_depth=cfg_modelo["max_depth"],
        learning_rate=cfg_modelo["learning_rate"], subsample=cfg_modelo["subsample"],
        colsample_bytree=cfg_modelo["colsample_bytree"], random_state=cfg_modelo["seed"],
        min_child_weight=cfg_modelo.get("min_child_weight", 1), tree_method="hist", n_jobs=-1, eval_metric="logloss", early_stopping_rounds=40,
    )


def probabilidade_base(linhas, p0, alpha=2.0):
    """Ponto de partida climatológico: frequência histórica de floração no mês-alvo daquele ponto,
    suavizada para a taxa geral do treino p0 quando há poucos anos: (k + alpha·p0) / (n + alpha)."""
    n = linhas["n_anos_alvo"].fillna(0).to_numpy(dtype=float)
    k = linhas["freq_floracao_alvo"].fillna(0).to_numpy(dtype=float) * n
    return np.clip((k + alpha * p0) / (n + alpha), 1e-4, 1 - 1e-4)


def margem_base(linhas, p0):
    p = probabilidade_base(linhas, p0)
    return np.log(p / (1 - p))


def treina(linhas, features, cfg):
    """Com cfg["modelo"]["base_climatologica"], o modelo parte da climatologia do ponto/mês
    (base_margin em log-odds) e aprende só o desvio explicado pelo estado atual."""
    treino, validacao, _ = divide(linhas, cfg["split"])
    modelo = novo_modelo(cfg["modelo"])
    X, y = linhas[features], linhas["floracao"]
    p0 = float(y[treino].mean())
    extra = {}
    if cfg["modelo"].get("base_climatologica", False):
        extra = {"base_margin": margem_base(linhas[treino], p0),
                 "base_margin_eval_set": [margem_base(linhas[validacao], p0)]}
    modelo.fit(X[treino], y[treino], eval_set=[(X[validacao], y[validacao])], verbose=False, **extra)
    modelo.algamar_p0 = p0
    return modelo


def probabilidade(modelo, linhas, cartao_ou_cfg):
    """Probabilidade prevista, aplicando a base climatológica quando o modelo foi treinado com ela."""
    features = cartao_ou_cfg["features"]
    if cartao_ou_cfg.get("base_climatologica", False):
        return modelo.predict_proba(linhas[features], base_margin=margem_base(linhas, cartao_ou_cfg["p0"]))[:, 1]
    return modelo.predict_proba(linhas[features])[:, 1]


def commit_git():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except OSError:
        return None


def salva(modelo, cartao, pasta, sobrescrever=False):
    """Grava o modelo (formato nativo do XGBoost, JSON) e o cartão com features, flags, split,
    limiar, métricas e proveniência dos dados. 13.8: nunca sobrescreve uma versão existente (sobrescrever=False)."""
    pasta = Path(pasta)
    if (pasta / "modelo.json").exists() and not sobrescrever:
        raise FileExistsError(f"{pasta} já tem um modelo; modelos antigos não são sobrescritos — use outra versão")
    pasta.mkdir(parents=True, exist_ok=True)
    if hasattr(modelo, "salva_em"):                                    # 14.1: ensemble de ordens (membro_XX.json + descritor)
        modelo.salva_em(pasta)
    else:
        modelo.get_booster().save_model(str(pasta / "modelo.json"))
    cartao = {**cartao, "salvo_em": dt.datetime.now(dt.UTC).isoformat(), "commit": commit_git()}
    (pasta / "cartao.json").write_text(json.dumps(cartao, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")


def carrega(pasta):
    pasta = Path(pasta)
    cartao = json.loads((pasta / "cartao.json").read_text(encoding="utf-8"))
    desc = json.loads((pasta / "modelo.json").read_text(encoding="utf-8"))
    if desc.get("tipo") == "ensemble_ordens":                         # 14.1
        from algamar_ml.modelos.ensemble import EnsembleOrdens
        return EnsembleOrdens.de_descritor(desc, pasta), cartao
    modelo = XGBClassifier()
    modelo.load_model(str(pasta / "modelo.json"))
    return modelo, cartao


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)
