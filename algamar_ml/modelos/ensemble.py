"""Ensemble de ordens (Etapa 14.1): K XGBoost iguais, cada um treinado com as linhas de treino numa ordem diferente.

Por quê: a 13.6 mediu que só a ordem das linhas de treino muda a PR-AUC em até ~0,02 (com subsample = 0,8 a ordem decide
quais linhas entram em cada árvore). A média das probabilidades de K ordens tira essa loteria do modelo registrado.

Aleatoriedade reprodutível com chaves derivadas, nunca reutilizadas:
- seed_mestra → numpy.random.SeedSequence(seed_mestra).spawn(K): um filho por membro;
- cada filho → spawn(2): uma chave para a permutação das linhas, outra para a seed interna do XGBoost.
As K × 2 seeds e as spawn_keys vão para o cartão e para o modelo.json do ensemble.

Funções puras: nada aqui altera o DataFrame recebido (as permutações criam cópias); com_calibracao devolve um ensemble novo.
Calibração: isotônica ajustada SÓ na validação (ajusta_isotonica); o limiar é re-escolhido na validação, nunca no teste.
"""
import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from algamar_ml.modelos import treino as tr

TIPO = "ensemble_ordens"


def seeds_derivadas(seed_mestra, k):
    """K pares (seed da permutação, seed do XGBoost) derivados de SeedSequence(seed_mestra); nenhum par repete chave."""
    out = []
    for i, filho in enumerate(np.random.SeedSequence(int(seed_mestra)).spawn(int(k))):
        ss_perm, ss_xgb = filho.spawn(2)
        out.append({"membro": i, "spawn_key": [int(x) for x in filho.spawn_key],
                    "seed_permutacao": int(ss_perm.generate_state(1, np.uint32)[0]),
                    "seed_xgboost": int(ss_xgb.generate_state(1, np.uint32)[0] % (2**31 - 1))})
    return out


def permutacao(n, seed_permutacao):
    return np.random.default_rng(int(seed_permutacao)).permutation(int(n))


def ajusta_isotonica(y, p):
    """Calibração isotônica (nós x→y, interpolação linear, extremos presos), ajustada nos dados recebidos (validação)."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(p, float), np.asarray(y, float))
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def aplica_isotonica(cal, p):
    return np.interp(np.asarray(p, float), cal["x"], cal["y"])


class EnsembleOrdens:
    """Média das probabilidades dos membros (+ isotônica opcional). Mesma interface que tr.probabilidade espera."""

    def __init__(self, membros, seeds, seed_mestra, p0=None, isotonica=None):
        self.membros = list(membros)
        self.seeds = list(seeds)
        self.seed_mestra = int(seed_mestra)
        self.algamar_p0 = p0
        self.isotonica = isotonica

    @property
    def arquivos_membros(self):
        return tuple(f"membro_{i:02d}.json" for i in range(len(self.membros)))

    @property
    def melhores_iteracoes(self):
        def it(m):
            try:
                return int(m.best_iteration)
            except (AttributeError, TypeError):
                return -1
        return [it(m) for m in self.membros]

    def com_calibracao(self, isotonica):
        return EnsembleOrdens(self.membros, self.seeds, self.seed_mestra, self.algamar_p0, isotonica)

    def prob_membros(self, X, base_margin=None):
        """(n, K): probabilidade de cada membro, sem calibração."""
        return np.column_stack([m.predict_proba(X, base_margin=base_margin)[:, 1] for m in self.membros])

    def predict_proba(self, X, base_margin=None, calibrada=True):
        p = self.prob_membros(X, base_margin).mean(axis=1)
        if calibrada and self.isotonica is not None:
            p = aplica_isotonica(self.isotonica, p)
        return np.column_stack([1 - p, p])

    def importancia_gain(self, features):
        tot = np.zeros(len(features))
        for m in self.membros:
            g = m.get_booster().get_score(importance_type="gain")
            v = np.array([g.get(f, 0.0) for f in features])
            tot += v / v.sum() if v.sum() else v
        return dict(zip(features, tot / len(self.membros)))

    def descritor(self):
        return {"tipo": TIPO, "versao_formato": 1, "k": len(self.membros), "membros": list(self.arquivos_membros),
                "seed_mestra": self.seed_mestra, "seeds": self.seeds, "p0": self.algamar_p0,
                "melhores_iteracoes": self.melhores_iteracoes, "isotonica": self.isotonica}

    def salva_em(self, pasta):
        pasta = Path(pasta)
        for m, nome in zip(self.membros, self.arquivos_membros):
            m.get_booster().save_model(str(pasta / nome))
        (pasta / "modelo.json").write_text(json.dumps(self.descritor(), indent=1), encoding="utf-8")

    @classmethod
    def de_descritor(cls, desc, pasta):
        membros = []
        for nome in desc["membros"]:
            m = XGBClassifier()
            m.load_model(str(Path(pasta) / nome))
            membros.append(m)
        return cls(membros, desc["seeds"], desc["seed_mestra"], desc.get("p0"), desc.get("isotonica"))


def treina_ensemble(linhas, features, cfg, k=8, seed_mestra=42, ao_terminar_membro=None):
    """K membros com os hiperparâmetros de cfg["modelo"]; o membro i vê as linhas de treino na permutação i e usa a seed
    XGBoost i (ambas derivadas de seed_mestra). Early stopping na validação, como em tr.treina. Não altera `linhas`."""
    treino, validacao, _ = tr.divide(linhas, cfg["split"])
    X_tr, y_tr = linhas.loc[treino, features], linhas.loc[treino, "floracao"].to_numpy()
    X_va, y_va = linhas.loc[validacao, features], linhas.loc[validacao, "floracao"].to_numpy()
    p0 = float(y_tr.mean())
    base = bool(cfg["modelo"].get("base_climatologica", False))
    m_tr = tr.margem_base(linhas[treino], p0) if base else None
    m_va = tr.margem_base(linhas[validacao], p0) if base else None
    seeds = seeds_derivadas(seed_mestra, k)
    membros = []
    for s in seeds:
        perm = permutacao(len(y_tr), s["seed_permutacao"])
        modelo = tr.novo_modelo({**cfg["modelo"], "seed": s["seed_xgboost"]})
        extra = {"base_margin": m_tr[perm], "base_margin_eval_set": [m_va]} if base else {}
        modelo.fit(X_tr.iloc[perm], y_tr[perm], eval_set=[(X_va, y_va)], verbose=False, **extra)
        membros.append(modelo)
        if ao_terminar_membro:
            ao_terminar_membro(s, modelo)
    return EnsembleOrdens(membros, seeds, seed_mestra, p0)
