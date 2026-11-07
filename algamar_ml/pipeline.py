"""Orquestração: painel → cubo → (física) → dataset multi-horizonte → treino/avaliação → previsão."""
import gc

import numpy as np
import pandas as pd

from algamar_ml import croco_ml, glorys_ml, metricas
from algamar_ml.espacial import regiao
from algamar_ml.features.horizonte import constroi_linhas, cubo_do_painel, indice_mes, nomes_features, texto_mes
from algamar_ml.fisica import contrato as ctr_fisico
from algamar_ml.fisica import feature_store
from algamar_ml.modelos import ensemble as ens
from algamar_ml.modelos import treino as tr

INICIO_EMISSOES = (2001, 1)   # primeiro ano do painel (2000) só alimenta climatologias e defasagens


def carrega_cubo(cfg):
    painel = pd.read_parquet(cfg["caminhos"]["painel"])
    return cubo_do_painel(painel), painel


def fisica(cfg, cubo):
    """Features físicas alinhadas ao cubo (estado no mês de emissão t0).
    13.5: grupos use_croco_* leem a tabela mensal do contrato 13.0 (croco_ml); senão, a feature store legada."""
    f = cfg["features"]
    if glorys_ml.grupos_ativos(f):        # 13.7: GLORYS mensal no mês de emissão (pode somar-se ao CROCO)
        g = glorys_ml.para_cubo(cfg["caminhos"]["glorys_mensal"], cubo, f)
        if croco_ml.grupos_ativos(f):
            m, n = fisica({**cfg, "features": {k: v for k, v in f.items() if not k.startswith("use_glorys")}}, cubo)
            g = {**m, **g}
        return g, list(g)
    if croco_ml.grupos_ativos(f):
        feats = croco_ml.features_ativas(f)
        tabela, rel = croco_ml.carrega(cfg["caminhos"]["croco_mensal"], feats, pontos_ml=cubo.pontos)
        matrizes, rel_cubo = croco_ml.para_cubo(tabela, cubo, feats)
        cubo.croco_validade = croco_ml.validade_por_ponto(tabela, cubo)
        cubo.croco_relatorio = {**rel, **rel_cubo, "grupos": croco_ml.grupos_ativos(f)}
        return matrizes, list(matrizes)
    if not f.get("use_navier_stokes_features", False):
        return {}, []
    tabela, manifesto = feature_store.le(cfg["caminhos"]["feature_store"], f["navier_stokes_fonte"])
    nomes = [n for n in ctr_fisico.conjunto(f["navier_stokes_conjunto"]) if n in tabela.columns]
    faltam = set(ctr_fisico.conjunto(f["navier_stokes_conjunto"])) - set(nomes)
    if faltam:
        print(f"Aviso: a feature store não tem {sorted(faltam)}; seguem só {nomes}")
    matrizes = feature_store.para_cubo(tabela, cubo, nomes)
    return matrizes, list(matrizes)


def dataset(cfg, cubo, matrizes_fisica=None, so_observados=True, excluir_nao_usar=None):
    """Linhas (ponto, t0, h). `excluir_nao_usar`: ids de ponto NAO_USAR_ESTUARIO_CANAL a remover (13.5); use o MESMO
    conjunto no baseline e no modelo com CROCO para que as comparações tenham as mesmas linhas."""
    ini = cfg.get("pipeline", {}).get("inicio_emissoes")            # 13.8: data centralizada no config
    ini = tuple(int(x) for x in ini.split("-")) if ini else INICIO_EMISSOES
    emissoes = range(int(indice_mes(*ini)), cubo.t_fim)
    linhas = constroi_linhas(cubo, cfg["alvo"]["limiar"], emissoes, cfg["previsao"]["horizontes"], fisica=matrizes_fisica)
    if excluir_nao_usar is not None and len(excluir_nao_usar):
        linhas = linhas[~linhas["point_id"].isin(excluir_nao_usar)].reset_index(drop=True)
    if so_observados:
        linhas = linhas[linhas["floracao"].notna()].reset_index(drop=True)
    return linhas


def treina_e_avalia(cfg, linhas, features, versao, flags, extra_cartao=None, ensemble=None):
    """ensemble (14.1): {"k", "seed_mestra", "calibracao"} treina um ensemble de ordens (algamar_ml/modelos/ensemble.py);
    a calibração isotônica e o limiar saem só da validação."""
    treino, validacao, teste = tr.divide(linhas, cfg["split"])
    y_val = linhas.loc[validacao, "floracao"].to_numpy(dtype=int)
    info_ens = None
    if ensemble:
        modelo = ens.treina_ensemble(linhas, features, cfg, int(ensemble["k"]), int(ensemble["seed_mestra"]))
    else:
        modelo = tr.treina(linhas, features, cfg)
    uso = {"features": features, "base_climatologica": bool(cfg["modelo"].get("base_climatologica", False)),
           "p0": modelo.algamar_p0}
    p_val = tr.probabilidade(modelo, linhas[validacao], uso)
    if ensemble:
        info_ens = {"tipo": ens.TIPO, "k": len(modelo.membros), "seed_mestra": modelo.seed_mestra, "seeds": modelo.seeds,
                    "derivacao": "numpy.random.SeedSequence(seed_mestra).spawn(k); cada filho.spawn(2) → (permutação, xgboost)",
                    "melhores_iteracoes": modelo.melhores_iteracoes, "calibracao": ensemble.get("calibracao") or "nenhuma",
                    "validacao_bruta": metricas.basicas(y_val, p_val, metricas.escolhe_limiar(y_val, p_val))}
        if ensemble.get("calibracao") == "isotonica":
            modelo = modelo.com_calibracao(ens.ajusta_isotonica(y_val, p_val))
            p_val = tr.probabilidade(modelo, linhas[validacao], uso)
            info_ens["isotonica_nos"] = len(modelo.isotonica["x"])
    limiar = metricas.escolhe_limiar(y_val, p_val)
    p_teste = tr.probabilidade(modelo, linhas[teste], uso)
    reg_teste = regiao(linhas.loc[teste, "longitude"], cfg["regioes"])
    reg_val = regiao(linhas.loc[validacao, "longitude"], cfg["regioes"])
    cartao = {
        "versao": versao,
        "status": cfg.get("produto", {}).get("status_modelo", "HINDCAST_VALIDADO"),
        "produto": cfg.get("produto", {}).get("nome"),
        "tipo": "previsão direta multi-horizonte (um modelo para h = 1..12)",
        "alvo": cfg["alvo"],
        "features": features,
        "base_climatologica": uso["base_climatologica"],
        "p0": uso["p0"],
        "flags": flags,
        "split": cfg["split"],
        "linhas": {"treino": int(treino.sum()), "validacao": int(validacao.sum()), "teste": int(teste.sum())},
        "periodos": {nome: [str(texto_mes(linhas.loc[m, "tau"].min())), str(texto_mes(linhas.loc[m, "tau"].max()))]
                     for nome, m in (("treino", treino), ("validacao", validacao), ("teste", teste))},
        "melhor_iteracao": int(np.median(modelo.melhores_iteracoes)) if ensemble else int(getattr(modelo, "best_iteration", -1) or -1),
        "limiar_alto": limiar,
        "limiar_moderado": round(limiar * cfg["previsao"]["fracao_moderado"], 4),
        "regra_limiar": "máximo F1 na validação",
        "validacao": metricas.relatorio(linhas[validacao], p_val, limiar, reg_val),
        "teste": metricas.relatorio(linhas[teste], p_teste, limiar, reg_teste),
        **({"ensemble": info_ens} if info_ens else {}),
        **(extra_cartao or {}),
    }
    gain = modelo.importancia_gain(features) if ensemble else modelo.get_booster().get_score(importance_type="gain")
    importancia = pd.Series(gain).reindex(features).fillna(0)
    cartao["importancia_gain"] = (importancia / importancia.sum()).round(4).sort_values(ascending=False).to_dict()
    return modelo, cartao, {"teste": teste, "p_teste": p_teste}


class FeatureSemSinal(ValueError):
    pass


def confere_sem_sinal(flags, cfg_features):
    """Fechamento (2026-10-09): grupos físicos que pioraram o modelo ou não mudaram nada ficam bloqueados
    ([features].sem_sinal). Ligar um deles exige permitir_sem_sinal = true (só para reproduzir experimentos antigos)."""
    ligados = [k for k in cfg_features.get("sem_sinal", []) if flags.get(k, False)]
    if ligados and not cfg_features.get("permitir_sem_sinal", False):
        raise FeatureSemSinal(f"{ligados} estão em [features].sem_sinal: não mostraram sinal (relatório de fechamento). "
                              "Para reproduzir um experimento antigo, use permitir_sem_sinal = true.")


def flags_de(cfg):
    f = cfg["features"]
    out = {k: bool(f.get(k, True)) for k in ("use_historical_features", "use_api_features", "use_salinity",
                                             "use_features_redundantes")}
    out["use_baseline_features"] = bool(f.get("use_baseline_features", out["use_historical_features"]))
    for k in ("use_navier_stokes_features", *croco_ml.FLAG_GRUPO, "use_glorys_ts", "use_glorys_mld"):
        out[k] = bool(f.get(k, False))
    confere_sem_sinal(out, f)
    return out


def features_de(cfg, nomes_fisica):
    return nomes_features(flags_de(cfg), nomes_fisica)


def libera():
    gc.collect()
