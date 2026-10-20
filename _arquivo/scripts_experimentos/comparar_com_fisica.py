"""O Navier–Stokes melhorou a previsão? Compara MODELO BASE (histórico + APIs) × MODELO COM FÍSICA.

Regras para a comparação ser justa:
- mesmas linhas (só onde as features físicas existem em t0), mesmo split, mesmos hiperparâmetros e seeds;
- o limiar de cada modelo é escolhido na validação; o teste só é usado para relatar;
- diferença de PR-AUC com IC 95 % por bootstrap em blocos (ano do alvo × região);
- controle opcional: as features físicas embaralhadas entre pontos no mesmo mês
  (se o "ganho" persistir com física embaralhada, ele não vem da física).

Uso: python comparar_com_fisica.py [--seeds 42,43,44] [--controle-embaralhado]
     python comparar_com_fisica.py --grupos CROCO_CORE[,CROCO_VERTICAL,...]   (13.5: tabela mensal do contrato 13.0)
Sem --grupos usa a feature store legada (python ingerir_fisica.py ...).
13.5: com --grupos, as linhas são as dos meses com CROCO em t0 (T de superfície presente), sem os pontos
NAO_USAR_ESTUARIO_CANAL; NaN físico (ex.: 50 m em água rasa) NÃO elimina a linha — fica NaN para o modelo.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from algamar_ml import config, metricas, pipeline
from algamar_ml.espacial import regiao
from algamar_ml.features.horizonte import nomes_features
from algamar_ml.modelos import treino as tr

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="42")
ap.add_argument("--controle-embaralhado", action="store_true")
ap.add_argument("--bootstrap", type=int, default=200)
ap.add_argument("--fonte", default=None, help="fonte na feature store (padrão: config)")
ap.add_argument("--conjunto", default=None, help="conjunto do contrato físico (padrão: config)")
ap.add_argument("--amostra-pontos", type=int, default=0, help="usa só N pontos sorteados (teste rápido)")
ap.add_argument("--grupos", default=None, help="grupos CROCO (13.5), ex.: CROCO_CORE,CROCO_VERTICAL")
args = ap.parse_args()

cfg = config.carrega()
from algamar_ml import croco_ml  # noqa: E402
if args.grupos:
    flags_croco = {f: g in args.grupos.split(",") for f, g in croco_ml.FLAG_GRUPO.items()}
    cfg["features"].update(flags_croco)
else:
    cfg["features"]["use_navier_stokes_features"] = True
if args.fonte:
    cfg["features"]["navier_stokes_fonte"] = args.fonte
if args.conjunto:
    cfg["features"]["navier_stokes_conjunto"] = args.conjunto
cubo, _ = pipeline.carrega_cubo(cfg)
matrizes, nomes_fisica = pipeline.fisica(cfg, cubo)
if args.controle_embaralhado:
    rng = np.random.default_rng(0)
    for nome in list(matrizes):
        a = matrizes[nome].copy()
        for t in range(a.shape[0]):
            rng.shuffle(a[t])
        matrizes[nome + "_embaralhado"] = a
excluir = np.where(getattr(cubo, "croco_validade", np.array([])) == croco_ml.NAO_USAR)[0] if args.grupos else None
linhas = pipeline.dataset(cfg, cubo, matrizes, excluir_nao_usar=excluir)
if args.amostra_pontos:
    escolhidos = np.random.default_rng(1).choice(len(cubo.pontos), args.amostra_pontos, replace=False)
    linhas = linhas[linhas["point_id"].isin(escolhidos)]
com_fisica = (linhas["croco_temperature_surface_t0"].notna() if args.grupos and "croco_temperature_surface_t0" in linhas
              else linhas[nomes_fisica].notna().all(axis=1))
linhas = linhas[com_fisica].reset_index(drop=True)
print(f"Linhas com física em t0 (usadas pelos dois modelos): {len(linhas):,}")

base_flags = {**pipeline.flags_de(cfg), "use_navier_stokes_features": False, **{f: False for f in croco_ml.FLAG_GRUPO}}
fis_flags = {**pipeline.flags_de(cfg), "use_navier_stokes_features": not args.grupos}
modelos = {"base": nomes_features(base_flags), "fisica": nomes_features(fis_flags, nomes_fisica)}
if args.controle_embaralhado:
    modelos["controle_embaralhado"] = nomes_features(fis_flags, [n + "_embaralhado" for n in nomes_fisica])

_, _, teste = tr.divide(linhas, cfg["split"])
y = linhas.loc[teste, "floracao"].to_numpy(dtype=int)
reg = regiao(linhas.loc[teste, "longitude"], cfg["regioes"])
blocos = (linhas.loc[teste, "tau"].to_numpy() // 12).astype(str) + "|" + reg.astype(str)
h = linhas.loc[teste, "h"].to_numpy()

resultado = {"criado_em": dt.datetime.now(dt.UTC).isoformat(), "linhas": int(len(linhas)),
             "fonte": ("croco_mensal:" + args.grupos) if args.grupos else cfg["features"]["navier_stokes_fonte"],
             "conjunto": args.grupos or cfg["features"]["navier_stokes_conjunto"],
             "features_fisicas": nomes_fisica, "seeds": args.seeds, "modelos": {}, "deltas": {}}
probs = {}
for seed in map(int, args.seeds.split(",")):
    cfg["modelo"]["seed"] = seed
    for nome, feats in modelos.items():
        _, cartao, extra = pipeline.treina_e_avalia(cfg, linhas, feats, f"{nome}-seed{seed}", fis_flags if nome != "base" else base_flags)
        resultado["modelos"][f"{nome}-seed{seed}"] = {"limiar": cartao["limiar_alto"], "teste": cartao["teste"]}
        probs.setdefault(nome, []).append(extra["p_teste"])

media = {k: np.mean(v, axis=0) for k, v in probs.items()}
for nome in [k for k in media if k != "base"]:
    d = {"geral": metricas.bootstrap_delta(y, media["base"], media[nome], blocos, n=args.bootstrap)}
    d["por_h"] = {str(k): metricas.bootstrap_delta(y[h == k], media["base"][h == k], media[nome][h == k], blocos[h == k], n=args.bootstrap)
                  for k in range(1, 13)}
    d["brier"] = {"base": float(np.mean((media["base"] - y) ** 2)), nome: float(np.mean((media[nome] - y) ** 2))}
    resultado["deltas"][f"{nome}_menos_base"] = d

pasta = Path("resultados")
pasta.mkdir(exist_ok=True)
saida = pasta / f"comparacao_fisica_{cfg['features']['navier_stokes_fonte']}_{dt.date.today().isoformat()}.json"
saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
for nome, d in resultado["deltas"].items():
    g = d["geral"]
    print(f"{nome}: ΔPR-AUC = {g['delta']:+.4f}  IC95 [{g['ic95'][0]:+.4f}, {g['ic95'][1]:+.4f}]")
    for k, v in d["por_h"].items():
        print(f"   h={k:>2}: {v['delta']:+.4f} [{v['ic95'][0]:+.4f}, {v['ic95'][1]:+.4f}]")
print(f"\nSalvo: {saida}")
print("Critério: a física melhorou se ΔPR-AUC > 0 com IC95 acima de 0, na maioria dos horizontes, e o controle embaralhado não mostrar ganho.")
