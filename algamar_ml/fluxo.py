"""Pipeline final do AlgaMar (Etapa 13.8): um fluxo único, configurado por config/ml.toml e registrado em log.
Produto: [produto].nome no config ("Mapa de Risco de Floração AlgaMar", satélite + climatologia; fechamento 2026-10-09).

  DADOS HISTÓRICOS + APIs AMBIENTAIS (+ física opcional: grupos CROCO/GLORYS, desligados; os sem sinal ficam bloqueados)
    → 1 ingestão → 2 validação → 3 alinhamento temporal → 4 alinhamento espacial (e escolha de t0)
    → 5 feature engineering → 6 modelo → 7 forecast H1–H12 → 8 mapa/risco (saída padrão Parquet + JSON)

Regra central: NUNCA inventar previsão.
- Ponto sem o estado obrigatório em t0 (pipeline.features_obrigatorias_t0): probabilidade nula e nível DADOS_INSUFICIENTES.
- Execução sem mês válido, sem feature exigida pelo modelo ou com pontos insuficientes demais: status DADOS_INSUFICIENTES,
  sem nenhuma probabilidade gravada.

Comportamento por modo de falha:
| falha                           | comportamento                                                                        |
| API indisponível                | registra INDISPONIVEL e segue com os dados locais; t0 recua (fallback) ou status     |
|                                 | DADOS_INSUFICIENTES                                                                  |
| CROCO indisponível              | se o modelo exige features físicas → DADOS_INSUFICIENTES; se não exige → registra e  |
|                                 | segue                                                                                |
| feature ausente                 | coluna exigida pelo modelo ausente → DADOS_INSUFICIENTES; NaN de estado obrigatório  |
|                                 | num ponto → ponto DADOS_INSUFICIENTES                                                |
| arquivo corrompido              | painel ilegível → ERRO_DADOS; arquivo bruto ilegível → variável anulada no mês (NaN) |
|                                 | → fallback                                                                           |
| data fora de ordem              | painel desordenado → reordenado (AVISO); chave repetida → ERRO_DADOS; tempos diários |
|                                 | fora de ordem → arquivo FALHA                                                        |
| coordenada inválida             | NaN ou fora da caixa do ML → linha descartada (AVISO, contada)                       |
| modelo alterado ou corrompido   | ERRO_MODELO (SHA256 diferente do MANIFESTO)                                          |
"""
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as _config, croco_ml, glorys_ml, operacional as op, pipeline, registro
from .espacial import regiao
from .features.horizonte import constroi_linhas, cubo_do_painel, texto_mes
from .modelos import treino as tr

SCHEMA_SAIDA = "1.1.0"   # 1.1.0: previsao.json e ULTIMA.json ganham "produto" (as colunas do Parquet não mudaram)
COLUNAS_SAIDA = ["run_id", "prediction_date", "target_date", "horizon", "point_id", "latitude", "longitude", "regiao",
                 "probabilidade", "probabilidade_climatologica", "classe", "nivel_de_risco", "status_ponto", "modelo",
                 "versao", "status_execucao", "classificacao_produto"]
CHAVE_PAINEL = ["latitude", "longitude", "ano", "mes"]
COLUNAS_PAINEL = CHAVE_PAINEL + ["clorofila_media", "clorofila_maxima", "temperatura_media"]


class FalhaPipeline(Exception):
    def __init__(self, status, motivo):
        super().__init__(f"{status}: {motivo}")
        self.status, self.motivo = status, motivo


def _sha_arquivo(p):
    try:
        return registro.sha256(p)
    except OSError:
        return None


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_config.RAIZ).stdout.strip() or None
    except OSError:
        return None


def caminho(cfg, chave):
    p = Path(cfg["pipeline"][chave])
    return p if p.is_absolute() else _config.RAIZ / p


# ------------------------------------------------------------------ log de execução
class Execucao:
    """Registro de uma execução: timestamp, dados (com hash), modelo, versão, features, período, resultado, erros."""

    def __init__(self, cfg, comando, config_arquivo=None):
        self.cfg = cfg
        self.run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        self.log = {"run_id": self.run_id, "timestamp_inicio": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                    "comando": comando, "commit_git": _commit(), "python": sys.version.split()[0],
                    "config": {"arquivo": str(config_arquivo or _config.PADRAO), "sha256": _sha_arquivo(config_arquivo or _config.PADRAO),
                               "versao_config": cfg.get("versao_config")},
                    "dados": {}, "modelo": {}, "features": [], "periodo": {}, "etapas": {}, "avisos": [], "erros": [],
                    "resultado": {}, "status": None}

    def aviso(self, msg):
        self.log["avisos"].append(msg)

    @contextmanager
    def etapa(self, nome):
        t0 = time.time(); reg = {"inicio": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
        self.log["etapas"][nome] = reg
        try:
            yield reg
            reg.setdefault("status", "OK")
        except FalhaPipeline as e:
            reg["status"] = e.status; reg["motivo"] = e.motivo
            raise
        except Exception as e:
            reg["status"] = "ERRO"; reg["motivo"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            reg["segundos"] = round(time.time() - t0, 2)

    def finaliza(self, status, resultado=None, erro=None):
        self.log["status"] = status
        self.log["resultado"].update(resultado or {})
        if erro is not None:
            self.log["erros"].append(erro)
        self.log["timestamp_fim"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        pasta = caminho(self.cfg, "logs"); pasta.mkdir(parents=True, exist_ok=True)
        txt = json.dumps(self.log, ensure_ascii=False, indent=1, default=str)
        (pasta / f"{self.run_id}.json").write_text(txt, encoding="utf-8")
        with open(pasta / "execucoes.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({k: self.log[k] for k in ("run_id", "timestamp_inicio", "timestamp_fim", "comando", "status")}
                               | {"modelo": self.log["modelo"].get("versao"), "periodo": self.log["periodo"],
                                  "erros": self.log["erros"]}, ensure_ascii=False, default=str) + "\n")
        return self.log


# ------------------------------------------------------------------ 1. ingestão
def ingestao(cfg, ex, consulta=None, baixar=False):
    from .apis.copernicus import PRODUTOS, cobertura
    consulta = consulta or cobertura
    with ex.etapa("1_ingestao") as reg:
        cat = {}
        for var in cfg["pipeline"]["apis"]:
            for p in PRODUTOS.get(var, []):
                try:
                    a, b = consulta(p.dataset_id)
                    cat[p.dataset_id] = {"variavel": var, "tipo": p.tipo, "de": str(a), "ate": str(b), "status": "OK"}
                except Exception as e:
                    cat[p.dataset_id] = {"variavel": var, "tipo": p.tipo, "status": "INDISPONIVEL", "erro": f"{type(e).__name__}: {str(e)[:200]}"}
                    ex.aviso(f"API indisponível: {p.dataset_id} — segue com os dados locais")
        reg["catalogo"] = cat
        reg["download"] = "não solicitado: usa os brutos/painel locais"
        if baixar:
            if all(v["status"] == "INDISPONIVEL" for v in cat.values()):
                reg["download"] = "pulado: APIs indisponíveis"
            else:
                r1 = subprocess.run([sys.executable, "atualizar_dados_atuais.py"], capture_output=True, text=True, cwd=_config.RAIZ)
                r2 = subprocess.run([sys.executable, "construir_painel.py"], capture_output=True, text=True, cwd=_config.RAIZ)
                reg["download"] = {"atualizar_exit": r1.returncode, "painel_exit": r2.returncode}
                if r1.returncode or r2.returncode:
                    ex.aviso("download/painel falhou; segue com os dados locais anteriores")
        if any(v["status"] == "INDISPONIVEL" for v in cat.values()):
            reg["status"] = "AVISO"


# ------------------------------------------------------------------ 2. validação
def le_painel(cfg, ex):
    p = cfg["caminhos"]["painel"]
    try:
        painel = pd.read_parquet(p)
    except Exception as e:
        raise FalhaPipeline("ERRO_DADOS", f"painel ilegível/corrompido ({p}): {type(e).__name__}: {str(e)[:200]}")
    ex.log["dados"]["painel"] = {"arquivo": str(p), "sha256": _sha_arquivo(p), "linhas": int(len(painel))}
    return painel


def validacao(cfg, ex, painel):
    with ex.etapa("2_validacao") as reg:
        falta = [c for c in COLUNAS_PAINEL if c not in painel]
        if falta:
            raise FalhaPipeline("ERRO_DADOS", f"painel sem as colunas {falta}")
        a = cfg["area"]
        ruim = (painel.latitude.isna() | painel.longitude.isna() | (painel.latitude < a["lat_min"] - 1e-6) | (painel.latitude > a["lat_max"] + 1e-6)
                | (painel.longitude < a["lon_min"] - 1e-6) | (painel.longitude > a["lon_max"] + 1e-6)
                | ~painel.mes.between(1, 12))
        if ruim.any():
            ex.aviso(f"{int(ruim.sum())} linhas do painel com coordenada/mês inválidos descartadas")
            painel = painel[~ruim]
        dup = painel.duplicated(CHAVE_PAINEL)
        if dup.any():
            raise FalhaPipeline("ERRO_DADOS", f"{int(dup.sum())} chaves (lat, lon, ano, mes) repetidas no painel")
        t = painel.ano * 12 + painel.mes
        fora = int((~t.groupby([painel.latitude, painel.longitude]).apply(lambda s: s.is_monotonic_increasing)).sum())
        if fora:
            ex.aviso(f"{fora} pontos com meses fora de ordem no painel: reordenado por (ano, mes, lat, lon)")
        painel = painel.sort_values(["ano", "mes", "latitude", "longitude"]).reset_index(drop=True)
        reg.update(coordenadas_invalidas=int(ruim.sum()), pontos_fora_de_ordem=fora, linhas_validas=int(len(painel)),
                   meses=[f"{int(painel.ano.min())}-{int(painel[painel.ano == painel.ano.min()].mes.min()):02d}",
                          f"{int(painel.ano.max())}-{int(painel[painel.ano == painel.ano.max()].mes.max()):02d}"])
        val = []
        for meta_f in sorted(Path(cfg["caminhos"]["brutos"]).glob("*/*/*.json")):
            try:
                meta = json.loads(meta_f.read_text(encoding="utf-8"))
            except Exception as e:
                val.append({"arquivo": str(meta_f), "status": "FALHA", "motivo": f"metadado ilegível: {e}"}); continue
            nc = meta_f.with_suffix(".nc")
            item = {"variavel": meta_f.parent.parent.name, "dataset_id": meta.get("dataset_id"), "ano": meta.get("ano"), "mes": meta.get("mes")}
            if not nc.exists():
                val.append({**item, "status": "FALHA", "motivo": "arquivo .nc ausente"}); continue
            try:
                import xarray as xr
                with xr.open_dataset(nc) as ds:
                    v = op.valida_tempos_diarios(ds["time"].values, meta["ano"], meta["mes"])
                val.append({**item, "status": v["status"], "checagens": v["checagens"]})
            except Exception as e:
                val.append({**item, "status": "FALHA", "motivo": f"arquivo corrompido: {type(e).__name__}: {str(e)[:150]}"})
        reg["brutos"] = {"arquivos": len(val), "por_status": pd.Series([x["status"] for x in val], dtype=object).value_counts().to_dict(),
                         "falhas": [x for x in val if x["status"] == "FALHA"]}
        ex.log["dados"]["brutos_validados"] = len(val)
        if ruim.any() or reg["brutos"]["falhas"]:
            reg["status"] = "AVISO"
    return painel, val


# ------------------------------------------------------------------ 3. alinhamento temporal
def alinhamento_temporal(cfg, ex, painel, val):
    col = {"clorofila": ["clorofila_media", "clorofila_maxima"], "temperatura": ["temperatura_media"], "salinidade": ["salinidade_media"]}
    with ex.etapa("3_alinhamento_temporal") as reg:
        anulados = []
        for x in val:
            if x["status"] == "FALHA" and x.get("ano") and x.get("variavel") in col:
                sel = (painel.ano == x["ano"]) & (painel.mes == x["mes"])
                fonte = f"fonte_{x['variavel']}"
                if fonte in painel and x.get("dataset_id"):
                    sel &= painel[fonte].astype(str).str.contains(str(x["dataset_id"]), regex=False)
                if sel.any():
                    painel.loc[sel, [c for c in col[x["variavel"]] if c in painel]] = np.nan
                    anulados.append(f"{x['variavel']} {x['ano']}-{int(x['mes']):02d}")
        reg.update(regra="mês civil UTC; arquivo bruto com FALHA → variável NaN naquele mês (nunca preenchida)", variaveis_anuladas=anulados)
    return painel


# ------------------------------------------------------------------ 4. alinhamento espacial e escolha de t0
def alinhamento_espacial(cfg, ex, painel, hoje, vars_modelo):
    with ex.etapa("4_alinhamento_espacial") as reg:
        grade = painel[["latitude", "longitude"]].drop_duplicates()
        em = op.decide_emissao(painel, grade, cfg, hoje, vars_modelo, max_atraso_meses=cfg["pipeline"]["max_atraso_meses"],
                               max_recuo=cfg["pipeline"]["max_recuo_meses"])
        reg.update(pontos_grade=int(len(grade)), tentativas=em["tentativas"], status_emissao=em["status"])
        if em["t0"] is None:
            raise FalhaPipeline("DADOS_INSUFICIENTES", em["motivo"])
        ex.log["periodo"] = {"hoje": str(hoje), "emissao_t0": em["issue_month"], "atraso_meses": em["atraso_meses"]}
        if em["status"] != "OK":
            ex.aviso(f"emissão {em['status']}: t0 = {em['issue_month']} ({em['atraso_meses']} meses de atraso)")
    return em


# ------------------------------------------------------------------ 5–7. features, modelo, forecast
def carrega_modelo(cfg, ex, versao):
    try:
        modelo, cartao, man = registro.carrega(cfg["caminhos"]["modelos"], versao)
    except registro.ErroRegistro as e:
        raise FalhaPipeline("ERRO_MODELO", str(e))
    ex.log["modelo"] = {"versao": versao, "manifesto_sha256": man["arquivos"], "status_cartao": cartao.get("status"),
                        "limiar_alto": cartao.get("limiar_alto"), "limiar_moderado": cartao.get("limiar_moderado")}
    ex.log["features"] = list(cartao["features"])
    return modelo, cartao


def _fisica(cfg, ex, cartao, cubo):
    flags = cartao.get("flags", {})
    precisa = flags.get("use_navier_stokes_features") or croco_ml.grupos_ativos(flags) or glorys_ml.grupos_ativos(flags)
    if not precisa:
        ex.log["dados"]["croco"] = "não exigido pelo modelo"
        return {}
    c = {**cfg, "features": {**cfg["features"], **flags}}
    try:
        m, _ = pipeline.fisica(c, cubo)
    except Exception as e:
        raise FalhaPipeline("DADOS_INSUFICIENTES", f"CROCO/física indisponível ({type(e).__name__}: {str(e)[:200]}); o modelo exige features físicas")
    ex.log["dados"]["croco"] = {"grupos": croco_ml.grupos_ativos(flags) + glorys_ml.grupos_ativos(flags), "features": list(m)}
    return m


def forecast(cfg, ex, painel, em, modelo, cartao):
    with ex.etapa("5_features") as reg:
        cubo = cubo_do_painel(painel)
        fis = _fisica(cfg, ex, cartao, cubo)
        linhas = constroi_linhas(cubo, cfg["alvo"]["limiar"], [em["t0"]], horizontes=cfg["previsao"]["horizontes"], fisica=fis,
                                 exige_estado=False)
        falta = [f for f in cartao["features"] if f not in linhas]
        if falta:
            raise FalhaPipeline("DADOS_INSUFICIENTES", f"features exigidas pelo modelo ausentes: {falta}")
        prov = op.proveniencia_features(painel, cartao["features"], em["t0"])
        if not all(p.get("todos_ate_t0", True) for p in prov.values()):
            raise FalhaPipeline("ERRO_DADOS", "feature usando mês posterior a t0 (vazamento)")
        obrig = [f for f in cfg["pipeline"]["features_obrigatorias_t0"] if f in linhas]
        insuf = linhas[obrig].isna().any(axis=1).to_numpy()
        frac = float(insuf.mean()) if len(insuf) else 1.0
        reg.update(linhas=int(len(linhas)), features=len(cartao["features"]), obrigatorias_t0=obrig,
                   frac_pontos_insuficientes=round(frac, 4), proveniencia=prov)
        if frac > cfg["pipeline"]["frac_max_pontos_insuficientes"]:
            raise FalhaPipeline("DADOS_INSUFICIENTES", f"{frac:.0%} dos pontos sem estado obrigatório em t0 ({obrig}); nenhuma probabilidade emitida")
    with ex.etapa("6_modelo") as reg:
        p = tr.probabilidade(modelo, linhas, cartao).astype(float)
        p[insuf] = np.nan                                              # nunca inventar: sem estado → sem probabilidade
        lim_a = cartao["limiar_alto"] if cfg["pipeline"]["limiar"] == "cartao" else float(cfg["pipeline"]["limiar"])
        lim_m = cartao.get("limiar_moderado", lim_a * cfg["previsao"].get("fracao_moderado", 0.5))
        reg.update(limiar_alto=lim_a, limiar_moderado=lim_m, origem_limiar=cfg["pipeline"]["limiar"])
    with ex.etapa("7_forecast") as reg:
        nivel = np.where(np.isnan(p), "DADOS_INSUFICIENTES", np.where(p >= lim_a, "ALTO", np.where(p >= lim_m, "MODERADO", "BAIXO")))
        status_exec = "OK" if em["status"] == "OK" else em["status"]
        prev = pd.DataFrame({
            "run_id": ex.run_id, "prediction_date": texto_mes(linhas["t0"].to_numpy()), "target_date": texto_mes(linhas["tau"].to_numpy()),
            "horizon": linhas["h"].astype("int16"), "point_id": linhas["point_id"].astype("int32"),
            "latitude": linhas["latitude"].astype(float).round(5), "longitude": linhas["longitude"].astype(float).round(5),
            "regiao": regiao(linhas["longitude"], cfg["regioes"]), "probabilidade": np.round(p, 4),
            "probabilidade_climatologica": linhas["freq_floracao_alvo"].astype(float).round(4),
            "classe": pd.array(np.where(np.isnan(p), pd.NA, (p >= lim_a).astype(int)), dtype="Int8"),
            "nivel_de_risco": nivel, "status_ponto": np.where(insuf, "DADOS_INSUFICIENTES", "OK"),
            "modelo": "XGBoost multi-horizonte com base climatológica", "versao": cartao["versao"],
            "status_execucao": status_exec, "classificacao_produto": cfg["pipeline"]["classificacao_produto"]})[COLUNAS_SAIDA]
        reg.update(linhas=int(len(prev)), horizontes=sorted(prev.horizon.unique().tolist()),
                   meses_alvo=[prev.target_date.min(), prev.target_date.max()], pontos_insuficientes=int(insuf.sum() // max(len(cfg["previsao"]["horizontes"]), 1)))
    return prev, status_exec


# ------------------------------------------------------------------ 8. saída padrão (mapa/risco)
def produto(cfg):
    p = cfg.get("produto", {})
    return {"nome": p.get("nome", "AlgaMar"), "id": p.get("id"), "descricao": p.get("descricao"),
            "fontes_de_dados": p.get("fontes_de_dados", [])}


def resumo_risco(prev):
    ok = prev[prev.status_ponto == "OK"]
    g = ok.groupby(["regiao", "target_date", "horizon"])
    r = g.agg(prob_media=("probabilidade", "mean"), prob_max=("probabilidade", "max"),
              prob_climatologica_media=("probabilidade_climatologica", "mean"), pontos=("point_id", "count"),
              frac_alto=("nivel_de_risco", lambda s: float((s == "ALTO").mean()))).reset_index()
    return r


def grava_saida(cfg, ex, status, prev=None, motivo=None):
    pasta = caminho(cfg, "saidas") / ex.run_id; pasta.mkdir(parents=True, exist_ok=True)
    meta = {"schema": SCHEMA_SAIDA, "run_id": ex.run_id, "status": status, "motivo": motivo,
            "produto": produto(cfg), "classificacao_produto": cfg["pipeline"]["classificacao_produto"], "modelo": ex.log["modelo"].get("versao"),
            "periodo": ex.log["periodo"], "gerado_em": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "colunas": COLUNAS_SAIDA, "arquivos": {}}
    if prev is not None:
        prev.to_parquet(pasta / "previsao.parquet", index=False)
        resumo_risco(prev).to_parquet(pasta / "risco_regioes.parquet", index=False)
        meta["arquivos"] = {"previsao": "previsao.parquet", "risco_regioes": "risco_regioes.parquet"}
        meta["resumo"] = {"linhas": int(len(prev)), "pontos": int(prev.point_id.nunique()),
                          "pontos_insuficientes": int(prev[prev.status_ponto != "OK"].point_id.nunique()),
                          "risco_por_horizonte": {int(h): {k: int(v) for k, v in g.nivel_de_risco.value_counts().items()}
                                                  for h, g in prev.groupby("horizon")}}
        meta["sha256"] = {a: registro.sha256(pasta / a) for a in meta["arquivos"].values()}
    (pasta / "previsao.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (caminho(cfg, "saidas") / "ULTIMA.json").write_text(json.dumps({"run_id": ex.run_id, "status": status, "produto": produto(cfg)["nome"], "pasta": str(pasta)},
                                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    return pasta, meta


# ------------------------------------------------------------------ comandos
def executa_previsao(cfg, versao=None, hoje=None, baixar=False, consulta_api=None, config_arquivo=None):
    """Fluxo completo. Devolve (status, log). Sempre grava log e uma saída com status (com ou sem probabilidades)."""
    versao = versao or cfg["modelo"]["oficial"]
    hoje = hoje or (dt.date.fromisoformat(cfg["pipeline"]["hoje"]) if cfg["pipeline"].get("hoje") else dt.date.today())
    ex = Execucao(cfg, {"acao": "previsao", "versao": versao, "hoje": str(hoje), "baixar": baixar}, config_arquivo)
    try:
        modelo, cartao = carrega_modelo(cfg, ex, versao)
        ingestao(cfg, ex, consulta_api, baixar)
        painel = le_painel(cfg, ex)
        painel, val = validacao(cfg, ex, painel)
        painel = alinhamento_temporal(cfg, ex, painel, val)
        vars_modelo = ["clorofila", "temperatura"] + (["salinidade"] if cartao.get("flags", {}).get("use_salinity") else [])
        em = alinhamento_espacial(cfg, ex, painel, hoje, vars_modelo)
        prev, status = forecast(cfg, ex, painel, em, modelo, cartao)
        pasta, meta = grava_saida(cfg, ex, status, prev)
        return status, ex.finaliza(status, {"saida": str(pasta), **meta.get("resumo", {})})
    except FalhaPipeline as e:
        pasta, _ = grava_saida(cfg, ex, e.status, None, e.motivo)
        return e.status, ex.finaliza(e.status, {"saida": str(pasta)}, erro=str(e))
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        pasta, _ = grava_saida(cfg, ex, "ERRO", None, f"{type(e).__name__}: {e}")
        return "ERRO", ex.finaliza("ERRO", {"saida": str(pasta)}, erro=tb)


def executa_treino(cfg, versao, config_arquivo=None):
    """Treina com o config atual e REGISTRA uma versão nova (recusa sobrescrever)."""
    ex = Execucao(cfg, {"acao": "treino", "versao": versao}, config_arquivo)
    try:
        try:
            pipeline.flags_de(cfg)                                     # recusa cedo os grupos sem sinal
        except pipeline.FeatureSemSinal as e:
            raise FalhaPipeline("ERRO_DADOS", str(e))
        if (Path(cfg["caminhos"]["modelos"]) / versao).exists():
            raise FalhaPipeline("ERRO_MODELO", f"versão '{versao}' já existe; modelos antigos não são sobrescritos")
        cfg["modelo"]["seed"] = int(cfg["pipeline"]["seeds"][0])
        painel = le_painel(cfg, ex)
        painel, _ = validacao(cfg, ex, painel)
        with ex.etapa("5_features") as reg:
            cubo = cubo_do_painel(painel)
            fis, nomes_fis = pipeline.fisica(cfg, cubo)
            linhas = pipeline.dataset(cfg, cubo, fis)
            feats = pipeline.features_de(cfg, nomes_fis)
            reg.update(linhas=int(len(linhas)), features=len(feats))
        ex.log["features"] = feats
        with ex.etapa("6_modelo") as reg:
            cfg_ens = cfg.get("ensemble", {})
            modelo, cartao, _ = pipeline.treina_e_avalia(cfg, linhas, feats, versao, pipeline.flags_de(cfg),
                                                         {"painel": ex.log["dados"]["painel"], "run_id": ex.run_id, "seed": cfg["modelo"]["seed"]},
                                                         ensemble=cfg_ens if cfg_ens.get("ativo") else None)
            man = registro.registra(modelo, cartao, cfg["caminhos"]["modelos"], versao)
            g = cartao["teste"]["geral"]["modelo"]
            reg.update(manifesto=man["arquivos"], teste_pr_auc=g["pr_auc"], limiar_alto=cartao["limiar_alto"])
        ex.log["modelo"] = {"versao": versao, "manifesto_sha256": man["arquivos"]}
        ex.log["periodo"] = {"split": cfg["split"]}
        return "OK", ex.finaliza("OK", {"versao": versao, "teste": cartao["teste"]["geral"]["modelo"]})
    except FalhaPipeline as e:
        return e.status, ex.finaliza(e.status, erro=str(e))
    except Exception:
        return "ERRO", ex.finaliza("ERRO", erro=traceback.format_exc(limit=5))


CODIGO_SAIDA = {"OK": 0, "ATRASADO": 0, "DESATUALIZADO": 10, "DADOS_INSUFICIENTES": 20, "ERRO_DADOS": 30, "ERRO_MODELO": 40, "ERRO": 50}
