# Etapa 13.8 — Pipeline final organizado e reproduzível

**Data:** 2026-10-09. **Ramo local:** `algamar/correcao-leakage`. Nada foi enviado ao GitHub.

**Classificação: PRONTO PARA HINDCAST** (ver §9).

---

## 1. Pipeline único

Um ponto de entrada, `algamar.py`, e um módulo de fluxo, `algamar_ml/fluxo.py`:

```
python algamar.py previsao                       # fluxo completo, modelo oficial, data de hoje
python algamar.py previsao --hoje 2026-10-09     # data fixa (reprodução)
python algamar.py previsao --versao X --baixar   # outro modelo registrado; baixa meses novos antes
python algamar.py treinar --versao NOVA          # treina e registra uma versão NOVA
python algamar.py modelos                        # lista o registro
```

As etapas, cada uma registrada no log com status e duração:

| # | etapa | o que faz |
|---|---|---|
| 1 | ingestão | consulta o catálogo Copernicus de CHL, SST e SSS (MY e NRT). Com `--baixar`, atualiza os brutos e o painel. |
| 2 | validação | painel: colunas, coordenadas, chave única e ordem. Brutos diários: abre o arquivo e confere fuso, duplicatas, ordem, mês e cobertura. |
| 3 | alinhamento temporal | mês civil UTC. Uma variável com arquivo bruto FALHA vira NaN naquele mês e nunca é preenchida. |
| 4 | alinhamento espacial | grade CHL 4 km (1 790 pontos), validação do mês e escolha de t0 com fallback (13.4). |
| 5 | feature engineering | features do cartão do modelo em t0. O CROCO/GLORYS entra só se o modelo exigir. A proveniência por feature e a prova de "nada depois de t0" ficam no log. |
| 6 | modelo | modelo do registro (SHA256 conferido) e limiar do cartão (escolhido na validação). |
| 7 | forecast H1–H12 | 12 horizontes por ponto. |
| 8 | mapa/risco | previsão por ponto, risco por região e mês-alvo, e JSON de metadados. |

**CROCO/Navier–Stokes:** entra pelas flags `use_croco_*` (13.5) ou `use_glorys_*` (13.7). O modelo oficial não usa física, porque a 13.7 mostrou ganho de só +0,003. O pipeline registra "CROCO não exigido pelo modelo".

**Execução real** (`--hoje 2026-10-09`, 1 min):
- status **OK**, t0 = 2026-09, 21 480 previsões, 0 pontos insuficientes;
- probabilidades iguais às do pipeline 13.4 (diferença máxima 1e-4, que é o arredondamento a 4 casas) e níveis de risco 100 % iguais.

## 2. Configuração centralizada (`config/ml.toml`)

| item | onde |
|---|---|
| paths | `[caminhos]` (dados, painel, brutos, CROCO, GLORYS, modelos) e `[pipeline]` saidas, logs |
| features | `[features]`: flags históricas, APIs e salinidade |
| flags CROCO | `[features]`: `use_croco_ts/core/vertical/current/dynamics`, `use_glorys_ts/mld` |
| model_version | `[modelo].oficial = "baseline-hist-api-v1"` |
| threshold | `[pipeline].limiar = "cartao"`: usa o limiar do cartão, escolhido na validação, nunca no teste. Aceita um número fixo. |
| horizons | `[previsao].horizontes = 1..12` |
| APIs | `[pipeline].apis` (produtos em `algamar_ml/apis/copernicus.py`) |
| seeds | `[pipeline].seeds`; a 1ª vira `modelo.seed` no treino |
| datas | `[split]` e `[pipeline]`: `inicio_emissoes`, `hoje`, `max_atraso_meses`, `max_recuo_meses` |
| regras de dado insuficiente | `[pipeline]`: `features_obrigatorias_t0`, `frac_max_pontos_insuficientes` |
| rótulo do produto | `[pipeline].classificacao_produto = "HINDCAST_VALIDADO"` |

## 3. Logs

Cada execução grava `logs/execucoes/<run_id>.json` e uma linha em `logs/execucoes/execucoes.jsonl`, com:
- timestamp de início e fim, comando e commit git;
- config (arquivo e SHA256);
- **dados**: painel (arquivo, SHA256, linhas), brutos validados, CROCO;
- **modelo**: versão, SHA256 do manifesto e limiares;
- **features** usadas;
- **período**: hoje, t0 e atraso;
- etapas (status, duração, detalhes), avisos e **erros** (com traceback quando inesperados);
- **resultado**: pasta de saída, linhas, pontos insuficientes, risco por horizonte.

## 4. Modelo versionado (`algamar_ml/registro.py`)

- `modelos/<versao>/` contém `modelo.json`, `cartao.json` e `MANIFESTO.json` (SHA256).
- **Nada é sobrescrito.**
  - `registra()` recusa uma versão existente.
  - `treino.salva()` passou a recusar sobrescrita.
  - `algamar.py treinar` com versão existente → ERRO_MODELO.
- **Conferência na leitura:** antes de usar, o SHA256 é conferido. Um arquivo alterado ou corrompido gera ERRO_MODELO.
- **Versões antigas:** ganharam `MANIFESTO.json` na 1ª leitura, sem alterar os arquivos do modelo.

## 5. Saída padrão (contrato `contrato/saida_previsao_v1.json`)

`saidas/<run_id>/`:
- **`previsao.parquet`**, com as colunas:
  - run_id, prediction_date, target_date, horizon, point_id, latitude, longitude, regiao;
  - **probabilidade** (NULO se faltam dados), probabilidade_climatologica, **classe**, **nivel_de_risco** (BAIXO/MODERADO/ALTO/DADOS_INSUFICIENTES), status_ponto;
  - modelo, **versao**, status_execucao, classificacao_produto.
- **`risco_regioes.parquet`**: o mapa de risco agregado, por região × mês-alvo × horizonte.
- **`previsao.json`**: schema, status, motivo, modelo, período, SHA256 dos arquivos e resumo.

`saidas/ULTIMA.json` aponta a execução mais recente, para o dashboard/API.

## 6. Testes de ponta a ponta (`tests/test_pipeline_final.py`, 16 testes)

Rodam em pasta temporária com painel sintético, treino real e registro real, sem acessar nenhuma API. A suíte completa tem **106 testes passando**.

| teste | verifica |
|---|---|
| fluxo completo | saída com as 17 colunas, 12 horizontes, ULTIMA.json, SHA256, log com todos os campos |
| reprodutível | duas execuções dão probabilidades bit a bit iguais |
| não sobrescreve | treinar de novo a mesma versão → ERRO_MODELO, arquivo intacto |
| modelo alterado | SHA256 diferente → ERRO_MODELO, sem previsão |
| API indisponível | segue com os dados locais, com aviso |
| CROCO indisponível | modelo que exige CROCO sem a tabela → DADOS_INSUFICIENTES |
| feature ausente | → DADOS_INSUFICIENTES |
| painel corrompido | → ERRO_DADOS |
| bruto corrompido | variável anulada no mês, t0 recua → ATRASADO |
| fora de ordem | reordena, avisa e dá o mesmo resultado |
| chave repetida | → ERRO_DADOS |
| coordenada inválida | linhas descartadas e contadas |
| ponto sem estado em t0 | probabilidade e classe NULAS, nível DADOS_INSUFICIENTES |
| sem meses válidos / dados velhos demais | → DADOS_INSUFICIENTES, **sem previsao.parquet** |

## 7. Modos de falha

| falha | comportamento | status (código de saída) |
|---|---|---|
| API indisponível | registra INDISPONIVEL e segue com dados locais; t0 recua até 3 meses | OK, ATRASADO (0) ou DESATUALIZADO (10); sem mês válido → DADOS_INSUFICIENTES (20) |
| CROCO indisponível | modelo exige física → para; modelo não exige → registra e segue | DADOS_INSUFICIENTES (20) / OK |
| feature ausente | coluna exigida ausente → para; estado obrigatório em t0 ausente num ponto → só esse ponto fica sem probabilidade; > 50 % dos pontos → para | DADOS_INSUFICIENTES (20) |
| arquivo corrompido | painel → para; bruto diário → variável anulada no mês, com fallback de t0 | ERRO_DADOS (30) / ATRASADO |
| data fora de ordem | painel reordenado com aviso; chave repetida → para; tempos diários fora de ordem → arquivo FALHA | OK / ERRO_DADOS (30) |
| coordenada inválida | NaN, fora da caixa do ML ou mês fora de 1–12 → linha descartada e contada | OK, com aviso |
| modelo alterado ou ausente | SHA256 diferente ou versão inexistente → para | ERRO_MODELO (40) |
| erro inesperado | traceback no log, saída só com o status | ERRO (50) |

## 8. Não inventar previsão

- **Pontos sem estado:** um ponto sem `chl_media_t0`, `chl_max_t0` ou `sst_t0` recebe **probabilidade e classe nulas** e nível DADOS_INSUFICIENTES. Até a 13.4, o XGBoost recebia NaN e devolvia uma probabilidade; isso foi corrigido.
- **Execuções bloqueadas:** sem mês válido, com feature exigida ausente ou com > 50 % dos pontos sem estado, **não há `previsao.parquet`**. Só o `previsao.json` é gravado, com o status e o motivo.

**Correção de reprodutibilidade:** o `.gitignore` tinha `modelos/`, que ignorava também o código `algamar_ml/modelos/` (treino). Agora os padrões são ancorados (`/modelos/`) e o código do treino entra no git.

## 9. Classificação

| nível | situação | por quê |
|---|---|---|
| PRONTO PARA PESQUISA | ✅ | config única, flags de física, experimentos reprodutíveis e versionados |
| **PRONTO PARA HINDCAST** | ✅ **(classificação atual)** | fluxo completo, logado e testado de ponta a ponta. Modelo imutável com hash. Saída padrão. Falhas tratadas sem inventar previsão. Validado em hindcast 2022–2026 (13.7). |
| PRONTO PARA PILOTO OPERACIONAL | ❌ ainda não | ver abaixo |
| PRONTO PARA PRODUÇÃO | ❌ | além do piloto: monitoramento, alertas, disponibilidade, recalibração, revisão humana e habilidade comprovada sobre a climatologia |

**O que falta para o piloto operacional:**
1. **Agendamento:** rodar `algamar.py previsao --baixar` todo mês, por exemplo pelo Agendador do Windows ou cron. Hoje é manual.
2. **Inputs NRT validados:** o hindcast usou CHL MY e SST REP; em tempo real entram NRT. A SSS NRT já foi desligada por viés (13.4).
3. **Previsões congeladas e verificadas:** arquivar cada `saidas/<run_id>` e comparar com o observado quando o mês-alvo chegar, por alguns meses.
4. **Recalibração:** as probabilidades são levemente subconfiantes (13.7).
5. **Rótulo honesto ao usuário:** o modelo é ≈ climatologia (13.7). O produto deve ser apresentado como "risco climatológico ajustado pelo estado atual".
