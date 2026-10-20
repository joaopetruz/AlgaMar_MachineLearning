# _arquivo

Coisas que **não** entram no resultado final (`python algamar.py previsao`) nem no treino (`python algamar.py treinar`).
Foram movidas da raiz em 2026-10-19 só para organizar; nada foi apagado. Para usar algo daqui, copie de volta para a raiz
(os scripts esperam rodar a partir dela).

| pasta | o que é |
|---|---|
| `scripts_experimentos/` | etapas 13.x/14.1, piloto CROCO, GLORYS, comparações com física, fechamento e diagnósticos |
| `scripts_equipe_exploracao/` | scripts originais da equipe (exploração, downloads antigos, modelos anteriores ao sazonal) |
| `scripts_substituidos/` | `treinar_previsao.py`, `prever_12_meses.py`, `pipeline_operacional.py`: substituídos por `algamar.py` |
| `docs/` | relatórios das etapas; o relatório final continua em `docs/` na raiz |
| `resultados/` | tabelas, métricas e logs dos experimentos |
| `baselines/` | métricas congeladas do baseline-hist-api-v1 |
| `previsoes/` | previsões legadas de `prever_12_meses.py` / `pipeline_operacional.py` |
| `saidas/` | execuções antigas do `algamar.py` (a apontada por `saidas/ULTIMA.json` ficou na raiz) |
| `modelos/` | modelos de teste/experimento (`teste-arquitetura*`, `e0-forecast-v1`) |
| `imagens/`, `paginas/` | gráficos PNG antigos e páginas HTML de teste |
