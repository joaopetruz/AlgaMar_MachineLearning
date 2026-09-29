# Levantamento técnico do Machine Learning do Algamar

## 0. Escopo e método

Este documento é um inventário técnico do workspace `/home/joao_rufo/ML-Algamar/AlgaMar_MachineLearning`. Ele descreve o que foi encontrado em código, artefatos e dados locais; não é a documentação acadêmica final e não cobre o Backend/PostgreSQL nem o Frontend.

As afirmações foram confrontadas com:

- código Python e seus caminhos de entrada/saída;
- cabeçalhos, contagens, tipos, valores ausentes, faixas numéricas e períodos dos CSVs;
- metadados observáveis dos artefatos Joblib;
- formato/tamanho dos arquivos NetCDF;
- `requirements.txt`, `venv/pyvenv.cfg`, `.gitignore`, `index.html` e configuração do VS Code.

Quando uma informação não está comprovada, ela é marcada como **NÃO IDENTIFICADO NO CÓDIGO** ou **NECESSITA CONFIRMAÇÃO DO DESENVOLVEDOR**.

Os CSVs, NetCDFs e PKLs não estão versionados pelo Git neste workspace: `.gitignore` contém `*.pkl`, `*.csv`, `*.txt` e `*.nc`, com exceção de `requirements.txt`. Portanto, a existência local desses artefatos não equivale a rastreabilidade histórica no repositório.

## 1. Síntese executiva verificável

O workspace contém dois fluxos de ML:

1. um fluxo diário de 2024 com Random Forest, defasagens e médias móveis, implementado em `feature_engineering.py`, `treinar_modelo.py` e `salvar_modelo.py`;
2. um fluxo sazonal mensal de 2000–2024, com salinidade e históricos, implementado em `feature_engineering_sazonal.py`, `treinar_modelo_sazonal.py` e `salvar_modelo_sazonal.py`.

A API, porém, não recria features nem chama serviços externos durante a previsão. Em `api.py`, ela carrega na inicialização quatro artefatos locais (`modelo_floracao_sazonal.pkl`, `features_modelo_sazonal.pkl`, `limiar_decisao_sazonal.pkl` e `versao_modelo.pkl`) e o CSV `dados/dados_features_sazonal_sp_com_salinidade.csv`.

O conteúdo efetivamente inspecionado dos artefatos da API é:

| Artefato carregado pela API | Conteúdo observado |
|---|---|
| `modelo_floracao_sazonal.pkl` | `sklearn.ensemble.RandomForestClassifier`, 300 árvores, `min_samples_leaf=2`, `class_weight="balanced_subsample"`, `random_state=42`, 9 features; não é XGBoost. |
| `features_modelo_sazonal.pkl` | Lista de 9 features; não contém as três features de salinidade presentes no script sazonal mais recente. |
| `limiar_decisao_sazonal.pkl` | `0.2`. |
| `versao_modelo.pkl` | `3.0.0-sazonal-25anos`. |

Isso diverge dos scripts `treinar_modelo_sazonal.py` e `salvar_modelo_sazonal.py`, que usam XGBoost, 12 features, limiar 0,5 e gravam a versão `7.0.0-xgboost-sazonal-salinidade`. A origem dessa divergência **NÃO FOI IDENTIFICADA NO CÓDIGO**.

## 2. Estrutura real do workspace

### Arquivos de aplicação e pipeline

| Caminho | Responsabilidade observada |
|---|---|
| `api.py` | Aplicação FastAPI, carga de artefatos, endpoints ambientais, climatologia e previsão. |
| `organizar_dados.py` | Alinha temperatura à grade da clorofila, converte Kelvin para Celsius, interpola e gera dados diários tratados de 2024. |
| `processar_dados_historicos.py` | Lê clorofila, temperatura, salinidade e precipitação para 2000–2024, agrega mensalmente e gera CSV histórico com precipitação. |
| `feature_engineering.py` | Cria features diárias, lags, médias móveis, distância a hotspots e target para 2024. |
| `feature_engineering_sazonal.py` | Cria features mensais sazonais, históricos, janelas de 3 anos, distância e target. |
| `treinar_modelo.py` | Treina/avalia Random Forest diário com corte temporal em novembro de 2024. |
| `treinar_modelo_sazonal.py` | Treina/avalia XGBoost sazonal com corte por ano até 2021/>2021. |
| `salvar_modelo.py` | Treina e salva o Random Forest diário de produção. |
| `salvar_modelo_sazonal.py` | Treina e salva o XGBoost sazonal descrito pelo script. |
| `ajuste_hiperparametros.py` | GridSearchCV de Random Forest no dataset sazonal completo. |
| `selecao_features_rfe.py` | RFECV de Random Forest para seleção de features no dataset completo. |
| `comparar_xgboost.py` | Compara três cenários XGBoost e imprime um resumo textual de Random Forest. |
| `integrar_eventos_reais.py` | Junta dois CSVs de eventos, normaliza coordenadas, cruza por ano/mês/proximidade e cria marcação de eventos reais. |
| `validar_com_eventos_reais.py` | Aplica o artefato sazonal a registros com `floracao_confirmada_real=1`. |
| `baixar_dados_historicos.py` | Baixa clorofila e temperatura anuais do Copernicus para 2000–2024. |
| `baixar_dados_completos.py` | Baixa as duas variáveis para 2024 em regiões maiores. |
| `baixar_clorofila_faltante.py` | Baixa clorofila de anos indicados manualmente, atualmente `[2000]`. |
| `baixar_temperatura.py` | Download de teste de temperatura de 01 a 07/01/2024. |
| `baixar_salinidade.py` | Baixa salinidade superficial (`sos`) de 2000–2024. |
| `baixar_precipitacao.py` | Consulta o arquivo histórico Open-Meteo por lotes e gera precipitação mensal. |
| `teste_conexao.py` | Teste de download de clorofila do Copernicus. |
| `explorar_dados.py` | Inspeção inicial do NetCDF de teste e geração de mapa de clorofila. |
| `analise_exploratoria.py` | Estatística, série temporal, dispersão e correlação em dados tratados de 2024. |
| `testar_configuracoes.py` | Testa limites de target e limiares de decisão no fluxo diário. |
| `testar_limiares.py` | Conta casos acima de vários limiares de clorofila. |
| `diagnosticar_merge.py` | Verifica coincidência de coordenadas entre precipitação e base sazonal. |
| `mapa_hotspots.py` | Agrega clorofila por local e gera mapa de hotspots. |
| `investigar_outlier.py` | Lista valores altos de clorofila e casos acima de 30 mg/m³. |
| `index.html` | Página local que consome `/climatologia` e `/predict`; não é documentada como Frontend neste levantamento. |
| `requirements.txt` | Faixas declaradas de dependências. |
| `venv/` | Ambiente local; não deve ser tratado como fonte de dependências reprodutível por estar ignorado. |

### Diretório `dados/`

Contém CSVs intermediários/finais, duas famílias de NetCDFs ambientais, CSVs de eventos e cópias dos artefatos de modelo. Os arquivos locais possuem aproximadamente 4,2 GB no workspace completo; os dois maiores artefatos são os modelos sazonais, com 1.147.274.393 bytes cada.

Não foram encontrados `README`, `pyproject.toml`, YAML, JSON de configuração, notebooks (`.ipynb`), testes automatizados, scripts de deploy ou arquivos `.env` no inventário.

## 3. Tecnologias e ambiente

### Dependências declaradas

| Tecnologia | Versão/faixa declarada | Finalidade |
|---|---|---|
| Python | **NÃO IDENTIFICADO NO `requirements.txt`**; `venv/pyvenv.cfg` registra 3.12.3 | Execução. |
| FastAPI | `>=0.100,<1` | API HTTP. |
| Uvicorn | `>=0.23,<1` | Servidor ASGI. |
| Pydantic | `>=2,<3` | Validação do body. |
| joblib | `>=1.3,<2` | Serialização/carga do modelo e configurações. |
| matplotlib | `>=3.7,<4` | Gráficos/mapas exploratórios. |
| numpy | `>=1.24,<3` | Operações numéricas e distâncias. |
| pandas | `>=2,<3` | CSV, agregações, features e tabelas. |
| scikit-learn | `>=1.3,<2` | Random Forest, métricas, GridSearchCV e RFECV. |
| xarray | `>=2023.1,<2027` | Leitura/interpolação/agregação de NetCDF. |
| netCDF4 | `>=1.6,<2` | Backend científico declarado para NetCDF. |
| copernicusmarine | `>=1.2,<2` | Downloads do Copernicus Marine. |
| `requests` | **NÃO DECLARADA** | Usada por `baixar_precipitacao.py` para Open-Meteo. |
| `xgboost` | **NÃO DECLARADA** | Importada por três scripts de treino/comparação. |

No ambiente `venv` local observado, foram reportadas: Python 3.12.3 no arquivo de configuração; FastAPI 0.141.1; Uvicorn 0.52.4; Pydantic 2.13.5; joblib 1.6.0; numpy 2.5.3; pandas 3.0.5; scikit-learn 1.9.1. `xarray`, `netCDF4`, `copernicusmarine`, `xgboost`, `requests` e `matplotlib` não estavam instalados nesse ambiente no momento da inspeção. O pandas 3.0.5 também não atende à faixa `<3` declarada.

Não há comando de instalação ou comando Uvicorn documentado em README, Makefile ou script. **EXECUÇÃO PADRONIZADA NÃO IDENTIFICADA NO WORKSPACE.** A referência mais concreta é o `index.html`, que espera `http://127.0.0.1:8000`.

## 4. Fontes de dados e aquisição

### Copernicus Marine

Os scripts usam `copernicusmarine.subset`:

| Dado | Dataset ID | Variável | Região principal | Período |
|---|---|---|---|---|
| Clorofila-a | `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D` | `CHL` | longitude -47.0 a -44.8; latitude -25.3 a -23.3 | 01/01–31/12 de cada ano 2000–2024 |
| Temperatura superficial | `METOFFICE-GLO-SST-L4-REP-OBS-SST` | `analysed_sst` | longitude -47.5 a -44.3; latitude -25.8 a -22.8 | 01/01–31/12 de cada ano 2000–2024 |
| Salinidade superficial | `cmems_obs-mob_glo_phy-sss_my_multi_P1D` em `baixar_salinidade.py` | `sos` | longitude -47.5 a -44.3; latitude -25.8 a -22.8 | 01/01–31/12 de cada ano 2000–2024 |

Os arquivos diários locais de clorofila e temperatura existem para todos os anos de 2000 a 2024, com nomes `clorofila_<ano>_sp.nc` e `temperatura_<ano>_sp.nc`. Não existem localmente arquivos `salinidade_<ano>_sp.nc`, embora `processar_dados_historicos.py` dependa deles. Não foi identificado mecanismo de autenticação, variável de ambiente ou arquivo de credenciais; a autenticação do cliente Copernicus, se necessária, fica **NÃO IDENTIFICADA NO CÓDIGO**.

Os NetCDFs são arquivos HDF5/NetCDF. As strings internas do arquivo de clorofila expõem metadados CF/ACDD, `latitude`, `longitude`, `time`, `CHL`, referência CMEMS e a natureza de concentração de clorofila-a. A inspeção estrutural por `xarray`/`netCDF4` não pôde ser executada no `venv` local porque esses pacotes não estavam instalados; dimensões, unidades e resolução exatas devem ser confirmadas em ambiente com os backends declarados.

### Open-Meteo

`baixar_precipitacao.py` consulta `https://archive-api.open-meteo.com/v1/archive` com:

- datas `2000-01-01` a `2024-12-31`;
- `daily=precipitation_sum`;
- timezone `America/Sao_Paulo`;
- lotes de 50 localizações;
- timeout HTTP de 60 segundos, até três tentativas quando a resposta vem vazia e espera de 4 segundos entre lotes.

O resultado diário é convertido para ano/mês e somado por coordenada, ano e mês. O script grava `dados/precipitacao_2000_2024_sp.csv`. Esse arquivo não existe no workspace atual.

### Eventos reais

`dados/habdata_algas.csv` tem 21 registros e `dados/habdata_sao_paulo.csv` tem 26. O primeiro possui mês, data, coordenadas, tipo de alga, efeitos e fonte; o segundo não possui coluna de mês. `integrar_eventos_reais.py` concatena ambos, corrige latitudes positivas invertendo o sinal, remove coordenadas ausentes, restringe eventos a 2016–2024 e descarta eventos sem mês reconhecido.

O cruzamento não usa distância geográfica real: calcula distância euclidiana em graus entre o evento e os pontos da base no mesmo ano e mês, escolhendo o menor ponto. O resultado é `floracao_confirmada_real`, uma marcação auxiliar; não substitui o target `floracao` usado pelo modelo.

## 5. Inventário dos datasets

As contagens abaixo são de registros de dados, descontando a linha de cabeçalho. Os CSVs foram processados em blocos para obter metadados sem carregar cada arquivo inteiro.

| Dataset | Registros | Colunas | Período observado | Coordenadas aproximadas | Nulos observados |
|---|---:|---:|---|---|---:|
| `dados_tratados_2024.csv` | 1.498.404 | 5 | 2024-01-01 a 2024-12-31 | 55 latitudes / 96 longitudes | 0 |
| `dados_tratados_2024_sp.csv` | 667.584 | 5 | 2024-01-01 a 2024-12-31 | 46 / 53 | 0 |
| `dados_features_2024.csv` | 1.469.746 | 17 | 2024-01-08 a 2024-12-31 | 55 / 96 | 0 |
| `dados_features_2024_sp.csv` | 654.816 | 17 | 2024-01-08 a 2024-12-31 | 46 / 53 | 0 |
| `dados_mensais_2000_2024_sp_com_salinidade.csv` | 523.425 | 8 | 2000–2024 | 45 / 53 | 0 |
| `dados_mensais_2000_2024_sp_completo.csv` | 520.735 | 14 | 2000–2024 | 45 / 53 | 0 |
| `dados_features_sazonal_sp.csv` | 525.312 | 14 | 2001–2024 | 46 / 53 | 0 |
| `dados_features_sazonal_sp_com_salinidade.csv` | 501.945 | 18 | 2001–2024 | 45 / 53 | 0 |
| `dados_features_sazonal_sp_com_eventos_reais.csv` | 501.945 | 19 | 2001–2024 | 45 / 53 | 0 |
| `dados_features_sazonal_sp_completo.csv` | 499.435 | 42 | 2001–2024 | 45 / 53 | 0 |
| `habdata_algas.csv` | 21 | 14 | 1920–2024 | eventos | campos ausentes conforme origem |
| `habdata_sao_paulo.csv` | 26 | 10 | **NÃO CALCULADO NA INSPEÇÃO** | eventos | 0 |

### Faixas observadas

- `dados_features_2024_sp.csv`: clorofila 0,03977316–42,95329; temperatura 16,9481–29,695 °C; distância a hotspot 0,0042459–2,00753; target 0: 596.702 e 1: 58.114.
- `dados_features_2024.csv`: clorofila 0,03640427–59,897423; temperatura 17,0694–31,3216 °C; distância 0,005893–3,511; target 0: 1.460.920 e 1: 8.826.
- `dados_features_sazonal_sp_com_salinidade.csv`: clorofila média 0,06526024–12,878313; máxima 0,07756001–52,582188; temperatura média 17,3023–29,8458 °C; salinidade média 32,8011–36,9426; distância 0,082609–2,00753; target 0: 406.340 e 1: 95.605.
- No dataset sazonal com eventos, `floracao_confirmada_real` também é binário; a contagem exata por valor não foi separada nesta inspeção, mas não há nulos.

Os datasets sazonais derivados não cobrem 2000 na saída porque a engenharia usa `shift(12)` e remove linhas sem ano anterior. A origem exata da diferença de quantidade de pontos entre datasets (45/46 latitudes, 53 longitudes) deve ser confirmada com os arquivos de aquisição e a grade efetivamente retornada pelo Copernicus.

## 6. Tratamento e transformação dos dados

### Fluxo diário (`organizar_dados.py`)

**Entrada:** `dados/clorofila_2024_sp.nc` e `dados/temperatura_2024_sp.nc`.

**Processamento:**

1. abre os dois NetCDFs com `xarray`;
2. interpola temperatura para as coordenadas de latitude/longitude da clorofila;
3. faz `xr.merge`;
4. converte `analysed_sst` de Kelvin para Celsius subtraindo 273,15;
5. converte para DataFrame e renomeia `CHL` para `clorofila` e `analysed_sst` para `temperatura_celsius`;
6. ordena por latitude, longitude e tempo;
7. interpola linearmente clorofila e temperatura dentro de cada ponto, com `limit_direction="both"`;
8. remove pontos cuja série de clorofila permaneça integralmente ausente;
9. remove linhas ainda sem temperatura;
10. grava `dados/dados_tratados_2024_sp.csv`.

Não há conversão de unidade explícita para clorofila; o código mantém o valor `CHL` e os gráficos usam mg/m³. O script paralelo `analise_exploratoria.py` aponta para `dados_tratados_2024.csv`, enquanto o pipeline SP usa `dados_tratados_2024_sp.csv`; ambos existem, mas sua geração não é idêntica no workspace.

### Fluxo histórico mensal (`processar_dados_historicos.py`)

**Entrada:** os NetCDFs anuais de clorofila, temperatura e salinidade e `dados/precipitacao_2000_2024_sp.csv`.

**Processamento:**

- abre os três dados por ano de 2000 a 2024;
- interpola temperatura e salinidade para a grade da clorofila;
- converte temperatura de Kelvin para Celsius;
- renomeia `CHL`, `analysed_sst` e `sos` para `clorofila`, `temperatura_celsius` e `salinidade`;
- retém locais que não têm clorofila ausente em 100% da série;
- cria `ano` e `mes` a partir de `time`;
- agrega por latitude, longitude, ano e mês usando média de clorofila, máximo de clorofila, média de temperatura e média de salinidade;
- arredonda coordenadas para cinco casas e faz `left merge` com precipitação;
- concatena anos e aplica `dropna()` em todas as colunas;
- grava `dados_mensais_2000_2024_sp_precipitacao.csv`.

Esse arquivo de saída e os NetCDFs de salinidade/precipitação de entrada não estão presentes atualmente. Os CSVs mensais com salinidade existentes demonstram que uma etapa equivalente foi executada em algum momento, mas a execução exata não pode ser reproduzida com os arquivos atuais.

## 7. Engenharia de atributos

### Fluxo diário

Em `feature_engineering.py`, depois da ordenação por local e tempo:

| Feature | Origem/transformação | Uso confirmado |
|---|---|---|
| `mes` | `time.dt.month` | Treino diário. |
| `dia_do_ano` | `time.dt.dayofyear` | Treino diário. |
| `estacao` | mapeamento fixo de mês para verão/outono/inverno/primavera | Gerada no CSV, não incluída na lista de features de `treinar_modelo.py`. |
| `clorofila_lag1` | `groupby(latitude,longitude).shift(1)` | Treino diário. |
| `clorofila_lag3` | shift de 3 registros no grupo | Treino diário. |
| `clorofila_lag7` | shift de 7 registros no grupo | Treino diário. |
| `temperatura_lag1` | shift de 1 registro no grupo | Treino diário. |
| `clorofila_media_7d` | rolling 7, `min_periods=1` | Treino diário. |
| `clorofila_media_14d` | rolling 14, `min_periods=1` | Treino diário. |
| `temperatura_media_7d` | rolling 7, `min_periods=1` | Treino diário. |
| `dist_hotspot` | mínimo de duas distâncias euclidianas em graus | Treino diário. |
| `latitude`, `longitude` | coordenadas originais | Treino diário. |

As lags são por número de registros ordenados, não por uma verificação explícita de intervalo diário. A implementação não usa fórmula geodésica, raio terrestre nem conversão de graus para quilômetros.

### Fluxo sazonal

Em `feature_engineering_sazonal.py`, sobre `dados_mensais_2000_2024_sp_com_salinidade.csv`:

| Feature | Construção confirmada | Usada no artefato carregado pela API? |
|---|---|---|
| `mes` | coluna mensal de entrada | Sim |
| `latitude`, `longitude` | coordenadas de entrada | Sim |
| `dist_hotspot` | menor de duas distâncias euclidianas em graus | Sim |
| `clorofila_media_ano_anterior` | `groupby(local).shift(12)` | Sim |
| `clorofila_maxima_ano_anterior` | `shift(12)` | Sim |
| `temperatura_ano_anterior` | `shift(12)` | Sim |
| `salinidade_ano_anterior` | `shift(12)` | Não; presente no script XGBoost e no body da API, mas ausente na lista serializada. |
| `clorofila_media_historica_mes` | média por local e mês em todos os anos disponíveis, incluindo a própria observação | Sim |
| `salinidade_media_historica_mes` | média por local e mês | Não na lista serializada; é retornada por `/climatologia`. |
| `clorofila_media_3anos` | rolling 36 com `min_periods=12` por local | Sim |
| `salinidade_media_3anos` | rolling 36 com `min_periods=12` | Não na lista serializada. |

O script sazonal remove linhas sem as três variáveis do ano anterior (`clorofila_media_ano_anterior`, `temperatura_ano_anterior`, `salinidade_ano_anterior`). A média histórica mensal é calculada diretamente com `transform("mean")`; não há exclusão explícita do ano corrente.

O dataset sazonal completo possui variáveis adicionais (`vento_velocidade`, `corrente_velocidade`, `nitrato`, `fosfato`, `oxigenio`, `ph`) e agregados/defasagens delas. Elas são candidatas em RFE/GridSearch e em comparações, mas não estão na lista efetiva serializada carregada pela API.

## 8. Variável-alvo

Há dois proxies de target:

- diário: em `feature_engineering.py`, `floracao = (clorofila > 1.5).astype(int)`;
- sazonal: em `feature_engineering_sazonal.py`, `floracao = (clorofila_maxima > 1.5).astype(int)`.

Assim, `0` significa que a condição do proxy não foi atingida e `1` significa que foi atingida. O código de treino imprime os nomes `Sem floração` e `Com floração`. O target não é uma medição toxicológica direta; a correspondência acadêmica entre 1,5 e floração tóxica **NECESSITA CONFIRMAÇÃO DO DESENVOLVEDOR**.

`floracao_confirmada_real` é uma coluna independente criada no cruzamento com eventos reais. Não há script que substitua `y` por essa coluna para treinar novamente o artefato da API.

## 9. Treinamento e modelos

### Random Forest diário

Evidência: `treinar_modelo.py`, linhas 6–70, e `salvar_modelo.py`, linhas 5–50.

- dataset de avaliação: `dados_features_2024_sp.csv`;
- 13 features listadas no código;
- target: `floracao`;
- divisão temporal: treino antes de `2024-11-01`, teste a partir de `2024-11-01`;
- classe: `RandomForestClassifier` do scikit-learn;
- `n_estimators=300`, `class_weight="balanced_subsample"`, `min_samples_leaf=2`, `random_state=42`, `n_jobs=-1`;
- produz probabilidades por `predict_proba(X)[:,1]`;
- o script testa limiares 0,5; 0,3; 0,2; 0,1; 0,05 e escolhe 0,2;
- `salvar_modelo.py` treina o artefato diário usando todos os dados de `dados_features_2024.csv`, não o arquivo SP usado no treino/avaliação;
- artefatos: `modelo_floracao.pkl`, `features_modelo.pkl` e `limiar_decisao.pkl`.

O artefato `dados/modelo_floracao.pkl` foi inspecionado: é Random Forest com 300 estimadores, 13 features, as mesmas configurações acima e classes `[0,1]`. O arquivo tem 180.186.409 bytes. A cópia `modelo_floracao.pkl` na raiz **não foi encontrada**; `salvar_modelo.py` grava na raiz, mas o artefato disponível está em `dados/`.

### Fluxo sazonal descrito pelos scripts

Evidência: `treinar_modelo_sazonal.py` e `salvar_modelo_sazonal.py`.

- dataset: `dados_features_sazonal_sp_com_salinidade.csv`;
- 12 features, incluindo salinidade do ano anterior, climatologia mensal e médias de 3 anos;
- split: anos `<=2021` para treino e `>2021` para teste;
- classe declarada: `xgboost.XGBClassifier`;
- `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `scale_pos_weight=(negativos/positivos)`, `random_state=42`, `n_jobs=-1`, `eval_metric="logloss"`;
- testa limiares de 0,6 a 0,2 e escolhe 0,5;
- `salvar_modelo_sazonal.py` treina novamente em todos os dados e salva os artefatos sazonais.

### Artefato sazonal efetivamente disponível

As cópias `modelo_floracao_sazonal.pkl` e `dados/modelo_floracao_sazonal.pkl` têm o mesmo SHA-256 e foram carregadas com sucesso como `RandomForestClassifier`, não como XGBClassifier. Metadados observados:

- 300 árvores;
- `criterion="gini"`;
- `max_depth=None`;
- `min_samples_leaf=2`;
- `class_weight="balanced_subsample"`;
- `random_state=42`, `n_jobs=-1`;
- 9 features: `mes`, `latitude`, `longitude`, `dist_hotspot`, `clorofila_media_ano_anterior`, `clorofila_maxima_ano_anterior`, `temperatura_ano_anterior`, `clorofila_media_historica_mes`, `clorofila_media_3anos`;
- classes `[0,1]`;
- tamanho: 1.147.274.393 bytes.

Portanto, o algoritmo efetivo da API é identificado pelo artefato como Random Forest. A cadeia que produziu esse artefato **NÃO ESTÁ PRESENTE** ou não coincide com os scripts atuais.

## 10. Seleção e ajuste de features/hiperparâmetros

`selecao_features_rfe.py` usa `RFECV` com Random Forest de 100 árvores, `min_samples_leaf=5`, `class_weight="balanced_subsample"`, três folds estratificados embaralhados, `scoring="roc_auc"`, `step=1` e mínimo de cinco features. Opera sobre amostra de 100.000 linhas do dataset sazonal completo. O resultado seria salvo em `features_selecionadas_rfe.pkl`, mas esse arquivo não existe no workspace.

`ajuste_hiperparametros.py` usa `GridSearchCV` de Random Forest, `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`, ROC-AUC e a grade:

- `n_estimators`: 200 ou 300;
- `max_depth`: 10, 20 ou `None`;
- `min_samples_leaf`: 2 ou 5.

O resultado seria salvo em `melhores_hiperparametros.pkl`, também ausente. Não foi localizada aplicação desses resultados ao artefato efetivo da API.

## 11. Avaliação, métricas e resultados

Os scripts de treino calculam `f1_score`, `recall_score`, `classification_report` e `confusion_matrix`. Os scripts de seleção/ajuste usam ROC-AUC. O workspace não contém logs de execução, arquivos de métricas, matrizes ou relatórios persistidos; por isso, os valores efetivamente obtidos para o artefato em produção são **MÉTRICAS DE AVALIAÇÃO NÃO IDENTIFICADAS**.

`comparar_xgboost.py` contém três blocos que recalculam F1/recall para XGBoost em treino `ano<=2021` e teste `ano>2021`, com vários limiares. Ao final, há três números hard-coded rotulados como resumo de Random Forest (`F1=0.671`, `Recall=82.5%`; `F1=0.615`, `Recall=72.0%`; `F1=0.623`, `Recall=73.1%`). Como não há saída de execução nem fonte intermediária desses números, eles devem ser tratados apenas como texto registrado no script, e não como métricas verificadas do artefato atual.

`validar_com_eventos_reais.py` calcula uma taxa de acerto somente sobre registros marcados como eventos reais e imprime `acertos / total`. Não existe log salvo dessa execução. Além disso, a validação é apenas sobre positivos confirmados; não demonstra precisão, especificidade, matriz de confusão ou desempenho geral.

## 12. Threshold e classificação de risco

No runtime da API, `limiar_decisao_sazonal.pkl` contém `0.2`. Em `/predictions` e `/predict`, a regra é exatamente:

```text
probabilidade >= limiar  -> "ALTO"
caso contrário            -> "BAIXO"
```

O script `salvar_modelo_sazonal.py` atual gravaria 0,5, mas o arquivo carregado contém 0,2. A origem do arquivo 0,2 **NECESSITA CONFIRMAÇÃO DO DESENVOLVEDOR**.

O `index.html` ainda classifica visualmente a probabilidade em cinco níveis (`Muito baixo`, `Baixo`, `Moderado`, `Alto`, `Muito alto`) usando limites 0,05, 0,15, 0,35, 0,60 e 1,01. Essa é uma classificação da camada HTML, distinta da classificação binária `ALTO`/`BAIXO` retornada pela API.

## 13. Artefatos, versão e compatibilidade

| Caminho | Tamanho aproximado | Tipo/conteúdo |
|---|---:|---|
| `modelo_floracao_sazonal.pkl` | 1,07 GiB | Random Forest sazonal, 9 features. |
| `dados/modelo_floracao_sazonal.pkl` | 1,07 GiB | Cópia byte a byte do anterior. |
| `dados/modelo_floracao.pkl` | 172 MiB | Random Forest diário, 13 features. |
| `features_modelo_sazonal.pkl` e `dados/features_modelo_sazonal.pkl` | 206 bytes | Cópias idênticas da lista de 9 features. |
| `dados/features_modelo.pkl` | 231 bytes | Lista de 13 features do modelo diário. |
| `limiar_decisao_sazonal.pkl` e cópia em `dados/` | 21 bytes | Float `0.2`. |
| `versao_modelo.pkl` e cópia em `dados/` | 35 bytes | String `3.0.0-sazonal-25anos`. |
| `dados/limiar_decisao.pkl` | 21 bytes | Float `0.2`. |

Ao carregar os modelos no ambiente local, o scikit-learn emitiu `InconsistentVersionWarning`: os estimadores foram serializados com scikit-learn 1.9.0 e carregados com 1.9.1. O código de versão do modelo é uma string manual; não há mecanismo automático que associe versão do modelo, hash, dataset, Python e dependências.

## 14. Inferência e API FastAPI

### Inicialização

Evidência: `api.py`, linhas 7–23.

Na importação do módulo, a API:

1. cria `FastAPI(title="API de Previsão de Floração de Algas - AlgarMar (Litoral SP)")`;
2. instala CORS com `allow_origins=["*"]`, `allow_methods=["*"]` e `allow_headers=["*"]`;
3. carrega os artefatos Joblib da raiz;
4. carrega todo `dados/dados_features_sazonal_sp_com_salinidade.csv` em memória.

Não há autenticação, autorização, rate limit, segredo, header de segurança ou middleware de tratamento de exceção implementado no código observado.

### Tabela de endpoints

| Método | Endpoint | Entrada | Saída | Finalidade |
|---|---|---|---|---|
| GET | `/` | nenhuma | `{"mensagem": ...}` | Mensagem de funcionamento/documentação. |
| GET | `/health` | nenhuma | `{"status":"ok"}` | Health check superficial. |
| GET | `/marine-data` | query `limit: int = 100` | `{"data":[...]}` | Retorna as últimas linhas do CSV ambiental. |
| GET | `/climatologia` | query `latitude`, `longitude` float | ponto encontrado e `climatologia_por_mes` | Consulta climatologia mensal do ponto de grade mais próximo. |
| GET | `/predictions` | query `limit: int = 100` | `{"predictions":[...]}` | Calcula previsões para as últimas linhas do CSV. |
| POST | `/predict` | JSON validado por `DadosSazonais` | `probability`, `risk`, `model_version` | Classifica uma entrada manual. |

Não existe `/predictions` POST; o endpoint é GET. Não há endpoint de aquisição em tempo real no código da API.

### Modelo Pydantic `DadosSazonais`

Todos os campos são obrigatórios e tipados, sem defaults nem validadores customizados:

| Campo | Tipo | Uso no runtime |
|---|---|---|
| `mes` | `int` | Selecionado pelo vetor de features serializado. |
| `latitude` | `float` | Selecionado. |
| `longitude` | `float` | Selecionado. |
| `dist_hotspot` | `float` | Selecionado. |
| `clorofila_media_ano_anterior` | `float` | Selecionado. |
| `clorofila_maxima_ano_anterior` | `float` | Selecionado. |
| `temperatura_ano_anterior` | `float` | Selecionado. |
| `salinidade_ano_anterior` | `float` | Aceito no body, mas não selecionado pelo artefato atual. |
| `clorofila_media_historica_mes` | `float` | Selecionado. |
| `salinidade_media_historica_mes` | `float` | Aceito no body, mas não selecionado pelo artefato atual. |
| `clorofila_media_3anos` | `float` | Selecionado. |
| `salinidade_media_3anos` | `float` | Aceito no body, mas não selecionado pelo artefato atual. |

O body exige 12 campos, mas `features_modelo_sazonal.pkl` contém apenas 9. Os três campos de salinidade ainda são necessários para passar pela validação Pydantic, embora sejam descartados na construção efetiva de `entrada`. Essa inconsistência é verificável em `api.py`, linhas 26–38 e 145.

### `/marine-data`

`limit` não tem default de validação, mínimo ou máximo declarados. A função aplica `dados_ambientais.tail(limit)`. Cada item retorna:

```json
{
  "latitude": "float arredondado a 4",
  "longitude": "float arredondado a 4",
  "year": "int",
  "month": "int",
  "temperature_celsius": "float arredondado a 2",
  "chlorophyll_mg_m3": "float arredondado a 4",
  "chlorophyll_max_mg_m3": "float arredondado a 4",
  "salinity_psu": "float arredondado a 3",
  "source": "Copernicus Marine Service"
}
```

Apesar do campo `source`, a função não faz uma chamada Copernicus; ela lê o CSV previamente carregado. A quantidade de itens é determinada por `limit` e disponibilidade de linhas. Não há erro HTTP customizado.

### `/climatologia`

Calcula distância euclidiana em graus entre a coordenada solicitada e todas as linhas da base, escolhe o menor ponto, filtra todas as linhas daquela latitude/longitude e retorna até 12 meses. Cada mês contém `clorofila_media_historica_mes` e `salinidade_media_historica_mes`, arredondadas. O retorno também informa `latitude_encontrada` e `longitude_encontrada`.

Não há validação de limites geográficos, fallback se o CSV estiver vazio ou conversão para distância física. A função usa a primeira linha se houver mais de uma no mesmo mês.

### `/predictions`

Seleciona as últimas `limit` linhas, extrai as colunas do vetor `features`, calcula `predict_proba(X)[:,1]`, aplica o threshold carregado e retorna:

```json
{
  "predictions": [
    {
      "latitude": 0.0,
      "longitude": 0.0,
      "year": 0,
      "month": 0,
      "probability": 0.0,
      "risk_level": "ALTO ou BAIXO",
      "model_version": "string carregada do PKL"
    }
  ]
}
```

Os números são arredondados a quatro casas, as coordenadas a quatro casas e `model_version` vem de `versao_modelo.pkl`.

### `/predict`

Recebe o body Pydantic, chama `model_dump()`, cria um DataFrame de uma linha e reordena apenas as colunas de `features`. Calcula `predict_proba(...)[0][1]`, aplica `probabilidade >= limiar` e retorna:

```json
{
  "probability": 0.0,
  "risk": "ALTO ou BAIXO",
  "model_version": "3.0.0-sazonal-25anos"
}
```

Não existe parâmetro de data/ano no endpoint manual; o mês é informado, mas os dados históricos necessários precisam ser fornecidos pelo cliente. Não há cálculo automático das features ambientais no endpoint.

## 15. Erros, CORS e segurança observável

- HTTP 422 é esperado implicitamente pelo FastAPI para body ausente, campos obrigatórios ausentes ou tipos incompatíveis; nenhuma resposta customizada foi implementada.
- HTTP 400, 404 e mensagens de domínio específicas não são lançados explicitamente no código.
- Exceções de arquivo, seleção de colunas, carga Joblib, conversão numérica ou previsão não são capturadas; tendem a resultar em erro 500 padrão do servidor, mas não há handler que defina a resposta.
- CORS está aberto a qualquer origem, método e header via `CORSMiddleware`; isso é o mecanismo efetivamente implementado, sem classificação automática de segurança.
- Não há autenticação, autorização, gestão de secrets, proteção de endpoint ou validação de faixa para coordenadas, mês e `limit`.
- O uso de Joblib/Pickle pressupõe que os arquivos são confiáveis; não há verificação de hash ou assinatura no carregamento.

## 16. Integração com o restante do sistema

O workspace não contém cliente HTTP para o Backend nem URL de Backend/PostgreSQL. Portanto:

- endpoints disponibilizados ao consumidor são os da tabela da API;
- formato de integração observável é HTTP/JSON;
- timeout, retry, circuit breaker e contrato com o Backend são **NÃO IDENTIFICADOS NO CÓDIGO**;
- persistência no PostgreSQL é escopo do outro workspace e não é documentada aqui.

Existe uma integração local demonstrável pelo `index.html`: ele usa `http://127.0.0.1:8000/climatologia` e `http://127.0.0.1:8000/predict`, faz 12 previsões em paralelo e interpreta visualmente o campo `probability`. Isso é evidência de consumidor local, não prova de integração com o Backend.

## 17. Pipeline técnico consolidado

O pipeline diário implementado é:

```text
Copernicus CHL + SST
        -> NetCDF 2024
        -> interpolação da grade / Kelvin para Celsius / preenchimento linear
        -> dados_tratados_2024_sp.csv
        -> mês, dia do ano, lags, médias móveis, distância a hotspot
        -> target clorofila > 1,5
        -> Random Forest diário
        -> probabilidade
        -> threshold escolhido no script (0,2)
        -> classificação binária
```

O pipeline sazonal descrito pelo código é:

```text
Copernicus CHL + SST + salinidade
        -> agregação mensal 2000–2024
        -> ano anterior, climatologia por mês, rolling de 3 anos, distância
        -> remoção de NaN do ano anterior
        -> target clorofila máxima > 1,5
        -> script atual: XGBoost com 12 features / threshold 0,5
        -> artefato efetivo: Random Forest com 9 features / threshold 0,2
        -> api.py
```

O caminho efetivamente usado pela API é:

```text
CSV sazonal local carregado na inicialização
        -> entrada manual ou últimas linhas do CSV
        -> seleção da lista de features serializada
        -> RandomForestClassifier serializado
        -> predict_proba
        -> threshold 0,2
        -> ALTO/BAIXO + versão
```

## 18. Testes e resultados verificáveis

Não foram encontrados testes unitários, testes de integração, coleção Postman, fixtures ou pipeline CI. Existem scripts exploratórios e de validação manual:

- `teste_conexao.py`: tentativa de download de clorofila;
- `explorar_dados.py`: impressão de resumo/estatística de NetCDF e geração de mapa;
- `analise_exploratoria.py`: gráficos e correlação;
- `testar_configuracoes.py` e `testar_limiares.py`: experimentos de limiares;
- `diagnosticar_merge.py`: diagnóstico de coordenadas;
- `validar_com_eventos_reais.py`: avaliação manual sobre positivos reais.

Não foram executados downloads, treinos ou regravações durante este levantamento. Não há resultados persistidos de HTTP 200, previsões ou métricas. O único resultado observável de desserialização foi a identificação dos tipos/parâmetros dos modelos e o aviso de versão do scikit-learn.

## 19. Tamanhos, execução e reprodutibilidade

### Arquivos grandes

- CSV diário: aproximadamente 127–286 MiB, conforme variante;
- CSV sazonal completo: aproximadamente 285 MiB;
- CSV sazonal com salinidade: aproximadamente 113 MiB;
- NetCDFs anuais de clorofila: cerca de 3,6 MiB cada na família SP;
- NetCDFs anuais de temperatura: cerca de 2,7 MiB cada na família SP;
- Random Forest diário: aproximadamente 172 MiB;
- Random Forest sazonal carregado pela API: aproximadamente 1,07 GiB, duplicado na raiz e em `dados/`.

O modelo sazonal é necessário para iniciar a API; os quatro PKLs da raiz e o CSV sazonal carregado também são necessários. Os NetCDFs e scripts de aquisição não são lidos por `api.py` em tempo de inferência.

### Reproduzível com arquivos atuais

- Inspeção/carregamento do artefato sazonal e inferência local: parcialmente, desde que o ambiente tenha dependências compatíveis e os caminhos relativos sejam preservados.
- Endpoints da API: parcialmente; depende dos PKLs da raiz e do CSV sazonal.
- Recriação do fluxo diário a partir dos CSVs tratados: potencialmente, com dependências ausentes no `venv` corrigidas.

### Depende de arquivos externos ausentes

- reconstrução mensal oficial via `processar_dados_historicos.py`: requer `salinidade_2000_sp.nc` até `salinidade_2024_sp.nc` e `precipitacao_2000_2024_sp.csv`;
- seleção RFE/comparação completa: requer `features_selecionadas_rfe.pkl`;
- reprodução do GridSearch: depende apenas do dataset completo e scikit-learn, mas o resultado `melhores_hiperparametros.pkl` não está disponível;
- aquisição: requer conectividade e configuração válida do Copernicus e Open-Meteo.

### Não reproduzível sem confirmação

- qual processo produziu o artefato sazonal Random Forest de 9 features;
- quais versões exatas de Python, scikit-learn, Joblib e demais bibliotecas foram usadas na serialização;
- quais valores reais foram produzidos pelos relatórios de avaliação;
- qual versão deve ser considerada oficial: `3.0.0-sazonal-25anos` do artefato ou `7.0.0-xgboost-sazonal-salinidade` dos scripts/HTML.

## 20. Limitações e inconsistências reais

1. Divergência entre código de treinamento sazonal e artefato em produção: XGBoost/12/0,5/7.0.0 versus Random Forest/9/0,2/3.0.0.
2. `requirements.txt` não declara `xgboost` nem `requests`, embora sejam importados.
3. O ambiente local não possui xarray, netCDF4, copernicusmarine, xgboost, requests e matplotlib.
4. O aviso de desserialização indica scikit-learn 1.9.0 no treinamento e 1.9.1 no carregamento.
5. Os arquivos de dados/modelos são ignorados pelo Git; não existe rastreabilidade de versão, hash publicado ou pipeline de artefatos.
6. Os NetCDFs de salinidade e CSV de precipitação requeridos por scripts históricos não estão disponíveis.
7. O modelo sazonal tem tamanho muito elevado para um artefato de serviço local, aproximadamente 1,07 GiB, com duas cópias.
8. O target sazonal é um proxy baseado em clorofila máxima, e a justificativa científica do limiar 1,5 não está no código.
9. O cálculo de distância usa graus euclidianos, não distância geodésica.
10. Não há testes automatizados, monitoramento de drift, validação de schema, calibração de probabilidade ou registro de métricas.
11. `api.py` exige 12 campos de entrada, mas o modelo efetivo usa 9; campos de salinidade do body são aceitos e descartados.
12. A API carrega a base inteira na inicialização e não possui tratamento explícito para falhas de arquivo/modelo.
13. O label `source` em `/marine-data` identifica Copernicus, mas o endpoint serve uma cópia CSV estática.

## 21. Informações que precisam ser confirmadas

### Dados

- Identidade e versão exatas dos produtos Copernicus usados na geração dos CSVs atuais.
- Unidades, resolução espacial, dimensão temporal e profundidade dos NetCDFs, usando os metadados completos do arquivo.
- Processo que gerou `dados_features_sazonal_sp_completo.csv` e suas variáveis adicionais.
- Motivo da diferença de grade entre datasets.
- Se a precipitação, vento, correntes, nutrientes, oxigênio e pH são fontes reais ou placeholders em algum estágio.

### Treinamento e algoritmo

- Script/commit que gerou o Random Forest sazonal de 9 features.
- Qual artefato é oficialmente publicado para consumo.
- Se o modelo deve ser Random Forest ou XGBoost.
- Se o treino oficial usou todos os dados ou o corte temporal.
- Valor e justificativa oficial do threshold.

### Métricas

- Valores efetivos de F1, recall, precision, accuracy, ROC-AUC e matriz de confusão.
- Dataset/período usados para cada métrica.
- Se os números hard-coded em `comparar_xgboost.py` são confiáveis e de qual execução vieram.
- Resultado completo da validação com eventos reais e definição de “acerto”.

### Features e target

- Significado científico do limiar de clorofila 1,5.
- Se a média histórica deve excluir o ano corrente.
- Se `salinidade_*` deve participar da inferência oficial.
- Unidade pretendida para `dist_hotspot` e se deve ser convertida para km.
- Se lags representam dias exatos ou apenas posições na série ordenada.

### Artefatos e infraestrutura

- Versão compatível de Python/scikit-learn/Joblib para produção.
- Política de armazenamento/versionamento dos PKLs e CSVs.
- Comando oficial de instalação e inicialização.
- URL, timeout e contrato do Backend.
- Configuração/autenticação Copernicus fora dos scripts.

## 22. Mapeamento para documentação acadêmica

| Capítulo | Informação disponível neste workspace | Lacunas principais | Evidências |
|---|---|---|---|
| 1. Introdução | Tema e finalidade inferível pelo título da API/HTML. | Motivação acadêmica e referências. | `api.py:7`, `index.html:5–6`. |
| 2. Fundamentação teórica | Nomes de variáveis/modelos. | Fundamentação científica de HAB, clorofila e risco. | **NÃO IDENTIFICADO NO CÓDIGO**. |
| 3. Visão geral do Algamar | Recorte litoral de SP e previsão sazonal. | Contexto integral do sistema. | `api.py`, `index.html`. |
| 4. Arquitetura | Fluxo local FastAPI → artefato → CSV; consumidor local HTTP. | Backend/PostgreSQL e deployment. | `api.py`, `index.html:240–258`. |
| 5. Banco de dados | Nenhum acesso PostgreSQL. | Todo o capítulo. | **FORA DESTE WORKSPACE**. |
| 6. Engenharia e fluxo dos dados | Aquisição, limpeza, agregação, interpolação e features. | Proveniência completa e credenciais. | `organizar_dados.py`, `processar_dados_historicos.py`, `feature_engineering*.py`. |
| 7. Machine Learning | Targets, features, scripts de treino, artefatos e divergências. | Métricas oficiais e origem do artefato. | `treinar_modelo*.py`, `salvar_modelo*.py`, PKLs. |
| 8. Backend | Nenhuma implementação. | Capítulo inteiro. | **OUTRO WORKSPACE**. |
| 9. APIs e integração | Endpoints, Pydantic, JSON, CORS e consumidor local. | Contrato com Backend. | `api.py`, `index.html`. |
| 10. Segurança | CORS aberto, sem auth/handlers explícitos. | Threat model e ambiente de produção. | `api.py:9–14`. |
| 11. Testes e validação | Scripts manuais e métricas calculadas em código. | Logs, testes automatizados e resultados. | `testar_*.py`, `validar_com_eventos_reais.py`. |
| 12. Infraestrutura | venv 3.12.3, requirements e artefatos locais. | Deploy, host/porta oficial e CI. | `venv/pyvenv.cfg`, `requirements.txt`. |
| 13. Frontend — reservado | Não documentado neste levantamento. | Workspace separado. | Seção abaixo. |
| 14. Resultados e discussão | Faixas e contagens dos datasets; nenhuma métrica oficial persistida. | Interpretação científica e resultados validados. | CSVs, scripts de treino. |
| 15. Trabalhos futuros | Lacunas e inconsistências identificadas. | Priorização pelos responsáveis. | Seção 20. |
| 16. Considerações finais | Inventário do pipeline real. | Síntese acadêmica. | Este documento. |
| 17. Referências | Identificadores Copernicus e URL Open-Meteo no código. | Referências bibliográficas formais. | Scripts de aquisição. |
| 18. Apêndices | Listas de features, endpoints, formatos e comandos implícitos. | Exemplos oficiais de payload/resposta e logs. | `api.py`, PKLs e CSVs. |

## 23. Frontend — documentação futura

O Frontend será documentado posteriormente em workspace separado.

Deverão ser analisados:

- tecnologias;
- arquitetura;
- páginas;
- componentes;
- consumo da API;
- autenticação;
- dashboards;
- mapas;
- visualização das previsões;
- UX/UI;
- responsividade;
- tratamento de erros;
- testes.

Neste workspace, `index.html` foi consultado apenas para confirmar o consumidor local da API e as URLs/formatos usados na integração; não foi produzido levantamento acadêmico do Frontend.

## 24. Matriz de evidências principais

| Informação | Evidência primária |
|---|---|
| CORS aberto | `api.py`, linhas 9–14. |
| Arquivos carregados pela API | `api.py`, linhas 16–23. |
| Campos do request | `api.py`, linhas 26–38. |
| `/marine-data` | `api.py`, linhas 51–72. |
| `/climatologia` e distância de ponto | `api.py`, linhas 75–112. |
| `/predictions` e threshold | `api.py`, linhas 115–140. |
| `/predict` | `api.py`, linhas 143–153. |
| Conversão Kelvin/Celsius e interpolação | `organizar_dados.py`, linhas 3–43; `processar_dados_historicos.py`, linhas 15–30. |
| Target diário | `feature_engineering.py`, linhas 61–64. |
| Target sazonal | `feature_engineering_sazonal.py`, linhas 31–32. |
| Features diárias | `feature_engineering.py`, linhas 7–54. |
| Features sazonais | `feature_engineering_sazonal.py`, linhas 7–25. |
| RF diário | `treinar_modelo.py`, linhas 39–47; `salvar_modelo.py`, linhas 27–37. |
| XGBoost descrito pelo script sazonal | `treinar_modelo_sazonal.py`, linhas 29–35; `salvar_modelo_sazonal.py`, linhas 20–26. |
| Artefato efetivo RF/9 features | inspeção Joblib de `modelo_floracao_sazonal.pkl` e `features_modelo_sazonal.pkl`. |
| Versão/threshold efetivos | inspeção Joblib de `versao_modelo.pkl` e `limiar_decisao_sazonal.pkl`. |
| Download Copernicus | `baixar_dados_historicos.py`, `baixar_dados_completos.py`, `baixar_salinidade.py`. |
| Download Open-Meteo | `baixar_precipitacao.py`, linhas 13–23 e 74–82. |
| Eventos reais | `integrar_eventos_reais.py`, linhas 4–76. |
| Versão Python declarada pelo ambiente | `venv/pyvenv.cfg`, linhas 1–5. |
| Dependências declaradas | `requirements.txt`, linhas 1–16. |
| Integração local | `index.html`, linhas 240–258 e 290–323. |

## 25. Conclusão do levantamento

O workspace permite confirmar a existência de um serviço FastAPI que faz inferência local sobre um Random Forest sazonal serializado, com probabilidades e classificação binária por limiar. Também permite reconstruir grande parte dos pipelines diário e sazonal pelos scripts e pelos CSVs.

Não permite afirmar, sem confirmação adicional, que o XGBoost descrito nos scripts seja o modelo servido atualmente, nem fornece métricas oficiais, origem versionada do artefato sazonal, configuração completa do Copernicus, integração com Backend ou procedimento de implantação. Essas lacunas estão separadas acima para evitar que a documentação acadêmica posterior trate intenção de código como comportamento efetivo.
