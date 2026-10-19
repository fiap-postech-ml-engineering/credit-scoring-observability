# Observabilidade de Pipelines e Modelos (Etapa 3)

> Visibilidade operacional dos pipelines de dados, da infraestrutura e dos
> modelos em produção.

## As três camadas

| Camada | Ferramenta | Pergunta que responde |
|---|---|---|
| Logs estruturados | JSON Lines (`logs/pipeline.jsonl`) | o que aconteceu, passo a passo? |
| Rastreamento de modelo | MLflow (SQLite + Model Registry) | este modelo degradou? desde quando? |
| Métricas e alertas | Prometheus + Grafana | está tudo de pé agora? o que disparou? |

Nenhuma delas é decorativa: o dashboard consolidado
(`reports/dashboard.html`) é montado a partir dos laudos que essas camadas
produzem, e o Prometheus lê os mesmos arquivos.

## Logs estruturados

Todo módulo loga com `logger.info("mensagem", extra={...})`. O `JsonFormatter`
promove os campos de `extra` ao topo do objeto JSON — formato que Loki, Datadog
ou CloudWatch ingerem sem parser customizado.

```json
{"timestamp":"2026-10-19T10:01:12+00:00","level":"INFO","logger":"src.pipeline.monitor",
 "message":"pipeline concluído","batch":"2024-05_recessao","status":"sucesso",
 "alerts":["degradacao_silenciosa","queda_de_desempenho"],"duration_seconds":4.12}
```

Saída dupla: terminal legível para humanos, arquivo em JSON para máquinas. O
arquivo é a fonte do relatório consolidado de execuções.

Um detalhe que custou um bug: `logging` levanta `KeyError` se o `extra` tentar
sobrescrever um atributo reservado do `LogRecord` — e `name`, `module` e
`filename` são nomes perfeitamente naturais num payload de domínio. O helper
`safe_extra()` prefixa os colidentes com `ctx_` em vez de proibir esses nomes
no resto do código.

## Rastreamento com MLflow

Dois tipos de run convivem no mesmo experimento, separados pela tag `run_type`:

**`training`** — um run por treino do baseline:
- parâmetros e métricas de validação de **cada candidato** (a escolha fica auditável);
- métricas do vencedor na Referência — a baseline contra a qual todo lote será comparado;
- o modelo com assinatura, registrado no **Model Registry** como `credit-scoring-baseline`;
- o dataset de Referência como artefato, fechando a linhagem.

**`monitoring`** — um run por lote avaliado:
- resultado do contrato (`contract_passed`, `contract_violations`);
- PSI de cada feature (`psi_age`, `psi_MonthlyIncome`, …);
- desempenho no lote (`prod_roc_auc`, `prod_approved_bad_rate`, …);
- variação contra a baseline (`delta_roc_auc`, …);
- flags de alerta (`silent_degradation`, `performance_degraded`);
- os HTMLs do Evidently como artefatos.

Com isso o MLflow responde, numa única tela, "este modelo degradou e desde
quando?" — comparando os runs `monitoring` ao longo dos lotes contra a
baseline do run `training`.

Duas notas de implementação:

- **SQLite, não arquivos.** O MLflow 3 colocou o backend de arquivos em
  *maintenance mode* e recusa `file://` por padrão. SQLite é o backend local
  suportado e o único que habilita o Model Registry.
- **`skops_trusted_types`.** O MLflow 3 serializa modelos sklearn com skops,
  que recusa classes de fora do sklearn. Como o `XGBClassifier` vive dentro do
  nosso `Pipeline`, ele é declarado explicitamente — a lista exata de tipos que
  nós mesmos colocamos no modelo, não um "confie em tudo".

Falha de rastreamento **nunca derruba o pipeline**: `log_monitoring_run`
captura a exceção, loga um aviso e devolve `None`. Observabilidade quebrada
degrada a visibilidade, não o serviço. O teste
`test_falha_de_rastreamento_nao_derruba_o_pipeline` garante isso.

## Métricas Prometheus

### O problema arquitetural

O monitoramento de drift é um job **batch**, e o Prometheus só sabe *raspar*
alvos que estão de pé. Um processo que roda, emite métricas e morre nunca é
raspado.

### A solução: a API como gateway de métricas

Em vez de subir um Pushgateway, `DriftMetricsCollector` é um collector
customizado que, **no momento do scrape**, lê os laudos JSON que o pipeline
deixou em `reports/` e os traduz em séries Prometheus.

Resultado: um único alvo de scrape (`api:8000/metrics`) expõe ao mesmo tempo as
métricas ao vivo da API e as da última execução do pipeline batch. E como os
laudos em disco são a fonte da verdade, reiniciar a API não perde histórico de
monitoramento — o que aconteceria com métricas em memória.

### Métricas de saúde documentadas

**Pipeline de dados**

| Métrica | Tipo | O que vigia |
|---|---|---|
| `credit_pipeline_task_status{batch,task}` | gauge | 1 sucesso, 0 falha/bloqueio |
| `credit_pipeline_task_duration_seconds{batch,task}` | gauge | latência por tarefa |
| `credit_pipeline_rows_processed{batch}` | gauge | volume processado |
| `credit_pipeline_last_run_timestamp_seconds` | gauge | detecta pipeline que parou de rodar |

**Qualidade de dados**

| Métrica | Tipo | O que vigia |
|---|---|---|
| `credit_contract_passed{batch}` | gauge | o lote passou no portão? |
| `credit_contract_violations{batch}` | gauge | quantas linhas reprovaram |
| `credit_contract_rejection_rate{batch}` | gauge | fração do lote reprovada |
| `credit_contract_violations_by_rule{batch,rule}` | gauge | *qual* regra foi violada |

**Modelo e drift**

| Métrica | Tipo | O que vigia |
|---|---|---|
| `credit_feature_psi{batch,feature}` | gauge | PSI por feature |
| `credit_max_psi{batch}` | gauge | pior PSI do lote |
| `credit_drifted_features_count{batch}` | gauge | quantas features em drift |
| `credit_dataset_drift{batch}` | gauge | drift do dataset inteiro |
| `credit_model_performance{batch,metric}` | gauge | AUC, KS, inadimplência aprovada… |
| `credit_model_performance_delta{batch,metric}` | gauge | variação contra a baseline |
| `credit_drift_alert{batch,alert}` | gauge | alerta ativo |
| `credit_silent_degradation{batch}` | gauge | **o alerta que só o cruzamento levanta** |

**API em tempo real**

| Métrica | Tipo | O que vigia |
|---|---|---|
| `credit_http_requests_total{method,path,status}` | counter | tráfego e taxa de erro |
| `credit_http_request_duration_seconds` | histogram | latência fim a fim |
| `credit_scoring_duration_seconds` | histogram | latência da inferência, sem rede |
| `credit_score_distribution` | histogram | distribuição dos scores emitidos |
| `credit_decisions_total{decision}` | counter | aprovações vs. recusas |
| `credit_contract_rejections_total{reason}` | counter | propostas barradas na borda |

A rota é rotulada pelo **template** (`/score`), não pelo caminho concreto —
rotular por caminho com parâmetros explodiria a cardinalidade das séries.

## Alertas

`observability/alert_rules.yml` promove as flags a notificações:

| Alerta | Expressão | Severidade |
|---|---|---|
| `DegradacaoSilenciosa` | `credit_silent_degradation > 0` | critical |
| `ContratoDeDadosViolado` | `credit_contract_passed == 0` | critical |
| `TarefaDoPipelineFalhou` | `credit_pipeline_task_status == 0` | critical |
| `ApiIndisponivel` | `up{job="credit-scoring-api"} == 0` | critical |
| `DriftDeFeature` | `credit_max_psi >= 0.25` | warning |
| `QuedaDeDesempenho` | `credit_model_performance_delta{metric="roc_auc"} <= -0.03` | warning |

Os limiares espelham `src/config/settings.py` de propósito: o pipeline já marca
o alerta no laudo e o Prometheus apenas o promove a notificação — a regra de
negócio vive num lugar só.

## Dashboard Grafana

`observability/grafana/dashboards/credit-scoring-drift.json`, provisionado
automaticamente, com 11 painéis: quatro KPIs no topo (alertas ativos,
degradação silenciosa, PSI máximo, inadimplência da carteira aprovada), o
heatmap de PSI por feature × lote, as duas séries de AUC, a tabela de status
das tarefas, as violações de contrato por regra e os dois painéis de API.

Cor segue a mesma disciplina do dashboard HTML: magnitude usa um só matiz
(azul sequencial), estado usa a paleta de status, e nenhum gráfico tem dois
eixos y.

## Dashboard HTML consolidado

Os relatórios do Evidently respondem muito bem "o que aconteceu neste lote?",
mas são quinze arquivos. Falta a visão de cima.

`reports/dashboard.html` (`src/pipeline/dashboard.py`) é um arquivo único,
offline, com tema claro e escuro, contendo KPIs, heatmap de PSI, as duas séries
temporais, alertas por lote com ícone e rótulo, status das tarefas, violações
do contrato e a tabela completa em texto para leitores de tela.

Os SVGs são montados à mão em vez de via biblioteca de plot por três motivos:
abre offline (zero CDN), pesa 46 KB em vez de 4 MB, e herda o tema da página
via CSS custom properties — coisa que um PNG não faz.
