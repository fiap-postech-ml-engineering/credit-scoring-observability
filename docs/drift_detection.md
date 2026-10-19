# Simulação e Detecção de Drift (Etapa 2)

> Diagnosticar a degradação do modelo ao longo do tempo.

## Referência vs. Produção

A separação é feita uma vez, em `src/credit_scoring/dataset.py`:

| Conjunto | Linhas | Papel |
|---|---|---|
| **Referência** | 88.761 | treina o baseline; é a régua de comparação |
| **Pool futuro** | 59.174 | matéria-prima dos lotes de produção |

O split é estratificado no alvo e **o pool futuro nunca é visto pelo treino**.
Isso importa para a honestidade da demonstração: Referência e pool partem da
mesma distribuição, então qualquer drift medido depois foi introduzido de
propósito — não é artefato da separação. O lote de controle
(`2024-01_estavel`) comprova: PSI máximo de 0,002 e zero features em drift.

## Os dois tipos de drift

| | Data drift | Concept drift |
|---|---|---|
| O que muda | `P(X)` — distribuição das features | `P(Y\|X)` — relação feature→alvo |
| Exemplo | inflação infla a renda nominal | recessão faz o mesmo perfil inadimplir |
| PSI / KS detectam? | **sim** | **não** |
| Precisa de rótulo? | não | **sim** |
| Resposta | recalibrar, reponderar | **retreinar** |

Essa distinção é o eixo do projeto. Monitoramento que só olha distribuição de
features é cego para metade do problema — e, como mostra a
[análise causal](causal_analysis.md), para a metade maior.

## Os cinco lotes mensais

Gerados por `src/drift/simulate.py`, determinísticos (mesma semente, mesmo lote):

| Lote | Tipo | O que muda |
|---|---|---|
| `2024-01_estavel` | nenhum | nada — controle contra falso positivo |
| `2024-02_inflacao_leve` | data | renda nominal +8% |
| `2024-03_inflacao_forte` | data | renda +45%, endividamento +60% |
| `2024-04_novo_perfil` | data + covariável | o anterior + base 12 anos mais jovem |
| `2024-05_recessao` | **concept** | features idênticas; jovens sem patrimônio passam a inadimplir |

### Por que a inflação degrada de verdade

Inflar a renda nominal não é truque: o modelo aprendeu **limiares nominais** de
renda no dataset de Referência. R$ 5.000 em 2024 não compram o que compravam no
treino, mas o modelo trata o valor como se comprassem. É degradação real por
uma causa real.

### Como o concept drift foi construído

O baseline aprendeu, no dado pré-crise, que cliente **jovem sem patrimônio
imobiliário** não é especialmente arriscado. Na recessão é exatamente esse
grupo que perde o emprego primeiro e não tem colateral para queimar.

`_recession_concept_drift` vira rótulos `0 → 1` nesse subgrupo, com
probabilidade proporcional à vulnerabilidade (jovem × sem colateral × rotativo
alto × dependentes), **sem tocar em nenhuma feature**.

Duas propriedades garantidas por teste:

- `test_concept_drift_so_mexe_no_alvo` — toda feature permanece idêntica;
- `test_concept_drift_so_piora_quem_estava_bom` — rótulos só viram de 0 para 1;
  ninguém "desinadimple" numa recessão.

## PSI e KS

`src/drift/statistics.py` implementa os dois testes à mão — o enunciado pede
um relatório estatístico próprio, conferível na mão.

**PSI (Population Stability Index)** é o critério. Faixas definidas pelos
quantis da Referência, o que o torna comparável entre features de escalas
diferentes (`age` e `MonthlyIncome`):

| PSI | Leitura |
|---|---|
| < 0,10 | estável |
| 0,10 – 0,25 | atenção |
| > 0,25 | drift relevante |

**KS (Kolmogorov-Smirnov)** é sinal de apoio, não critério. Motivo: com 8.000
linhas por lote, qualquer teste de hipótese rejeita a igualdade por diferenças
irrelevantes. O teste `test_psi_e_estavel_quanto_ao_tamanho_da_amostra`
formaliza por que o PSI é que decide.

## Resultados

Baseline de validação (fora da amostra): **ROC AUC 0,8687 · KS 0,5920 ·
inadimplência da carteira aprovada 2,00%**.

| Lote | Tipo | ROC AUC | Δ AUC | PSI máx | nº em drift | Inadimpl. aprovada | Alertas |
|---|---|---|---|---|---|---|---|
| `2024-01_estavel` | nenhum | 0,8711 | **+0,0024** | 0,002 | 0 | 1,71% | — |
| `2024-02_inflacao_leve` | data | 0,8619 | −0,0068 | 0,055 | 0 | 1,97% | — |
| `2024-03_inflacao_forte` | data | 0,8311 | −0,0376 | **0,467** | 2 | 1,92% | queda de desempenho, drift de feature |
| `2024-04_novo_perfil` | data+cov | 0,8254 | −0,0432 | **0,549** | 3 | 1,79% | queda, drift de dataset, drift de feature |
| `2024-05_recessao` | **concept** | 0,8233 | −0,0454 | **0,002** | **0** | **4,48%** | **degradação silenciosa**, queda |

### O achado

Compare as duas últimas linhas. A queda de AUC é praticamente a mesma
(−0,0432 vs. −0,0454), mas:

- em `2024-04`, o PSI grita: **0,549**, três features em drift;
- em `2024-05`, o PSI é **0,002** e *nenhum* p-valor de KS é significativo
  (0,17 a 1,0) — as features são estatisticamente indistinguíveis da
  Referência;
- e, ainda assim, `2024-05` é o lote **mais caro**: a inadimplência na carteira
  aprovada sobe de 2,00% para **4,48%** — 2,25× o prejuízo.

Tabela completa de PSI/KS do lote de recessão (`reports/2024-05_recessao__psi_ks.csv`):

```
feature                                psi    ks_stat   ks_p_value   severity
DebtRatio                           0.0020    0.0121       0.2290    estavel
MonthlyIncome                       0.0015    0.0129       0.1692    estavel
NumberOfOpenCreditLinesAndLoans     0.0008    0.0066       0.9031    estavel
age                                 0.0008    0.0111       0.3273    estavel
RevolvingUtilizationOfUnsecuredLines 0.0005   0.0121       0.2278    estavel
...
```

Um painel de data drift mostraria tudo verde enquanto a carteira dobra de
prejuízo. **É literalmente a degradação silenciosa do enunciado.**

## Como o alerta é decidido

`DriftReport.alerts()` combina os dois mundos:

| Alerta | Condição |
|---|---|
| `drift_de_feature` | PSI máximo ≥ 0,25 |
| `drift_de_dataset` | ≥ 30% das features em drift |
| `queda_de_desempenho` | Δ ROC AUC ≤ −0,03 contra a baseline |
| `degradacao_silenciosa` | queda de desempenho **e** zero features em drift |

O último é o alerta que só existe porque cruzamos as duas fontes. Nenhum
monitor de distribuição sozinho consegue levantá-lo.

## Relatórios gerados

Por lote, três relatórios do Evidently mais dois laudos textuais:

| Arquivo | Conteúdo |
|---|---|
| `{lote}__data_drift.html` | `DataDriftPreset` — distribuição de cada feature |
| `{lote}__classification.html` | `ClassificationPreset` — o único que vê concept drift |
| `{lote}__data_summary.html` | `DataSummaryPreset` — sanidade do lote |
| `{lote}__psi_ks.csv` | laudo estatístico próprio, feature a feature |
| `{lote}__drift.json` | resumo estruturado consumido por MLflow, Prometheus e dashboard |

Mais o dashboard consolidado em `reports/dashboard.html`, que é a visão de cima
que quinze arquivos separados não dão.
