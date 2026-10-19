# Contratos de Dados (Etapa 1)

> Garantir que lixo não entre no modelo.

## Por que Pandera

Avaliamos Pandera e Great Expectations. Escolhemos **Pandera** por três razões:

1. **Code-first e tipado.** O contrato é uma classe Python (`DataFrameModel`),
   então vive no mesmo lugar do código que ele protege, é revisado no mesmo PR
   e o `ty` consegue checá-lo.
2. **Roda no CI em menos de um segundo.** Great Expectations exige um *data
   context*, um store de expectativas e um passo de build dos Data Docs —
   infraestrutura demais para um contrato de dez regras.
3. **Os limites são reutilizáveis.** As constantes de `src/contracts/schemas.py`
   são importadas tanto pelos schemas Pydantic da API quanto pelo gerador de
   lotes inválidos. Um único lugar define que idade mínima é 18.

O que perdemos com essa escolha — os Data Docs em HTML que o Great
Expectations gera de graça — foi recuperado no dashboard consolidado
(`reports/dashboard.html`), que tem uma seção dedicada às violações por regra.

## Os dois schemas

| Schema | Uso | Alvo |
|---|---|---|
| `CreditInferenceSchema` | API em tempo real, lotes não rotulados | não |
| `CreditReferenceSchema` | treino, lotes rotulados de produção | sim |

Ambos usam `strict=True` (recusa coluna desconhecida) e `coerce=True`
(normaliza dtypes antes de validar).

`strict=True` não é detalhe de estilo: é o que impede um upstream desavisado
de injetar uma coluna de PII — um CPF, por exemplo — num lote que vai para o
modelo. Essa regra aparece como defeito injetado nos testes exatamente por
isso. Ver [governança e LGPD](governance_lgpd.md).

## As regras

Dez regras rígidas, derivadas da análise do dataset bruto:

| # | Regra | Coluna | Motivo |
|---|---|---|---|
| 1 | `0 ≤ x ≤ 10` | `RevolvingUtilizationOfUnsecuredLines` | é proporção; o bruto chega a 50.708 |
| 2 | `18 ≤ idade ≤ 100` | `age` | menor não contrata crédito; o bruto tem `age = 0` |
| 3 | `0 ≤ x ≤ 20` | contadores de atraso (3 colunas) | bloqueia os sentinelas 96/98 do sistema legado |
| 4 | `0 ≤ x ≤ 10` | `DebtRatio` | razão dívida/renda; o bruto chega a 329.664 |
| 5 | não nulo, `0 ≤ x ≤ 1.000.000` | `MonthlyIncome` | renda é obrigatória para decidir crédito |
| 6 | sem duplicatas | lote inteiro | duplicata é dupla contagem do mesmo tomador |
| 7 | lote não vazio | lote inteiro | lote vazio é falha silenciosa de ingestão |
| 8 | `x ∈ {0, 1}` | `SeriousDlqin2yrs` | o problema é binário |
| 9 | não nulo, `0 ≤ x ≤ 20` | `NumberOfDependents` | 3.924 nulos no bruto |
| 10 | nenhuma coluna extra | lote inteiro | barra PII e campos não contratados |

## O dado bruto viola nove delas

Validar `cs-training.csv` cru contra o contrato reprova **21,4% das linhas**:

```
CreditReferenceContract               o lote contém linhas duplicadas         1
DebtRatio                             less_than_or_equal_to(10.0)         28877
MonthlyIncome                         less_than_or_equal_to(1000000.0)        4
                                      not_nullable                        29731
NumberOfDependents                    not_nullable                         3924
NumberOfTime30-59DaysPastDueNotWorse  less_than_or_equal_to(20)             269
NumberOfTime60-89DaysPastDueNotWorse  less_than_or_equal_to(20)             269
NumberOfTimes90DaysLate               less_than_or_equal_to(20)             269
NumberRealEstateLoansOrLines          less_than_or_equal_to(30)               2
RevolvingUtilizationOfUnsecuredLines  less_than_or_equal_to(10.0)           241
age                                   greater_than_or_equal_to(18)            1
                                      less_than_or_equal_to(100)             13
```

Isso não é um defeito da demonstração — é o ponto dela. O contrato foi escrito
a partir do que o negócio aceita, e o dado real não cumpre. A ponte entre os
dois é a etapa de transformação.

## A transformação, passo a passo

`src/credit_scoring/dataset.py::clean` leva o bruto ao contrato registrando
cada decisão. De 150.000 linhas sobram **147.935** (1,38% descartado):

| Passo | Efeito | Linhas | Decisão |
|---|---|---|---|
| `drop_duplicates` | descarta | −609 | dupla contagem |
| `drop_invalid_age` | descarta | −14 | erro de cadastro, não outlier |
| `drop_sentinel_late` | descarta | −225 | 96/98 significam "sem histórico avaliável" |
| `clip_revolving` | corrige | 241 | cauda absurda, linha válida |
| `clip_debt_ratio` | corrige | 28.844 | idem |
| `clip_real_estate` | corrige | 2 | idem |
| `impute_income_by_age_band` | corrige | 29.134 | mediana da faixa etária |
| `clip_income` | corrige | 4 | teto do contrato |
| `impute_dependents` | corrige | 3.808 | moda (zero) |
| `drop_duplicates_post_transform` | descarta | −1.217 | winsorização colapsa linhas |

Duas decisões merecem justificativa:

**Por que imputar renda em vez de descartar?** Descartar 19,8% das linhas
enviesaria a Referência contra quem não declara renda — justamente o perfil
mais informal, que é parte relevante da carteira. A imputação pela mediana da
faixa etária é conservadora e, sobretudo, **auditável**: o contador
`imputed_income` vai para o log estruturado e vira métrica de pipeline. Uma
imputação que ninguém consegue contar depois é imputação silenciosa, e é isso
que o projeto inteiro combate.

**Por que o segundo `drop_duplicates`?** Porque a winsorização e a imputação
*criam* duplicatas: duas linhas que diferiam só num `DebtRatio` de 300.000 vs.
400.000 viram idênticas depois do clip em 10. Sem essa segunda passada, o
próprio dataset de Referência reprovaria no contrato — a transformação
produziria dado que o contrato recusa. Essa é a classe de bug que uma
validação pós-transformação pega e uma pré-transformação não.

## O portão em ação

`src/contracts/bad_batch.py` monta um lote com **11 defeitos injetados**, um
por regra, cada um um defeito realista de pipeline de crédito:

| Injeção | Defeito | Regra que intercepta |
|---|---|---|
| `renda_nula` | 40 propostas sem renda no onboarding | `MonthlyIncome:not_nullable` |
| `menor_de_idade` | 15 solicitantes com menos de 18 | `age:greater_than_or_equal_to(18)` |
| `idade_impossivel` | 5 idades acima de 100 | `age:less_than_or_equal_to(100)` |
| `renda_negativa` | sinal invertido na integração | `MonthlyIncome:greater_than_or_equal_to(0.0)` |
| `sentinela_atraso` | códigos 96/98 vazando como contagem | `NumberOfTime30-59…`, `NumberOfTimes90DaysLate` |
| `endividamento_absurdo` | `DebtRatio` de 329.664x | `DebtRatio:less_than_or_equal_to(10.0)` |
| `utilizacao_absurda` | utilização de 50.708x o limite | `RevolvingUtilization…:less_than_or_equal_to(10.0)` |
| `alvo_invalido` | alvo = 2 num problema binário | `SeriousDlqin2yrs:isin([0, 1])` |
| `dependentes_nulos` | 15 linhas sem dependentes | `NumberOfDependents:not_nullable` |
| `coluna_inesperada` | coluna `cpf` (PII!) de um upstream | `column_in_schema` |
| `linhas_duplicadas` | reprocessamento duplicando linhas | `o lote contém linhas duplicadas` |

Resultado (`reports/lote_corrompido__validation.json`):

```
lote_corrompido: 530 linhas, 167 violações em 14 regras, 25,5% reprovado
status do pipeline: BLOQUEADA
tarefas inferencia / deteccao_drift / publicacao: BLOQUEADAS
```

**Nenhuma linha desse lote chegou ao modelo.** É isso que `enforce` garante:
ele levanta `DataContractViolationError` e o encadeamento para ali.

O manifesto em `reports/lote_corrompido__manifest.json` cruza "o que foi
injetado" com "o que o contrato pegou", e o teste
`test_lote_corrompido_dispara_todas_as_injecoes` falha se alguém afrouxar uma
regra. No CI isso é um job com nome próprio (`contracts-gate`), para a falha
aparecer como "portão de dados quebrou" e não escondida no meio da suíte.

## O contrato também vale na borda

Os mesmos limites são aplicados proposta a proposta pelos schemas Pydantic da
API (`src/api/schemas.py`), que importam as constantes de `schemas.py`. Uma
proposta com idade 15 recebe **422** e o modelo nunca é chamado — mesmo portão,
outro ponto de aplicação.

## Drift não é dado inválido

Uma confusão que vale explicitar: um lote em drift **passa** no contrato. O
teste `test_lotes_com_data_drift_continuam_validos_no_contrato` formaliza isso.

Dado inválido é lixo — barra. Dado em drift é informação legítima de um mundo
que mudou — deixa entrar e **mede**. Barrar drift seria cegar o monitoramento
exatamente quando ele mais importa.
