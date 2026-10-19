#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = credit-scoring-observability
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: help requirements create_environment data train pipeline contracts drift attribution \
        dashboard mlflow api stack stack-down test test-slow test-cov lint lint-fix \
        lint-fix-unsafe format format-fix format-diff check check-slow clean clean-artifacts

.DEFAULT_GOAL := help

help:
	@echo "Comandos disponíveis:"
	@echo "  make requirements        - Instala dependências Python (uv sync)"
	@echo "  make create_environment  - Cria ambiente virtual com uv"
	@echo ""
	@echo "  make data                - Baixa o dataset e separa Referência / pool futuro"
	@echo "  make train               - Treina o baseline e registra o run no MLflow"
	@echo "  make contracts           - Demonstra o contrato bloqueando um lote corrompido"
	@echo "  make drift               - Gera os cinco lotes mensais de produção"
	@echo "  make attribution         - Experimento causal: data drift vs. concept drift"
	@echo "  make pipeline            - Execução ponta a ponta (dados -> relatórios)"
	@echo "  make dashboard           - Caminho do dashboard HTML consolidado"
	@echo ""
	@echo "  make api                 - Sobe a API localmente (sem Docker)"
	@echo "  make mlflow              - Sobe a UI do MLflow em localhost:5000"
	@echo "  make stack               - Sobe API + MLflow + Prometheus + Grafana (Docker)"
	@echo "  make stack-down          - Derruba a stack do Docker Compose"
	@echo ""
	@echo "  make test                - Roda testes com output verboso (exclui slow)"
	@echo "  make test-slow           - Roda apenas os testes marcados como slow"
	@echo "  make test-cov            - Roda testes com cobertura (relatório HTML)"
	@echo ""
	@echo "  make lint                - Verifica estilo do código com Ruff"
	@echo "  make lint-fix            - Corrige automaticamente issues de linting"
	@echo "  make lint-fix-unsafe     - Corrige automaticamente issues com unsafe-fixes"
	@echo "  make format              - Verifica formatação sem modificar (Ruff)"
	@echo "  make format-fix          - Formata código com Ruff"
	@echo "  make format-diff         - Mostra diferenças de formatação sem modificar"
	@echo ""
	@echo "  make check               - Executa lint, format e testes (exclui slow)"
	@echo "  make check-slow          - Executa lint, format e todos os testes"
	@echo ""
	@echo "  make clean               - Remove arquivos temporários"
	@echo "  make clean-artifacts     - Remove dados, modelos e relatórios gerados"

## Dependências

requirements:
	uv sync --extra dev --extra lint --extra test --extra typecheck

create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> Ambiente virtual criado. Ative com:"
	@echo ">>> Windows: .\\.venv\\Scripts\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

## Projeto

data:
	uv run python -m src.credit_scoring.dataset

train:
	uv run python -m src.credit_scoring.train

drift:
	uv run python -m src.drift.simulate

attribution:
	uv run python -m src.drift.attribution

contracts:
	uv run python -c "from src.observability.logging_setup import setup_logging; setup_logging(); from src.pipeline.run import demo_contract_block; print(demo_contract_block())"

pipeline:
	uv run python -m src.pipeline.run

dashboard:
	@echo "Abra: file://$(CURDIR)/reports/dashboard.html"

## Serviços

api:
	uv run uvicorn src.api.api:api --host 0.0.0.0 --port 8000 --reload

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

stack:
	docker compose up --build

stack-down:
	docker compose down

## Testes

test:
	uv run python -m pytest tests/ -v --no-cov -m "not slow"

test-slow:
	uv run python -m pytest tests/ -v --no-cov -m "slow"

test-cov:
	uv run python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

## Lint

lint:
	uv run ruff check src/ tests/

lint-fix:
	uv run ruff check src/ tests/ --fix

lint-fix-unsafe:
	uv run ruff check src/ tests/ --fix --unsafe-fixes

## Formatação

format:
	uv run ruff format --check src/ tests/

format-fix:
	uv run ruff format src/ tests/

format-diff:
	uv run ruff format --diff src/ tests/

## Checks combinados

check: lint format test
	@echo "✓ Todos os checks passaram!"

check-slow: lint format test test-slow
	@echo "✓ Todos os checks (incluindo slow) passaram!"

## Limpeza

clean:
	rm -rf .pytest_cache .coverage .coverage.* htmlcov .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-artifacts:
	rm -rf data/raw/*.csv data/reference/*.csv data/production/*.csv
	rm -rf models/*.joblib reports/*.html reports/*.json reports/*.csv
	rm -rf logs/*.jsonl mlflow.db mlartifacts/
	@echo "Artefatos removidos. Rode 'make pipeline' para regenerar tudo."
