"""Configurações do projeto, carregadas de variáveis de ambiente/.env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Alvo binário do dataset Give Me Some Credit: 1 = inadimplência grave em 2 anos.
TARGET = "SeriousDlqin2yrs"

# Coluna de score produzida pelo modelo e consumida pelos relatórios do Evidently.
PREDICTION = "prediction"
PREDICTION_PROBA = "prediction_proba"

# Features usadas pelo modelo. A ordem importa: é o contrato de entrada do
# pipeline sklearn/XGBoost e é reaproveitada pelos schemas Pandera.
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


class Settings(BaseSettings):
    """Caminhos e limiares do projeto. Tudo sobrescrevível por env/.env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Dados -------------------------------------------------------------
    raw_path: Path = BASE_DIR / "data" / "raw" / "credit.csv"
    reference_path: Path = BASE_DIR / "data" / "reference" / "reference.csv"
    production_dir: Path = BASE_DIR / "data" / "production"

    # --- Artefatos ---------------------------------------------------------
    model_path: Path = BASE_DIR / "models" / "credit_model.joblib"
    reports_dir: Path = BASE_DIR / "reports"

    # --- Observabilidade ---------------------------------------------------
    # MLflow 3 descontinuou o backend de arquivos ("maintenance mode"); SQLite
    # é o backend local suportado e o único que habilita o Model Registry.
    mlflow_tracking_uri: str = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
    mlflow_artifact_root: Path = BASE_DIR / "mlartifacts"
    mlflow_experiment: str = "credit-scoring-observability"
    mlflow_registered_model: str = "credit-scoring-baseline"
    log_dir: Path = BASE_DIR / "logs"

    # --- Limiares de drift -------------------------------------------------
    # PSI: < 0.1 estável, 0.1–0.25 atenção, > 0.25 drift relevante (prática de
    # mercado em credit scoring, usada aqui como limiar de alerta).
    psi_alert_threshold: float = 0.25
    psi_warn_threshold: float = 0.10

    # KS e demais testes do Evidently: p-valor abaixo disso indica drift.
    drift_p_value: float = 0.05

    # Fração de colunas em drift que caracteriza drift do dataset inteiro.
    dataset_drift_share: float = 0.3

    # Queda de ROC AUC (em pontos absolutos) que caracteriza degradação do modelo.
    auc_degradation_threshold: float = 0.03

    # --- Modelo ------------------------------------------------------------
    decision_threshold: float = 0.5
    random_state: int = 42


settings = Settings()
