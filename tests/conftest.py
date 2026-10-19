"""Fixtures compartilhadas pelos testes.

A estratégia aqui é não depender de rede nem de arquivos grandes: um
`synthetic_reference` determinístico cobre quase tudo. Os testes que exigem o
dataset real ou o modelo treinado são marcados como `slow` e pulados quando os
artefatos não existem, para que `make test` continue rodando num clone limpo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.settings import FEATURES, TARGET, settings


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def synthetic_reference() -> pd.DataFrame:
    """Lote sintético válido: passa no contrato por construção.

    Os valores ficam deliberadamente longe das bordas do contrato, para que um
    teste que queira violar uma regra precise fazê-lo de propósito.
    """
    generator = np.random.default_rng(42)
    n = 600

    data = {
        "RevolvingUtilizationOfUnsecuredLines": generator.uniform(0.0, 1.0, n).round(4),
        "age": generator.integers(25, 75, n),
        "NumberOfTime30-59DaysPastDueNotWorse": generator.integers(0, 4, n),
        "DebtRatio": generator.uniform(0.0, 2.0, n).round(4),
        "MonthlyIncome": generator.uniform(1500.0, 15000.0, n).round(2),
        "NumberOfOpenCreditLinesAndLoans": generator.integers(1, 20, n),
        "NumberOfTimes90DaysLate": generator.integers(0, 3, n),
        "NumberRealEstateLoansOrLines": generator.integers(0, 5, n),
        "NumberOfTime60-89DaysPastDueNotWorse": generator.integers(0, 3, n),
        "NumberOfDependents": generator.integers(0, 5, n),
    }
    frame = pd.DataFrame(data)

    # Alvo correlacionado com o risco, para que o modelo tenha o que aprender.
    risk = (
        0.6 * frame["RevolvingUtilizationOfUnsecuredLines"]
        + 0.3 * frame["DebtRatio"]
        + 0.25 * frame["NumberOfTimes90DaysLate"]
        - 0.012 * frame["age"]
    )
    probability = 1 / (1 + np.exp(-(risk - 0.2)))
    frame[TARGET] = (generator.random(n) < probability).astype(int)

    # Garante as duas classes mesmo num sorteio azarado.
    frame.loc[frame.index[:30], TARGET] = 1
    frame.loc[frame.index[30:60], TARGET] = 0

    # Remove duplicatas que o sorteio possa ter produzido: o contrato as proíbe.
    return frame[[TARGET, *FEATURES]].drop_duplicates().reset_index(drop=True)


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "reports"
    directory.mkdir()
    return directory


@pytest.fixture
def trained_bundle(synthetic_reference: pd.DataFrame, tmp_path: Path) -> dict:
    """Treina um modelo minúsculo sobre o lote sintético, sem tocar no MLflow."""
    from src.credit_scoring.train import train

    model_path = tmp_path / "model.joblib"
    train(reference=synthetic_reference, model_path=model_path, log_to_mlflow=False)

    import joblib

    return joblib.load(model_path)


def requires_artifacts() -> bool:
    """Os artefatos do pipeline real existem neste clone?"""
    return settings.model_path.exists() and settings.reference_path.exists()


skip_without_artifacts = pytest.mark.skipif(
    not requires_artifacts(),
    reason="exige `make pipeline` — artefatos do pipeline real não encontrados",
)
