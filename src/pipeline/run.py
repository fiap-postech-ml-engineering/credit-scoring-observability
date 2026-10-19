"""Execução ponta a ponta do projeto: do dado bruto aos relatórios de drift.

`uv run python -m src.pipeline.run` reproduz o projeto inteiro, em ordem:

1. **Preparo dos dados** — baixa o bruto, limpa e separa Referência / pool futuro.
2. **Treino do baseline** — só se ainda não houver modelo (use `--retrain` para forçar).
3. **Demonstração do contrato** — joga um lote deliberadamente corrompido contra
   o portão e comprova que a ingestão é bloqueada (entregável da Etapa 1).
4. **Simulação do drift** — gera os cinco lotes mensais de produção.
5. **Monitoramento** — roda o pipeline completo em cada lote.
6. **Consolidação** — grava o histórico de execuções e renderiza o dashboard HTML.

Tudo é idempotente: rodar duas vezes não duplica dado nem invalida relatório.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.contracts.bad_batch import build_bad_batch
from src.contracts.schemas import CreditReferenceSchema
from src.contracts.validate import save_result_json, validate_batch
from src.credit_scoring.dataset import prepare_datasets
from src.credit_scoring.train import load_model, train
from src.drift.simulate import SCENARIOS, build_all_batches
from src.observability.logging_setup import setup_logging
from src.pipeline.dashboard import render_dashboard
from src.pipeline.monitor import PipelineRun, monitor_batch, save_runs

logger = logging.getLogger(__name__)

BAD_BATCH_NAME = "lote_corrompido"


def demo_contract_block(reports_dir: Path | None = None) -> Path:
    """Força um lote inválido contra o contrato e persiste o laudo da rejeição.

    Este é o entregável explícito da Etapa 1: a prova de que o portão barra
    dado ruim. O lote carrega 11 defeitos distintos, um por regra do contrato.
    """
    reports_dir = reports_dir or settings.reports_dir
    bad_batch, manifest = build_bad_batch()

    result = validate_batch(bad_batch, CreditReferenceSchema, BAD_BATCH_NAME)

    if result.passed:  # pragma: no cover - indicaria contrato frouxo demais
        raise AssertionError("o lote corrompido passou pelo contrato — alguma regra foi afrouxada")

    caught = set(result.rules_violated)
    not_caught = [
        {"injection": defect.name, "rule": rule}
        for defect in manifest
        for rule in defect.expected_rules
        if rule not in caught
    ]

    laudo = save_result_json(result, reports_dir / f"{BAD_BATCH_NAME}__validation.json")

    # O manifesto fica ao lado do laudo: é o que permite auditar "foi injetado
    # X, o contrato pegou X" sem reler o código.
    manifest_path = reports_dir / f"{BAD_BATCH_NAME}__manifest.json"
    pd.DataFrame(
        [
            {
                "injection": defect.name,
                "description": defect.description,
                "expected_rules": " | ".join(defect.expected_rules),
                "caught": all(rule in caught for rule in defect.expected_rules),
            }
            for defect in manifest
        ]
    ).to_json(manifest_path, orient="records", indent=2, force_ascii=False)

    logger.info(
        "contrato bloqueou o lote corrompido",
        extra={
            "batch": BAD_BATCH_NAME,
            "n_rows": result.n_rows,
            "n_failures": result.n_failures,
            "rejection_rate": round(result.rejection_rate, 4),
            "rules_violated": len(result.rules_violated),
            "injections": len(manifest),
            "injections_not_caught": not_caught,
        },
    )

    if not_caught:  # pragma: no cover - regressão do contrato
        raise AssertionError(f"injeções não interceptadas pelo contrato: {not_caught}")

    return laudo


def run_all(retrain: bool = False, log_to_mlflow: bool = True) -> list[PipelineRun]:
    """Roda o projeto inteiro e devolve as execuções do pipeline."""
    log_file = setup_logging()
    logger.info("execução ponta a ponta iniciada", extra={"log_file": str(log_file)})

    # 1. Dados
    reference_path, _, cleaning = prepare_datasets(force=retrain)
    if cleaning:
        logger.info("estatísticas de limpeza", extra=cleaning.to_dict())

    # 2. Modelo
    if retrain or not settings.model_path.exists():
        train()
    bundle = load_model()
    logger.info(
        "modelo em produção carregado",
        extra={
            "candidate": bundle["candidate"],
            "threshold": bundle["threshold"],
            "baseline_roc_auc": round(bundle["validation_metrics"]["roc_auc"], 4),
        },
    )

    # 3. Portão de dados: a prova de que dado ruim é bloqueado
    demo_contract_block()

    # 4. Lotes de produção com drift
    batch_paths = build_all_batches()

    # 5. Monitoramento lote por lote
    reference = pd.read_csv(reference_path)
    runs: list[PipelineRun] = []
    for scenario in SCENARIOS:
        path = batch_paths[scenario.name]
        try:
            runs.append(
                monitor_batch(
                    batch_name=scenario.name,
                    batch_path=path,
                    reference=reference,
                    bundle=bundle,
                    log_to_mlflow=log_to_mlflow,
                    scenario_tags={"drift_kind": scenario.kind, **scenario.tags},
                )
            )
        except Exception:
            # Um lote problemático não deve impedir o monitoramento dos demais;
            # a falha já foi logada com stack trace em `_run_task`.
            logger.exception("lote abortado", extra={"batch": scenario.name})

    # 6. Consolidação
    runs_path = save_runs(runs)
    dashboard_path = render_dashboard(runs)

    logger.info(
        "execução ponta a ponta concluída",
        extra={
            "batches": len(runs),
            "com_alerta": sum(1 for r in runs if r.alerts),
            "runs_path": str(runs_path),
            "dashboard": str(dashboard_path),
        },
    )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="reconstrói os datasets e retreina o baseline do zero",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="não registra runs no MLflow (útil em CI)",
    )
    args = parser.parse_args()

    runs = run_all(retrain=args.retrain, log_to_mlflow=not args.no_mlflow)

    for run in runs:
        logger.info(
            "resumo do lote",
            extra={
                "batch": run.batch,
                "status": run.status,
                "alerts": run.alerts or ["nenhum"],
            },
        )


if __name__ == "__main__":
    main()
