"""Logging estruturado em JSON, centralizando os logs do pipeline (Etapa 3).

Todo módulo do projeto loga com `logger.info("mensagem", extra={...})`. O
formatter abaixo transforma isso em uma linha JSON por evento, com os campos
de `extra` promovidos ao topo do objeto — formato que qualquer coletor
(Loki, Datadog, CloudWatch) ingere sem parsing customizado.

Os logs vão para stdout **e** para `logs/pipeline.jsonl`. O arquivo é o que
alimenta o relatório consolidado de execuções em `src/pipeline/report.py`:
é dele que saem "status de execução das tarefas" e "quantidade de alertas"
documentados como métricas de saúde operacional.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any

from src.config.settings import settings

# Atributos que o logging coloca em todo LogRecord. Tudo que não está aqui
# veio do `extra=` da chamada e é, portanto, contexto do evento.
_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def safe_extra(payload: dict[str, Any]) -> dict[str, Any]:
    """Prepara um dicionário para ser passado como `extra=` ao logging.

    O `logging` levanta `KeyError` se o `extra` tentar sobrescrever um atributo
    reservado do `LogRecord` — e `name`, `module` e `filename` são nomes de
    campo perfeitamente naturais num payload de domínio. Em vez de proibir
    esses nomes no resto do código, prefixamos os colidentes com `ctx_`.
    """
    return {(f"ctx_{k}" if k in _RESERVED else k): v for k, v in payload.items()}


class JsonFormatter(logging.Formatter):
    """Serializa cada LogRecord como uma linha JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    json_stdout: bool = False,
) -> Path:
    """Configura o logging do processo e devolve o caminho do arquivo de log.

    `json_stdout=False` mantém o terminal legível para humanos e reserva o
    JSON para o arquivo; `True` emite JSON nos dois (modo container).
    """
    log_file = log_file or settings.log_dir / "pipeline.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(
        JsonFormatter()
        if json_stdout
        else logging.Formatter("%(levelname)-8s %(name)s: %(message)s")
    )
    root.addHandler(stream)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    # O MLflow é extremamente verboso no nível INFO; só erros interessam aqui.
    for noisy in ("mlflow", "py4j", "urllib3", "matplotlib", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return log_file


def read_events(log_file: Path | None = None) -> list[dict[str, Any]]:
    """Lê os eventos JSON já registrados — base do relatório de execuções."""
    log_file = log_file or settings.log_dir / "pipeline.jsonl"
    if not log_file.exists():
        return []

    events: list[dict[str, Any]] = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
