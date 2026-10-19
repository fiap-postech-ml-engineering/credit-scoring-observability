# Estágio base: ferramentas comuns aos demais estágios.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

WORKDIR /app

# Estágio builder: resolve e instala só as dependências de produção,
# nas versões exatas travadas no lockfile.
FROM base AS builder

COPY pyproject.toml uv.lock LICENSE README.md ./

RUN uv sync --frozen --no-dev

COPY src ./src

# Estágio final: nenhuma ferramenta de build, cache do uv ou dependência de
# desenvolvimento chega aqui — só o ambiente virtual resolvido e o código.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Usuário não-root: se a aplicação for comprometida, o processo não tem
# privilégios de root dentro do container.
RUN addgroup --system app \
    && adduser --system --ingroup app app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY pyproject.toml README.md LICENSE ./

RUN mkdir -p /app/data /app/models /app/reports /app/logs /app/mlartifacts \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"]

CMD ["uvicorn", "src.api.api:api", "--host", "0.0.0.0", "--port", "8000"]
