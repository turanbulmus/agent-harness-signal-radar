FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN groupadd --system radar && useradd --system --gid radar --create-home radar

COPY pyproject.toml LICENSE README.md ./
COPY harness_signal_radar ./harness_signal_radar
COPY eval ./eval
COPY scripts ./scripts

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[gcp]" && \
    chown -R radar:radar /app

USER radar

EXPOSE 8080

CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8080", "."]
