FROM python:3.12.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt

FROM python:3.12.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY frontend/ ./frontend

WORKDIR /app/frontend

RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/frontend/entrypoint.sh"]