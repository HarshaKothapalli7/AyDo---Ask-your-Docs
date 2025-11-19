FROM python:3.12.12-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

FROM python:3.12.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

COPY frontend/ ./frontend

WORKDIR /app/frontend

ENTRYPOINT ["./entrypoint.sh"]