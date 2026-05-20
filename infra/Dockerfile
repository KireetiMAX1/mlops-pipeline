FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/        ./api/
COPY data/       ./data/
COPY training/   ./training/
COPY monitoring/ ./monitoring/

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV REGISTERED_MODEL_NAME=churn-predictor
ENV MODEL_STAGE=Production

EXPOSE 8000

CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
