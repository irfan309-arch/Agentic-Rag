#!/bin/bash
set -e

echo "Starting FastAPI on :8000..."

uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 &

sleep 2

echo "Starting Streamlit on port ${PORT:-8501}..."

exec streamlit run app.py \
    --server.port "${PORT:-8501}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false