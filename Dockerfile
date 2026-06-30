FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project files
COPY . .

# Generate data + train on build (optional — remove for faster builds)
# RUN python -m src.data.generator
# RUN python -m src.models.trainer

# Expose ports: API (8000) + Dashboard (8501) + MLflow (5000)
EXPOSE 8000 8501 5000

# Default: run dashboard
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
