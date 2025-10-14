FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run expects the service to listen on $PORT (default 8080)
ENV PORT=8080

# Expose port for local runs (optional; Cloud Run ignores EXPOSE)
EXPOSE 8080

# Run the Streamlit dashboard (serves HTTP for Cloud Run)
# Use exec form with shell to ensure environment variables are expanded
CMD ["/bin/bash", "-c", "streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0"]
