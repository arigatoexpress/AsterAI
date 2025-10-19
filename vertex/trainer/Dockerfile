FROM python:3.11-slim
WORKDIR /app
COPY train.py /app/train.py
RUN pip install --no-cache-dir google-cloud-bigquery xgboost joblib pandas scikit-learn
ENTRYPOINT ["python", "/app/train.py"]


