FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN pip install --no-cache-dir flask requests gunicorn

COPY app.py ./app.py

EXPOSE 8080

CMD ["gunicorn", "-w", "2", "-b", ":8080", "app:app"]


