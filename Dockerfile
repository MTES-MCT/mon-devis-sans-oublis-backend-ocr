FROM python:3.13-slim

RUN pip install --upgrade pip

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    clamav clamav-daemon \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .

CMD ["gunicorn", "-b", "0.0.0.0:80", "api:app"]
