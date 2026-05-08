FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV BASE_DIR=/app
ENV UPLOAD_DIR=/app/data/uploads
ENV RESULTS_DIR=/app/data/api_results

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

