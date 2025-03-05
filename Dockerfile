FROM python:3.8-slim

WORKDIR /app

COPY ../requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ../src /app/src
COPY ..models /app/models

EXPOSE 5000

CMD ["python", "src/fraud_detection_api.py"]