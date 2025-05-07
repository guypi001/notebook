FROM python:3.10-slim
WORKDIR /app

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif
COPY . .

EXPOSE 8000
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
