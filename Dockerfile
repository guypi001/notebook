# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Lancer le serveur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
