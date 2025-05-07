FROM python:3.10-slim

# (facultatif mais conseillé) paquets build essentiels minima
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# ↑ version CPU pour éviter 3 Go de librairies CUDA inutiles :contentReference[oaicite:5]{index=5}

# Code applicatif
COPY . .

EXPOSE 8000
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
# `${PORT}` respecte la variable que Render fixe au runtime :contentReference[oaicite:6]{index=6}
