# Dockerfile para ML Worker com suporte a GPU
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y     gcc     g++     && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements primeiro para aproveitar cache de camadas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte e artefatos
COPY src/ ./src/
COPY data/ ./data/
COPY artifacts/ ./artifacts/

# Comando para iniciar o worker
CMD ["python", "-m", "src.ml_worker"]
