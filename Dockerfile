# Imagem enxuta com Python
FROM python:3.12-slim

WORKDIR /app

# Instala as dependências primeiro (melhor cache de camadas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API, o pacote src (features) e o modelo treinado
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Sobe a API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
