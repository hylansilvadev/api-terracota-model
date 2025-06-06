# Use Python 3.11 slim como base
FROM python:3.11-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Copia os arquivos de requirements primeiro (para cache de layers)
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código da aplicação
COPY . .

# Expõe a porta 8000
EXPOSE 8000

# Comando para executar a aplicação
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]