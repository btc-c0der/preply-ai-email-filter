FROM python:3.9-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos de requisitos
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Baixar dados do NLTK
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copiar código da aplicação
COPY . .

# Expor porta
EXPOSE 7860

# Comando para iniciar a aplicação
CMD ["python", "app.py"]
