#!/bin/bash

# Script para desenvolvimento local

echo "🔧 Configurando ambiente de desenvolvimento..."

# Instalar dependências
echo "📦 Instalando dependências..."
pip install -r requirements.txt

# Gerar dados de exemplo
echo "📊 Gerando dados de exemplo..."
python data_generator.py

# Treinar modelo
echo "🤖 Treinando modelo..."
python email_classifier.py

echo "✅ Ambiente configurado!"
echo "🚀 Para iniciar a aplicação: python app.py"
