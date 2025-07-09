#!/bin/bash

# Script para deploy da aplicação no Kubernetes

echo "🚀 Iniciando deploy do Email Filter App no Kubernetes..."

# Construir imagem Docker
echo "📦 Construindo imagem Docker..."
docker build -t email-filter:latest .

# Aplicar configurações do Kubernetes
echo "⚙️ Aplicando ConfigMap..."
kubectl apply -f k8s/configmap.yaml

echo "🚀 Aplicando Deployment..."
kubectl apply -f k8s/deployment.yaml

echo "🌐 Aplicando Ingress..."
kubectl apply -f k8s/ingress.yaml

# Verificar status
echo "✅ Verificando status do deployment..."
kubectl get deployments
kubectl get services
kubectl get ingress

echo "🎉 Deploy concluído!"
echo "📱 Acesse a aplicação em: http://email-filter.local"
echo "📊 Para verificar logs: kubectl logs -f deployment/email-filter-app"
