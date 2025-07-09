# 🤖 Filtro de E-mails com IA - Projeto Educacional

## 🎯 Sobre o Projeto

Este projeto implementa um **sistema de filtragem de e-mails usando Inteligência Artificial**, desenvolvido para fins educacionais no curso de **IA Básica**. O sistema classifica e-mails como **relevantes** ou **não relevantes** usando técnicas de **Processamento de Linguagem Natural (NLP)** e **Machine Learning**.

### 📚 Conceitos Abordados

- **Inteligência Artificial para usuários**
- **Ética e responsabilidade em IA**
- **Legislação (LGPD, GDPR)**
- **Processamento de Linguagem Natural**
- **Classificação de texto**
- **Interface web com Gradio**
- **Deploy com Kubernetes**

## 🛠️ Tecnologias Utilizadas

- **Python 3.9+**
- **Gradio** - Interface web interativa
- **Scikit-learn** - Machine Learning
- **NLTK** - Processamento de linguagem natural
- **Pandas/NumPy** - Manipulação de dados
- **Docker** - Containerização
- **Kubernetes** - Orquestração

## 📁 Estrutura do Projeto

```
preply-ai-email-filter/
├── app.py                    # Aplicação principal com Gradio
├── email_classifier.py       # Modelo de classificação
├── data_generator.py         # Gerador de dados simulados
├── smtp_integration.py       # Integração SMTP/IMAP
├── test_smtp_integration.py  # Testes SMTP
├── requirements.txt          # Dependências Python
├── Dockerfile               # Imagem Docker
├── setup.sh                 # Script de configuração
├── deploy.sh                # Script de deploy K8s
├── SMTP_CONFIG.md           # Configurações SMTP
├── k8s/                     # Configurações Kubernetes
│   ├── deployment.yaml
│   ├── ingress.yaml
│   └── configmap.yaml
└── README.md                # Este arquivo
```

## 🚀 Como Usar

### 1. Configuração Local

```bash
# Clonar o repositório
git clone <repository-url>
cd preply-ai-email-filter

# Executar setup
chmod +x setup.sh
./setup.sh

# Iniciar aplicação
python app.py
```

### 2. Usando Docker

```bash
# Construir imagem
docker build -t email-filter:latest .

# Executar container
docker run -p 7860:7860 email-filter:latest
```

### 3. Deploy no Kubernetes

```bash
# Executar deploy
chmod +x deploy.sh
./deploy.sh

# Verificar status
kubectl get pods
kubectl get services
```

## 🎮 Funcionalidades

### 🔧 Treinamento
- Geração de dataset simulado (1000+ e-mails)
- Treinamento do modelo de IA
- Visualização de métricas

### 📧 Classificação Individual
- Interface para testar e-mails individuais
- Exemplos predefinidos
- Visualização de confiança

### 📊 Análise em Lote
- Upload de arquivos CSV
- Processamento de múltiplos e-mails
- Relatórios detalhados

### 📈 Métricas
- Estatísticas do modelo
- Palavras mais importantes
- Matriz de confusão

### 📨 Integração SMTP/IMAP
- **Conexão com servidores de email:** Gmail, Outlook, Yahoo, iCloud
- **Download automático de emails** de pastas específicas
- **Classificação em tempo real** com modelo de IA
- **Configuração segura** com App Passwords
- **Análise estatística** dos emails baixados
- **Export para CSV** dos resultados

## 🧠 Como Funciona

### 1. Geração de Dados
```python
# Categorias de e-mails
categorias = {
    'importante': ['reunião', 'projeto', 'deadline'],
    'promocional': ['desconto', 'promoção', 'oferta'],
    'spam': ['ganhe', 'grátis', 'clique aqui'],
    'pessoal': ['família', 'amigo', 'convite']
}
```

### 2. Processamento de Texto
- Tokenização
- Remoção de stopwords
- Stemming (português)
- Vetorização TF-IDF

### 3. Classificação
- Modelo: Naive Bayes
- Features: TF-IDF de assunto + conteúdo
- Saída: Relevante/Não relevante + confiança

## 📊 Métricas do Modelo

O modelo atinge aproximadamente:
- **Acurácia**: ~85-90%
- **Precisão**: ~85-90%
- **Recall**: ~85-90%
- **F1-Score**: ~85-90%

## 🎓 Aspectos Educacionais

### Ética em IA
- Transparência nas decisões
- Explicabilidade do modelo
- Viés algorítmico
- Responsabilidade humana

### Legislação
- LGPD (Lei Geral de Proteção de Dados)
- GDPR (General Data Protection Regulation)
- AI Act Europeu
- Direito à explicação

### Boas Práticas
- Dados sintéticos para treinamento
- Validação cruzada
- Métricas de performance
- Monitoramento contínuo

## 🔧 Configuração Avançada

### Variáveis de Ambiente
```bash
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
MODEL_MAX_FEATURES=5000
DATASET_SIZE=1000
```

### Kubernetes Resources
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

## 🐛 Solução de Problemas

### Erro de NLTK
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Erro de Memória
- Reduzir número de features no TF-IDF
- Diminuir tamanho do dataset
- Aumentar recursos do container

### Erro de Encoding
- Verificar encoding UTF-8 nos arquivos CSV
- Usar parâmetro `encoding='utf-8'` no pandas

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📝 Licença

Este projeto é para fins educacionais e está sob licença MIT.

## 👨‍🏫 Autor

Desenvolvido para o curso de **IA Básica** - Projeto prático de filtragem de e-mails com IA.

---

### 🎯 Próximos Passos

- [ ] Implementar mais algoritmos de classificação
- [ ] Adicionar suporte a múltiplos idiomas
- [ ] Melhorar interface com mais visualizações
- [ ] Implementar API REST
- [ ] Adicionar autenticação
- [ ] Monitoramento e logs avançados

---

**🚀 Pronto para começar? Execute `./setup.sh` e comece a experimentar com IA!**