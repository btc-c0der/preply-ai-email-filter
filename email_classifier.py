import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class EmailClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
        self.model = MultinomialNB()
        self.is_trained = False
        
        # Portuguese stopwords (basic set)
        self.stop_words = set(['a', 'o', 'e', 'é', 'de', 'do', 'da', 'para', 'com', 'em', 'por', 'se', 'no', 'na', 'um', 'uma', 'que', 'não', 'mais', 'como', 'mas', 'ao', 'sua', 'seu', 'ou', 'ser', 'ter', 'todo', 'todos', 'esta', 'este', 'isso', 'aqui', 'ali', 'já', 'foi', 'são', 'muito', 'bem', 'pode', 'vai', 'vou', 'até', 'quando', 'onde', 'quem', 'qual', 'pela', 'pelo', 'nos', 'nas', 'dos', 'das', 'seus', 'suas', 'meu', 'minha', 'me', 'te', 'lhe', 'nos', 'vos', 'lhes'])
    
    def preprocess_text(self, text):
        """Pré-processa texto para classificação"""
        if pd.isna(text):
            return ""
        
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^a-záàâãéèêíìîóòôõúùûç\s]', '', text)
        
        # Tokenizar simples
        tokens = text.split()
        
        # Remover stopwords
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def prepare_features(self, df):
        """Prepara features combinando assunto e conteúdo"""
        # Combina assunto e conteúdo
        df['combined_text'] = df['subject'].fillna('') + ' ' + df['content'].fillna('')
        
        # Pré-processa o texto
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        return df
    
    def train(self, df):
        """Treina o modelo de classificação"""
        print("🔄 Preparando dados para treinamento...")
        
        # Prepara features
        df = self.prepare_features(df)
        
        # Divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['is_relevant'], 
            test_size=0.2, 
            random_state=42,
            stratify=df['is_relevant']
        )
        
        print("🔄 Vetorizando texto...")
        # Vetoriza o texto
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print("🔄 Treinando modelo...")
        # Treina o modelo
        self.model.fit(X_train_vec, y_train)
        
        # Avalia o modelo
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Modelo treinado com sucesso!")
        print(f"📊 Acurácia: {accuracy:.3f}")
        
        # Relatório detalhado
        print("\n📋 Relatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=['Não Relevante', 'Relevante']))
        
        self.is_trained = True
        
        # Salva estatísticas de treino
        self.train_stats = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, target_names=['Não Relevante', 'Relevante'], output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return accuracy
    
    def predict(self, subject, content):
        """Prediz se um email é relevante"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        # Prepara o texto
        combined_text = f"{subject} {content}"
        processed_text = self.preprocess_text(combined_text)
        
        # Vetoriza
        text_vec = self.vectorizer.transform([processed_text])
        
        # Prediz
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        return {
            'is_relevant': bool(prediction),
            'confidence': float(max(probability)),
            'probabilities': {
                'not_relevant': float(probability[0]),
                'relevant': float(probability[1])
            }
        }
    
    def get_feature_importance(self, top_n=20):
        """Retorna as palavras mais importantes para classificação"""
        if not self.is_trained:
            return []
        
        # Pega os coeficientes do modelo
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]
        
        # Ordena por importância
        feature_importance = list(zip(feature_names, coefficients))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return feature_importance[:top_n]
    
    def save_model(self, filename='email_classifier.pkl'):
        """Salva o modelo treinado"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'train_stats': self.train_stats
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ Modelo salvo em {filename}")
    
    def load_model(self, filename='email_classifier.pkl'):
        """Carrega modelo treinado"""
        model_data = joblib.load(filename)
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.train_stats = model_data['train_stats']
        self.is_trained = True
        print(f"✅ Modelo carregado de {filename}")
    
    def plot_confusion_matrix(self):
        """Plota matriz de confusão"""
        if not self.is_trained:
            return None
        
        plt.figure(figsize=(8, 6))
        cm = np.array(self.train_stats['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Não Relevante', 'Relevante'],
                   yticklabels=['Não Relevante', 'Relevante'])
        plt.title('Matriz de Confusão')
        plt.ylabel('Valor Real')
        plt.xlabel('Predição')
        plt.tight_layout()
        return plt

def main():
    # Carrega dataset
    print("📂 Carregando dataset...")
    df = pd.read_csv('emails_dataset.csv')
    
    # Cria e treina classificador
    classifier = EmailClassifier()
    accuracy = classifier.train(df)
    
    # Salva modelo
    classifier.save_model()
    
    # Mostra palavras mais importantes
    print("\n🔍 Palavras mais importantes para classificação:")
    important_features = classifier.get_feature_importance(10)
    for feature, importance in important_features:
        print(f"{feature}: {importance:.3f}")
    
    # Testa com alguns exemplos
    print("\n🧪 Testando com exemplos:")
    test_emails = [
        ("Reunião urgente amanhã", "Precisamos discutir o projeto. Por favor confirme presença."),
        ("50% OFF em tudo!", "Promoção imperdível! Corra para a loja virtual."),
        ("Convite aniversário", "Oi! Vai rolar festa no sábado. Você vem?"),
        ("PARABÉNS! Você ganhou!", "Clique aqui para resgatar seu prêmio de R$ 10.000!")
    ]
    
    for subject, content in test_emails:
        result = classifier.predict(subject, content)
        relevance = "RELEVANTE" if result['is_relevant'] else "NÃO RELEVANTE"
        print(f"📧 '{subject}' → {relevance} (confiança: {result['confidence']:.3f})")

if __name__ == "__main__":
    main()
