import pandas as pd
import numpy as np
from faker import Faker
import random
import json
from datetime import datetime, timedelta

fake = Faker('pt_BR')

class EmailDataGenerator:
    def __init__(self):
        self.categories = {
            'importante': {
                'keywords': ['reunião', 'projeto', 'deadline', 'urgente', 'contrato', 'proposta', 'cliente', 'entrega', 'prazo'],
                'subjects': [
                    'Reunião de projeto amanhã às 14h',
                    'Proposta comercial - Prazo até sexta',
                    'Contrato para revisão urgente',
                    'Deadline do projeto movido',
                    'Cliente solicitou alterações',
                    'Entrega do relatório mensal',
                    'Revisão de código necessária',
                    'Aprovação de orçamento',
                    'Feedback sobre apresentação'
                ],
                'senders': ['gerente@empresa.com', 'cliente@cliente.com', 'projeto@empresa.com', 'rh@empresa.com']
            },
            'promocional': {
                'keywords': ['desconto', 'promoção', 'oferta', 'cupom', 'black friday', 'liquidação', 'cashback'],
                'subjects': [
                    '50% OFF em todos os produtos!',
                    'Black Friday chegou - Até 70% de desconto',
                    'Promoção relâmpago - Apenas hoje',
                    'Cupom de desconto exclusivo',
                    'Liquidação de fim de estoque',
                    'Cashback de 20% em compras',
                    'Oferta especial para você'
                ],
                'senders': ['promocoes@loja.com', 'ofertas@marketplace.com', 'newsletter@empresa.com']
            },
            'spam': {
                'keywords': ['ganhe', 'grátis', 'clique aqui', 'parabéns', 'sorteio', 'prêmio', 'dinheiro'],
                'subjects': [
                    'PARABÉNS! Você ganhou R$ 10.000!',
                    'Clique aqui para receber seu prêmio',
                    'Sorteio de iPhone - Você foi sorteado!',
                    'Ganhe dinheiro trabalhando em casa',
                    'Oferta GRÁTIS por tempo limitado',
                    'Você foi selecionado para um prêmio'
                ],
                'senders': ['naoresponda@spam.com', 'premio@sorteio.com', 'ganhe@dinheiro.com']
            },
            'pessoal': {
                'keywords': ['família', 'amigo', 'aniversário', 'convite', 'pessoal', 'férias'],
                'subjects': [
                    'Convite para aniversário no sábado',
                    'Fotos das férias em família',
                    'Que tal um almoço amanhã?',
                    'Parabéns pelo novo emprego!',
                    'Reunião de família no domingo',
                    'Lembrete: consulta médica'
                ],
                'senders': ['maria@gmail.com', 'joao@hotmail.com', 'ana@yahoo.com']
            }
        }
    
    def generate_email_content(self, category, subject):
        """Gera conteúdo do email baseado na categoria"""
        if category == 'importante':
            contents = [
                f"Olá,\n\nEspero que esteja bem. Gostaria de discutir sobre {subject.lower()}.\n\nPor favor, confirme sua disponibilidade.\n\nAtenciosamente,\n{fake.name()}",
                f"Prezado(a),\n\nSegue em anexo os documentos relacionados a {subject.lower()}.\n\nQualquer dúvida, estou à disposição.\n\nCordialmente,\n{fake.name()}",
                f"Bom dia,\n\nPrecisamos alinhar alguns pontos sobre {subject.lower()}.\n\nPodemos conversar hoje?\n\nAbraços,\n{fake.name()}"
            ]
        elif category == 'promocional':
            contents = [
                f"🎉 {subject} 🎉\n\nNão perca esta oportunidade única!\n\nVisite nossa loja online e aproveite.\n\nEquipe de Marketing",
                f"Olá!\n\n{subject}\n\nVálido até {fake.date_between(start_date='today', end_date='+30d')}.\n\nCompre agora!",
                f"Promoção especial para você!\n\n{subject}\n\nUse o cupom: DESCONTO20\n\nEquipe Comercial"
            ]
        elif category == 'spam':
            contents = [
                f"PARABÉNS!!!\n\n{subject}\n\nClique no link abaixo AGORA!\n\nwww.site-suspeito.com/premio",
                f"Você foi SELECIONADO!\n\n{subject}\n\nRápido! Oferta por tempo limitado!\n\nClique aqui: link-perigoso.com",
                f"🚨 URGENTE 🚨\n\n{subject}\n\nNão perca esta chance única!\n\nConfirme seus dados: site-falso.com"
            ]
        else:  # pessoal
            contents = [
                f"Oi!\n\n{subject}\n\nEspero sua resposta!\n\nBeijos,\n{fake.first_name()}",
                f"Olá querido(a),\n\nLembrete: {subject.lower()}\n\nNos vemos em breve!\n\nCom carinho,\n{fake.first_name()}",
                f"Oi!\n\nTudo bem? {subject.lower()}\n\nMe avisa se pode!\n\nAbraços,\n{fake.first_name()}"
            ]
        
        return random.choice(contents)
    
    def generate_dataset(self, num_emails=1000):
        """Gera dataset de emails simulados"""
        emails = []
        
        # Distribuição por categoria
        category_distribution = {
            'importante': 0.25,
            'promocional': 0.35,
            'spam': 0.20,
            'pessoal': 0.20
        }
        
        for i in range(num_emails):
            # Escolhe categoria baseada na distribuição
            category = np.random.choice(
                list(category_distribution.keys()),
                p=list(category_distribution.values())
            )
            
            # Gera dados do email
            subject = random.choice(self.categories[category]['subjects'])
            sender = random.choice(self.categories[category]['senders'])
            content = self.generate_email_content(category, subject)
            
            # Adiciona variação nos dados
            if random.random() < 0.1:  # 10% de chance de misturar categorias
                mixed_category = random.choice(list(self.categories.keys()))
                subject = f"{subject} - {random.choice(self.categories[mixed_category]['subjects'])}"
            
            # Define relevância (importante e pessoal = relevante)
            is_relevant = 1 if category in ['importante', 'pessoal'] else 0
            
            email = {
                'id': i + 1,
                'sender': sender,
                'subject': subject,
                'content': content,
                'category': category,
                'is_relevant': is_relevant,
                'timestamp': fake.date_time_between(start_date='-30d', end_date='now').isoformat()
            }
            
            emails.append(email)
        
        return pd.DataFrame(emails)
    
    def save_dataset(self, df, filename='emails_dataset.csv'):
        """Salva dataset em arquivo CSV"""
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Dataset salvo em {filename}")
        return filename

def main():
    # Gera dataset de emails
    generator = EmailDataGenerator()
    
    # Gera 1000 emails simulados
    print("Gerando dataset de emails simulados...")
    df = generator.generate_dataset(1000)
    
    # Salva dataset
    filename = generator.save_dataset(df)
    
    # Mostra estatísticas
    print("\n📊 Estatísticas do Dataset:")
    print(f"Total de emails: {len(df)}")
    print("\nDistribuição por categoria:")
    print(df['category'].value_counts())
    print("\nDistribuição por relevância:")
    print(df['is_relevant'].value_counts().rename({0: 'Não relevante', 1: 'Relevante'}))
    
    # Mostra alguns exemplos
    print("\n📧 Exemplos de emails gerados:")
    for category in df['category'].unique():
        print(f"\n--- {category.upper()} ---")
        example = df[df['category'] == category].iloc[0]
        print(f"De: {example['sender']}")
        print(f"Assunto: {example['subject']}")
        print(f"Conteúdo: {example['content'][:100]}...")

if __name__ == "__main__":
    main()
