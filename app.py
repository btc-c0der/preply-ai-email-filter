import gradio as gr
import pandas as pd
import numpy as np
from email_classifier import EmailClassifier
from data_generator import EmailDataGenerator
from smtp_integration import SMTPEmailFetcher, EmailProcessor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import json

class EmailFilterApp:
    def __init__(self):
        self.classifier = EmailClassifier()
        self.generator = EmailDataGenerator()
        self.smtp_fetcher = SMTPEmailFetcher()
        self.email_processor = EmailProcessor(self.classifier)
        self.is_model_loaded = False
        self.dataset = None
        self.downloaded_emails = []
        
        # Tenta carregar modelo existente
        if os.path.exists('email_classifier.pkl'):
            try:
                self.classifier.load_model()
                self.email_processor.classifier = self.classifier
                self.is_model_loaded = True
                print("✅ Modelo carregado automaticamente")
            except Exception as e:
                print(f"⚠️ Erro ao carregar modelo: {e}")
    
    def generate_and_train(self, num_emails=1000):
        """Gera dataset e treina modelo"""
        try:
            # Gera dataset
            self.dataset = self.generator.generate_dataset(num_emails)
            
            # Treina modelo
            accuracy = self.classifier.train(self.dataset)
            
            # Salva modelo
            self.classifier.save_model()
            self.is_model_loaded = True
            
            # Estatísticas
            stats = self.dataset['category'].value_counts().to_dict()
            relevance_stats = self.dataset['is_relevant'].value_counts().to_dict()
            
            result = f"""
✅ **Modelo treinado com sucesso!**

📊 **Estatísticas do Dataset:**
- Total de emails: {len(self.dataset)}
- Acurácia: {accuracy:.3f}

📈 **Distribuição por categoria:**
{chr(10).join([f"- {k}: {v}" for k, v in stats.items()])}

🎯 **Distribuição por relevância:**
- Relevantes: {relevance_stats.get(1, 0)}
- Não relevantes: {relevance_stats.get(0, 0)}
"""
            
            return result, self.create_category_chart(), self.create_relevance_chart()
            
        except Exception as e:
            return f"❌ Erro: {str(e)}", None, None
    
    def classify_email(self, subject, content):
        """Classifica um email"""
        if not self.is_model_loaded:
            return "⚠️ Modelo não carregado. Gere e treine o modelo primeiro."
        
        try:
            result = self.classifier.predict(subject, content)
            
            relevance = "🟢 **RELEVANTE**" if result['is_relevant'] else "🔴 **NÃO RELEVANTE**"
            confidence = result['confidence']
            
            response = f"""
{relevance}

📊 **Confiança:** {confidence:.3f}

📈 **Probabilidades:**
- Relevante: {result['probabilities']['relevant']:.3f}
- Não relevante: {result['probabilities']['not_relevant']:.3f}
"""
            
            return response
            
        except Exception as e:
            return f"❌ Erro na classificação: {str(e)}"
    
    def analyze_batch(self, file):
        """Analisa um lote de emails de um arquivo CSV"""
        if not self.is_model_loaded:
            return "⚠️ Modelo não carregado. Gere e treine o modelo primeiro."
        
        try:
            # Lê arquivo CSV
            df = pd.read_csv(file.name)
            
            # Verifica se tem as colunas necessárias
            if 'subject' not in df.columns or 'content' not in df.columns:
                return "❌ Arquivo deve ter colunas 'subject' e 'content'"
            
            # Classifica emails
            results = []
            for _, row in df.iterrows():
                result = self.classifier.predict(row['subject'], row['content'])
                results.append({
                    'subject': row['subject'],
                    'is_relevant': result['is_relevant'],
                    'confidence': result['confidence']
                })
            
            results_df = pd.DataFrame(results)
            
            # Estatísticas
            total = len(results_df)
            relevant = results_df['is_relevant'].sum()
            not_relevant = total - relevant
            avg_confidence = results_df['confidence'].mean()
            
            analysis = f"""
📊 **Análise do Lote:**
- Total de emails: {total}
- Relevantes: {relevant}
- Não relevantes: {not_relevant}
- Confiança média: {avg_confidence:.3f}
"""
            
            return analysis, results_df
            
        except Exception as e:
            return f"❌ Erro na análise: {str(e)}", None
    
    def get_model_stats(self):
        """Retorna estatísticas do modelo"""
        if not self.is_model_loaded:
            return "⚠️ Modelo não carregado."
        
        stats = self.classifier.train_stats
        
        return f"""
📊 **Estatísticas do Modelo:**

🎯 **Acurácia:** {stats['accuracy']:.3f}

📈 **Métricas por Classe:**
- **Não Relevante:**
  - Precisão: {stats['classification_report']['Não Relevante']['precision']:.3f}
  - Recall: {stats['classification_report']['Não Relevante']['recall']:.3f}
  - F1-Score: {stats['classification_report']['Não Relevante']['f1-score']:.3f}

- **Relevante:**
  - Precisão: {stats['classification_report']['Relevante']['precision']:.3f}
  - Recall: {stats['classification_report']['Relevante']['recall']:.3f}
  - F1-Score: {stats['classification_report']['Relevante']['f1-score']:.3f}
"""
    
    def get_feature_importance(self):
        """Retorna palavras mais importantes"""
        if not self.is_model_loaded:
            return "⚠️ Modelo não carregado."
        
        features = self.classifier.get_feature_importance(15)
        
        result = "🔍 **Palavras mais importantes para classificação:**\n\n"
        for feature, importance in features:
            result += f"- **{feature}**: {importance:.3f}\n"
        
        return result
    
    def create_category_chart(self):
        """Cria gráfico de distribuição por categoria"""
        if self.dataset is None:
            return None
        
        fig = px.pie(
            values=self.dataset['category'].value_counts().values,
            names=self.dataset['category'].value_counts().index,
            title="Distribuição por Categoria"
        )
        return fig
    
    def create_relevance_chart(self):
        """Cria gráfico de distribuição por relevância"""
        if self.dataset is None:
            return None
        
        relevance_counts = self.dataset['is_relevant'].value_counts()
        relevance_counts.index = ['Não Relevante', 'Relevante']
        
        fig = px.bar(
            x=relevance_counts.index,
            y=relevance_counts.values,
            title="Distribuição por Relevância",
            color=relevance_counts.index,
            color_discrete_map={'Relevante': 'green', 'Não Relevante': 'red'}
        )
        return fig
    
    def test_smtp_connection(self, provider, server, port, username, password, use_ssl):
        """Testa conexão SMTP"""
        try:
            port = int(port) if port else 993
            
            if provider and provider != "custom":
                success, message = self.smtp_fetcher.connect_with_provider(provider, username, password)
            else:
                success, message = self.smtp_fetcher.test_connection(server, port, username, password, use_ssl)
            
            if success:
                self.smtp_fetcher.disconnect()
                return f"✅ {message}"
            else:
                return f"❌ {message}"
                
        except Exception as e:
            return f"❌ Erro: {str(e)}"
    
    def connect_smtp(self, provider, server, port, username, password, use_ssl):
        """Conecta ao servidor SMTP"""
        try:
            port = int(port) if port else 993
            
            if provider and provider != "custom":
                success, message = self.smtp_fetcher.connect_with_provider(provider, username, password)
            else:
                success, message = self.smtp_fetcher.connect(server, port, username, password, use_ssl)
            
            if success:
                folders = self.smtp_fetcher.list_folders()
                folder_list = "\n".join([f"📁 {folder}" for folder in folders[:10]])  # Mostra primeiras 10 pastas
                
                return f"✅ {message}\n\n📂 **Pastas disponíveis:**\n{folder_list}", gr.update(choices=folders, value="INBOX")
            else:
                return f"❌ {message}", gr.update(choices=[], value=None)
                
        except Exception as e:
            return f"❌ Erro: {str(e)}", gr.update(choices=[], value=None)
    
    def disconnect_smtp(self):
        """Desconecta do servidor SMTP"""
        if self.smtp_fetcher.is_connected:
            self.smtp_fetcher.disconnect()
            return "Status: Desconectado", gr.update(choices=["INBOX"], value="INBOX")
        return "Status: Já desconectado", gr.update(choices=["INBOX"], value="INBOX")
    
    def download_emails(self, folder, limit, days_back):
        """Baixa emails do servidor"""
        if not self.smtp_fetcher.is_connected:
            return "❌ Não conectado ao servidor. Conecte primeiro.", None, None
        
        try:
            self.downloaded_emails = self.smtp_fetcher.fetch_emails(folder, limit, days_back)
            
            # Processa os emails para exibição
            if self.downloaded_emails:
                # Classifica emails se modelo estiver carregado
                if self.is_model_loaded:
                    self.downloaded_emails = self.email_processor.classify_emails(self.downloaded_emails)
                
                # Prepara dados para tabela
                table_data = []
                for email in self.downloaded_emails:
                    # Trunca textos longos para exibição
                    sender = email.get('sender', '')[:30]
                    subject = email.get('subject', '')[:50]
                    category = email.get('category', 'não classificado')
                    is_relevant = "✅" if email.get('is_relevant') else "❌"
                    date = email.get('timestamp', '')[:10]  # apenas YYYY-MM-DD
                    
                    table_data.append([sender, subject, category, is_relevant, date])
                
                # Estatísticas
                stats = self.email_processor.get_statistics(self.downloaded_emails)
                stats_text = self._format_email_stats(stats)
                
                return f"✅ {len(self.downloaded_emails)} emails baixados com sucesso!", table_data, stats_text
            else:
                return "⚠️ Nenhum email encontrado no período especificado.", [], "Nenhum email baixado"
                
        except Exception as e:
            return f"❌ Erro ao baixar emails: {str(e)}", None, None
    
    def classify_downloaded_emails(self):
        """Classifica emails baixados"""
        if not self.downloaded_emails:
            return "⚠️ Nenhum email baixado para classificar.", None
        
        if not self.is_model_loaded:
            return "⚠️ Modelo não carregado. Treine o modelo primeiro.", None
        
        try:
            self.downloaded_emails = self.email_processor.classify_emails(self.downloaded_emails)
            
            # Prepara dados para tabela
            table_data = []
            for email in self.downloaded_emails:
                sender = email.get('sender', '')[:30]
                subject = email.get('subject', '')[:50]
                category = email.get('category', 'não classificado')
                is_relevant = "✅" if email.get('is_relevant') else "❌"
                date = email.get('timestamp', '')[:10]
                
                table_data.append([sender, subject, category, is_relevant, date])
            
            return "✅ Emails classificados com sucesso!", table_data
                
        except Exception as e:
            return f"❌ Erro na classificação: {str(e)}", None
    
    def save_downloaded_emails(self):
        """Salva emails baixados em CSV"""
        if not self.downloaded_emails:
            return "⚠️ Nenhum email baixado para salvar."
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"downloaded_emails_{timestamp}.csv"
            
            result = self.email_processor.save_to_csv(self.downloaded_emails, filename)
            return f"✅ {result}"
                
        except Exception as e:
            return f"❌ Erro ao salvar: {str(e)}"
    
    def _format_email_stats(self, stats):
        """Formata estatísticas de emails para exibição"""
        if not stats:
            return "Nenhuma estatística disponível"
        
        text = f"""
📊 **Estatísticas dos Emails:**
- Total: {stats.get('total', 0)}
- Relevantes: {stats.get('relevant', 0)} ({stats.get('relevant_pct', 0):.1f}%)
- Não relevantes: {stats.get('not_relevant', 0)} ({stats.get('not_relevant_pct', 0):.1f}%)

📁 **Distribuição por pastas:**
{chr(10).join([f"- {k}: {v}" for k, v in stats.get('folders', {}).items()[:5]])}

📋 **Distribuição por categorias:**
{chr(10).join([f"- {k}: {v}" for k, v in stats.get('categories', {}).items()])}

📅 **Período:** {stats.get('date_range', 'N/A')}
"""
        return text
    
    def create_smtp_charts(self):
        """Cria gráficos para emails SMTP"""
        if not self.downloaded_emails:
            # Retorna gráficos vazios
            empty_fig1 = px.bar(x=[], y=[])
            empty_fig2 = px.bar(x=[], y=[])
            return empty_fig1, empty_fig2
        
        # DataFrame para análise
        emails_df = pd.DataFrame(self.downloaded_emails)
        
        # Gráfico de categorias
        if 'category' in emails_df.columns:
            category_counts = emails_df['category'].value_counts().reset_index()
            category_counts.columns = ['Categoria', 'Contagem']
            
            cat_fig = px.bar(
                category_counts,
                x='Categoria',
                y='Contagem',
                title="Distribuição por Categoria",
                color='Categoria'
            )
        else:
            cat_fig = px.bar(x=[], y=[])
        
        # Gráfico de relevância
        if 'is_relevant' in emails_df.columns:
            emails_df['Relevância'] = emails_df['is_relevant'].apply(
                lambda x: 'Relevante' if x else 'Não Relevante'
            )
            
            relevance_counts = emails_df['Relevância'].value_counts().reset_index()
            relevance_counts.columns = ['Relevância', 'Contagem']
            
            rel_fig = px.pie(
                relevance_counts,
                names='Relevância',
                values='Contagem',
                title="Distribuição por Relevância",
                color='Relevância',
                color_discrete_map={'Relevante': 'green', 'Não Relevante': 'red'}
            )
        else:
            rel_fig = px.pie(names=[], values=[])
        
        return cat_fig, rel_fig
    
    def create_interface(self):
        """Cria interface Gradio"""
        with gr.Blocks(title="🤖 Filtro de E-mails com IA", theme=gr.themes.Soft()) as app:
            gr.Markdown("""
            # 🤖 Filtro de E-mails com IA
            ## Projeto Educacional - Inteligência Artificial Básica
            
            Este sistema usa **Inteligência Artificial** para classificar emails como **relevantes** ou **não relevantes**.
            
            ### 🎯 Funcionalidades:
            - Geração de dados simulados
            - Treinamento de modelo de IA
            - Classificação individual de emails
            - Análise em lote
            - Visualização de métricas
            """)
            
            with gr.Tabs():
                # Tab 1: Treinamento
                with gr.TabItem("🔧 Treinamento"):
                    gr.Markdown("### Gerar Dataset e Treinar Modelo")
                    
                    with gr.Row():
                        num_emails = gr.Slider(
                            minimum=100,
                            maximum=5000,
                            value=1000,
                            step=100,
                            label="Número de emails para gerar"
                        )
                        train_btn = gr.Button("🚀 Gerar e Treinar", variant="primary")
                    
                    train_output = gr.Markdown(label="Resultado do Treinamento")
                    
                    with gr.Row():
                        category_chart = gr.Plot(label="Distribuição por Categoria")
                        relevance_chart = gr.Plot(label="Distribuição por Relevância")
                    
                    train_btn.click(
                        fn=self.generate_and_train,
                        inputs=[num_emails],
                        outputs=[train_output, category_chart, relevance_chart]
                    )
                
                # Tab 2: Classificação Individual
                with gr.TabItem("📧 Classificação Individual"):
                    gr.Markdown("### Classifique um Email")
                    
                    with gr.Row():
                        with gr.Column():
                            subject_input = gr.Textbox(
                                label="Assunto do Email",
                                placeholder="Ex: Reunião urgente amanhã"
                            )
                            content_input = gr.Textbox(
                                label="Conteúdo do Email",
                                placeholder="Ex: Precisamos discutir o projeto...",
                                lines=5
                            )
                            classify_btn = gr.Button("🔍 Classificar", variant="primary")
                        
                        with gr.Column():
                            classification_output = gr.Markdown(label="Resultado da Classificação")
                    
                    # Exemplos predefinidos
                    gr.Markdown("### 📝 Exemplos para Testar:")
                    
                    examples = [
                        ["Reunião de projeto amanhã", "Olá, precisamos discutir os próximos passos do projeto. Confirme sua presença."],
                        ["50% OFF em tudo!", "Promoção imperdível! Corra para nossa loja virtual e aproveite."],
                        ["Convite para aniversário", "Oi! Vai rolar festa no sábado. Você vem? Confirma aí!"],
                        ["PARABÉNS! Você ganhou R$ 10.000!", "Clique aqui para resgatar seu prêmio. Oferta por tempo limitado!"]
                    ]
                    
                    gr.Examples(
                        examples=examples,
                        inputs=[subject_input, content_input],
                        outputs=classification_output,
                        fn=self.classify_email
                    )
                    
                    classify_btn.click(
                        fn=self.classify_email,
                        inputs=[subject_input, content_input],
                        outputs=classification_output
                    )
                
                # Tab 3: Análise em Lote
                with gr.TabItem("📊 Análise em Lote"):
                    gr.Markdown("### Analisar Múltiplos Emails")
                    gr.Markdown("Faça upload de um arquivo CSV com colunas 'subject' e 'content'")
                    
                    file_input = gr.File(
                        label="Arquivo CSV",
                        file_types=[".csv"]
                    )
                    analyze_btn = gr.Button("📈 Analisar Lote", variant="primary")
                    
                    batch_output = gr.Markdown(label="Resultado da Análise")
                    batch_results = gr.Dataframe(label="Resultados Detalhados")
                    
                    analyze_btn.click(
                        fn=self.analyze_batch,
                        inputs=[file_input],
                        outputs=[batch_output, batch_results]
                    )
                
                # Tab 4: Métricas e Estatísticas
                with gr.TabItem("📊 Métricas"):
                    gr.Markdown("### Estatísticas do Modelo")
                    
                    with gr.Row():
                        stats_btn = gr.Button("📊 Ver Estatísticas", variant="secondary")
                        features_btn = gr.Button("🔍 Palavras Importantes", variant="secondary")
                    
                    stats_output = gr.Markdown(label="Estatísticas do Modelo")
                    features_output = gr.Markdown(label="Palavras Importantes")
                    
                    stats_btn.click(
                        fn=self.get_model_stats,
                        outputs=stats_output
                    )
                    
                    features_btn.click(
                        fn=self.get_feature_importance,
                        outputs=features_output
                    )
                
                # Tab 5: Integração SMTP
                with gr.TabItem("📧 Integração SMTP"):
                    gr.Markdown("### Conectar e Baixar Emails do Servidor SMTP")
                    
                    with gr.Row():
                        provider_input = gr.Dropdown(
                            label="Provedor de Email",
                            choices=["Gmail", "Outlook", "Yahoo", "Custom"],
                            value="Gmail"
                        )
                        server_input = gr.Textbox(
                            label="Servidor SMTP",
                            placeholder="Ex: smtp.gmail.com",
                            value="smtp.gmail.com"
                        )
                        port_input = gr.Textbox(
                            label="Porta",
                            placeholder="Ex: 587",
                            value="587"
                        )
                    
                    with gr.Row():
                        username_input = gr.Textbox(
                            label="Usuário",
                            placeholder="Seu email",
                            lines=1
                        )
                        password_input = gr.Textbox(
                            label="Senha",
                            placeholder="Sua senha de app",
                            lines=1,
                            type="password"
                        )
                    
                    use_ssl = gr.Checkbox(label="Usar SSL", value=True)
                    
                    connect_btn = gr.Button("🔌 Conectar", variant="primary")
                    disconnect_btn = gr.Button("❌ Desconectar", variant="secondary")
                    
                    with gr.Row():
                        folder_input = gr.Dropdown(
                            label="Pasta",
                            choices=[],
                            value="INBOX"
                        )
                        limit_input = gr.Textbox(
                            label="Limite de Emails",
                            placeholder="Ex: 100",
                            value="100"
                        )
                        days_input = gr.Textbox(
                            label="Dias para trás",
                            placeholder="Ex: 30",
                            value="30"
                        )
                    
                    download_btn = gr.Button("⬇️ Baixar Emails", variant="primary")
                    classify_download_btn = gr.Button("🔄 Reclassificar Emails", variant="secondary")
                    
                    # Resultados
                    connection_output = gr.Markdown(label="Resultado da Conexão")
                    download_output = gr.Markdown(label="Resultado do Download")
                    stats_output_smtp = gr.Markdown(label="Estatísticas dos Emails")
                    
                    # Gráficos
                    with gr.Row():
                        category_chart_smtp = gr.Plot(label="Distribuição por Categoria (SMTP)")
                        relevance_chart_smtp = gr.Plot(label="Distribuição por Relevância (SMTP)")
                    
                    # Ações dos botões
                    connect_btn.click(
                        fn=self.connect_smtp,
                        inputs=[provider_input, server_input, port_input, username_input, password_input, use_ssl],
                        outputs=[connection_output, folder_input]
                    )
                    
                    disconnect_btn.click(
                        fn=self.disconnect_smtp,
                        outputs=[connection_output, folder_input]
                    )
                    
                    download_btn.click(
                        fn=self.download_emails,
                        inputs=[folder_input, limit_input, days_input],
                        outputs=[download_output, stats_output_smtp, category_chart_smtp, relevance_chart_smtp]
                    )
                    
                    classify_download_btn.click(
                        fn=self.classify_downloaded_emails,
                        outputs=[download_output, stats_output_smtp]
                    )
                
                # Tab 5: SMTP Integration
                with gr.TabItem("📨 SMTP Integration"):
                    gr.Markdown("### Conectar e Baixar Emails do Servidor SMTP/IMAP")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🔧 Configuração de Conexão")
                            
                            provider_dropdown = gr.Dropdown(
                                choices=["custom", "gmail", "outlook", "yahoo", "icloud"],
                                value="gmail",
                                label="Provedor de Email",
                                allow_custom_value=True
                            )
                            
                            with gr.Row():
                                smtp_server = gr.Textbox(
                                    label="Servidor IMAP",
                                    placeholder="imap.gmail.com",
                                    value="imap.gmail.com"
                                )
                                smtp_port = gr.Number(
                                    label="Porta",
                                    value=993,
                                    precision=0
                                )
                            
                            smtp_username = gr.Textbox(
                                label="Email/Username",
                                placeholder="seu.email@gmail.com"
                            )
                            smtp_password = gr.Textbox(
                                label="Senha/App Password",
                                type="password",
                                placeholder="sua senha ou app password"
                            )
                            
                            smtp_ssl = gr.Checkbox(
                                label="Usar SSL",
                                value=True
                            )
                            
                            with gr.Row():
                                test_btn = gr.Button("🧪 Testar Conexão", variant="secondary")
                                connect_btn = gr.Button("🔗 Conectar", variant="primary")
                                disconnect_btn = gr.Button("🔌 Desconectar", variant="stop")
                            
                            connection_status = gr.Markdown("Status: Desconectado")
                        
                        with gr.Column():
                            gr.Markdown("#### 📥 Download de Emails")
                            
                            folder_dropdown = gr.Dropdown(
                                choices=["INBOX"],
                                value="INBOX",
                                label="Pasta/Folder",
                                allow_custom_value=True
                            )
                            
                            email_limit = gr.Slider(
                                minimum=10,
                                maximum=500,
                                value=100,
                                step=10,
                                label="Limite de Emails"
                            )
                            
                            days_back = gr.Slider(
                                minimum=1,
                                maximum=90,
                                value=30,
                                step=1,
                                label="Dias para trás"
                            )
                            
                            with gr.Row():
                                download_btn = gr.Button("📥 Baixar Emails", variant="primary")
                                classify_smtp_btn = gr.Button("🤖 Reclassificar", variant="secondary")
                                save_smtp_btn = gr.Button("💾 Salvar CSV", variant="secondary")
                            
                            download_status = gr.Markdown("Aguardando download...")
                    
                    # Resultados dos emails baixados
                    gr.Markdown("### 📊 Emails Baixados")
                    
                    smtp_emails_df = gr.Dataframe(
                        label="Emails do Servidor",
                        headers=["Remetente", "Assunto", "Categoria", "Relevante", "Data"],
                        interactive=False
                    )
                    
                    smtp_stats = gr.Markdown("Nenhum email baixado ainda")
                    
                    # Gráficos dos emails SMTP
                    with gr.Row():
                        smtp_category_chart = gr.Plot(label="Categorias (SMTP)")
                        smtp_relevance_chart = gr.Plot(label="Relevância (SMTP)")
                    
                    # Event handlers para SMTP
                    def update_server_config(provider):
                        if provider == "custom":
                            return "", 993
                        elif provider == "gmail":
                            return "imap.gmail.com", 993
                        elif provider == "outlook":
                            return "outlook.office365.com", 993
                        elif provider == "yahoo":
                            return "imap.mail.yahoo.com", 993
                        elif provider == "icloud":
                            return "imap.mail.me.com", 993
                        return "", 993
                    
                    provider_dropdown.change(
                        fn=update_server_config,
                        inputs=[provider_dropdown],
                        outputs=[smtp_server, smtp_port]
                    )
                    
                    test_btn.click(
                        fn=self.test_smtp_connection,
                        inputs=[provider_dropdown, smtp_server, smtp_port, smtp_username, smtp_password, smtp_ssl],
                        outputs=connection_status
                    )
                    
                    connect_btn.click(
                        fn=self.connect_smtp,
                        inputs=[provider_dropdown, smtp_server, smtp_port, smtp_username, smtp_password, smtp_ssl],
                        outputs=[connection_status, folder_dropdown]
                    )
                    
                    disconnect_btn.click(
                        fn=self.disconnect_smtp,
                        outputs=[connection_status, folder_dropdown]
                    )
                    
                    download_btn.click(
                        fn=self.download_emails,
                        inputs=[folder_dropdown, email_limit, days_back],
                        outputs=[download_status, smtp_emails_df, smtp_stats]
                    )
                    
                    classify_smtp_btn.click(
                        fn=self.classify_downloaded_emails,
                        outputs=[download_status, smtp_emails_df]
                    )
                    
                    save_smtp_btn.click(
                        fn=self.save_downloaded_emails,
                        outputs=download_status
                    )
                    
                    # Atualiza gráficos quando emails são baixados
                    def update_smtp_charts():
                        return self.create_smtp_charts()
                    
                    download_btn.click(
                        fn=update_smtp_charts,
                        outputs=[smtp_category_chart, smtp_relevance_chart]
                    )
                    
                    classify_smtp_btn.click(
                        fn=update_smtp_charts,
                        outputs=[smtp_category_chart, smtp_relevance_chart]
                    )
            
            gr.Markdown("""
            ---
            ### 🎓 Sobre o Projeto
            
            Este é um projeto educacional que demonstra:
            - **Processamento de Linguagem Natural (NLP)**
            - **Classificação de texto com Machine Learning**
            - **Desenvolvimento de interface com Gradio**
            - **Integração com servidores SMTP/IMAP**
            - **Boas práticas em IA e ética**
            
            **Algoritmos utilizados:**
            - TF-IDF para vetorização de texto
            - Naive Bayes para classificação
            - Pré-processamento com NLTK
            
            **Integração SMTP/IMAP:**
            - Suporte para Gmail, Outlook, Yahoo, iCloud
            - Download automático de emails
            - Classificação em tempo real
            - Análise estatística
            
            **Segurança:**
            - Use App Passwords para Gmail/Yahoo
            - Conexões SSL/TLS seguras
            - Dados processados localmente
            
            **Desenvolvido para fins educacionais - Curso de IA Básica**
            """)
        
        return app

def main():
    # Cria aplicação
    app_instance = EmailFilterApp()
    app = app_instance.create_interface()
    
    # Lança aplicação
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main()
