import pytest
import unittest.mock as mock
from smtp_integration import SMTPEmailFetcher, EmailProcessor
from email_classifier import EmailClassifier
import imaplib
import email
from datetime import datetime
import ssl
import socket
import pandas as pd

class TestSMTPEmailFetcher:
    """Testes para SMTPEmailFetcher"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.fetcher = SMTPEmailFetcher()
    
    def test_init(self):
        """Testa inicialização"""
        assert self.fetcher.connection is None
        assert not self.fetcher.is_connected
        assert 'gmail' in self.fetcher.provider_configs
        assert 'outlook' in self.fetcher.provider_configs
        assert 'yahoo' in self.fetcher.provider_configs
        assert 'icloud' in self.fetcher.provider_configs
    
    def test_provider_configs(self):
        """Testa configurações dos provedores"""
        gmail_config = self.fetcher.provider_configs['gmail']
        assert gmail_config['server'] == 'imap.gmail.com'
        assert gmail_config['port'] == 993
        assert gmail_config['ssl'] is True
        assert gmail_config['auth_type'] == 'oauth2_or_app_password'
        
        outlook_config = self.fetcher.provider_configs['outlook']
        assert outlook_config['server'] == 'outlook.office365.com'
        assert outlook_config['port'] == 993
        assert outlook_config['ssl'] is True
        
        yahoo_config = self.fetcher.provider_configs['yahoo']
        assert yahoo_config['server'] == 'imap.mail.yahoo.com'
        assert yahoo_config['port'] == 993
        assert yahoo_config['auth_type'] == 'app_password'
        
        icloud_config = self.fetcher.provider_configs['icloud']
        assert icloud_config['server'] == 'imap.mail.me.com'
        assert icloud_config['port'] == 993
        assert icloud_config['auth_type'] == 'app_password'
    
    @mock.patch('imaplib.IMAP4_SSL')
    def test_connect_success(self, mock_imap):
        """Testa conexão bem-sucedida"""
        # Mock do objeto IMAP
        mock_conn = mock.MagicMock()
        mock_conn.login.return_value = ('OK', ['Success'])
        mock_imap.return_value = mock_conn
        
        success, message = self.fetcher.connect('imap.test.com', 993, 'user@test.com', 'password')
        
        assert success is True
        assert 'sucesso' in message.lower()
        assert self.fetcher.is_connected is True
        assert self.fetcher.server == 'imap.test.com'
        assert self.fetcher.port == 993
        assert self.fetcher.username == 'user@test.com'
        mock_imap.assert_called_once_with('imap.test.com', 993)
        mock_conn.login.assert_called_once_with('user@test.com', 'password')
    
    @mock.patch('imaplib.IMAP4')
    def test_connect_success_no_ssl(self, mock_imap):
        """Testa conexão bem-sucedida sem SSL"""
        mock_conn = mock.MagicMock()
        mock_conn.login.return_value = ('OK', ['Success'])
        mock_imap.return_value = mock_conn
        
        success, message = self.fetcher.connect('imap.test.com', 143, 'user@test.com', 'password', use_ssl=False)
        
        assert success is True
        assert 'sucesso' in message.lower()
        mock_imap.assert_called_once_with('imap.test.com', 143)
    
    @mock.patch('imaplib.IMAP4_SSL')
    def test_connect_login_failure(self, mock_imap):
        """Testa falha no login"""
        mock_conn = mock.MagicMock()
        mock_conn.login.return_value = ('NO', ['Authentication failed'])
        mock_imap.return_value = mock_conn
        
        success, message = self.fetcher.connect('imap.test.com', 993, 'user@test.com', 'wrongpassword')
        
        assert success is False
        assert 'falha no login' in message.lower()
        assert self.fetcher.is_connected is False
    
    @mock.patch('imaplib.IMAP4_SSL')
    def test_connect_imap_error(self, mock_imap):
        """Testa erro IMAP na conexão"""
        mock_imap.side_effect = imaplib.IMAP4.error("Connection failed")
        
        success, message = self.fetcher.connect('imap.test.com', 993, 'user@test.com', 'password')
        
        assert success is False
        assert 'erro imap' in message.lower()
        assert self.fetcher.is_connected is False
    
    @mock.patch('imaplib.IMAP4_SSL')
    def test_connect_ssl_error(self, mock_imap):
        """Testa erro SSL na conexão"""
        mock_imap.side_effect = ssl.SSLError("SSL handshake failed")
        
        success, message = self.fetcher.connect('imap.test.com', 993, 'user@test.com', 'password')
        
        assert success is False
        assert 'erro ssl' in message.lower()
        assert self.fetcher.is_connected is False
    
    @mock.patch('imaplib.IMAP4_SSL')
    def test_connect_general_exception(self, mock_imap):
        """Testa exceção geral na conexão"""
        mock_imap.side_effect = Exception("Network error")
        
        success, message = self.fetcher.connect('imap.test.com', 993, 'user@test.com', 'password')
        
        assert success is False
        assert 'erro de conexão' in message.lower()
        assert self.fetcher.is_connected is False
    
    @mock.patch('imaplib.IMAP4_SSL')
    def test_connect_with_provider_gmail(self, mock_imap):
        """Testa conexão com provedor Gmail"""
        mock_conn = mock.MagicMock()
        mock_conn.login.return_value = ('OK', ['Success'])
        mock_imap.return_value = mock_conn
        
        success, message = self.fetcher.connect_with_provider('gmail', 'user@gmail.com', 'password')
        
        assert success is True
        mock_imap.assert_called_once_with('imap.gmail.com', 993)
        mock_conn.login.assert_called_once_with('user@gmail.com', 'password')
    
    @mock.patch('imaplib.IMAP4_SSL')
    def test_connect_with_provider_outlook(self, mock_imap):
        """Testa conexão com provedor Outlook"""
        mock_conn = mock.MagicMock()
        mock_conn.login.return_value = ('OK', ['Success'])
        mock_imap.return_value = mock_conn
        
        success, message = self.fetcher.connect_with_provider('outlook', 'user@outlook.com', 'password')
        
        assert success is True
        mock_imap.assert_called_once_with('outlook.office365.com', 993)
    
    @mock.patch('imaplib.IMAP4_SSL')
    def test_connect_with_provider_yahoo(self, mock_imap):
        """Testa conexão com provedor Yahoo"""
        mock_conn = mock.MagicMock()
        mock_conn.login.return_value = ('OK', ['Success'])
        mock_imap.return_value = mock_conn
        
        success, message = self.fetcher.connect_with_provider('yahoo', 'user@yahoo.com', 'password')
        
        assert success is True
        mock_imap.assert_called_once_with('imap.mail.yahoo.com', 993)
    
    @mock.patch('imaplib.IMAP4_SSL')
    def test_connect_with_provider_icloud(self, mock_imap):
        """Testa conexão com provedor iCloud"""
        mock_conn = mock.MagicMock()
        mock_conn.login.return_value = ('OK', ['Success'])
        mock_imap.return_value = mock_conn
        
        success, message = self.fetcher.connect_with_provider('icloud', 'user@icloud.com', 'password')
        
        assert success is True
        mock_imap.assert_called_once_with('imap.mail.me.com', 993)
    
    def test_connect_with_invalid_provider(self):
        """Testa conexão com provedor inválido"""
        success, message = self.fetcher.connect_with_provider('invalid_provider', 'user@test.com', 'password')
        
        assert success is False
        assert 'não suportado' in message.lower()
    
    def test_disconnect_when_connected(self):
        """Testa desconexão quando conectado"""
        # Simula conexão ativa
        mock_conn = mock.MagicMock()
        self.fetcher.connection = mock_conn
        self.fetcher.is_connected = True
        
        self.fetcher.disconnect()
        
        assert not self.fetcher.is_connected
        assert self.fetcher.connection is None
        mock_conn.close.assert_called_once()
        mock_conn.logout.assert_called_once()
    
    def test_disconnect_when_not_connected(self):
        """Testa desconexão quando não conectado"""
        self.fetcher.disconnect()
        
        assert not self.fetcher.is_connected
        assert self.fetcher.connection is None
    
    def test_disconnect_with_exception(self):
        """Testa desconexão com exceção"""
        mock_conn = mock.MagicMock()
        mock_conn.close.side_effect = Exception("Close failed")
        self.fetcher.connection = mock_conn
        self.fetcher.is_connected = True
        
        # Não deve lançar exceção
        self.fetcher.disconnect()
        
        assert not self.fetcher.is_connected
        assert self.fetcher.connection is None

class TestEmailProcessor:
    """Testes para EmailProcessor"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.processor = EmailProcessor()
        
        # Mock do classificador
        self.mock_classifier = mock.MagicMock()
        self.mock_classifier.is_trained = True
        self.mock_classifier.predict.return_value = {
            'is_relevant': True,
            'confidence': 0.85
        }
        self.processor.classifier = self.mock_classifier
    
    def test_init(self):
        """Testa inicialização"""
        processor = EmailProcessor()
        assert processor.classifier is None
        
        processor_with_classifier = EmailProcessor(self.mock_classifier)
        assert processor_with_classifier.classifier is not None
    
    def test_classify_emails_with_model(self):
        """Testa classificação com modelo treinado"""
        emails = [
            {
                'id': '1',
                'sender': 'test@test.com',
                'subject': 'Reunião importante',
                'content': 'Precisamos discutir o projeto',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        result = self.processor.classify_emails(emails)
        
        assert len(result) == 1
        assert result[0]['is_relevant'] == 1
        assert 'confidence' in result[0]
        assert 'category' in result[0]
        self.mock_classifier.predict.assert_called_once()
    
    def test_classify_emails_without_model(self):
        """Testa classificação sem modelo"""
        processor = EmailProcessor()
        emails = [{'subject': 'Test', 'content': 'Test'}]
        
        result = processor.classify_emails(emails)
        
        assert result == emails  # Deve retornar inalterado
    
    def test_classify_emails_categories(self):
        """Testa categorização por palavras-chave"""
        emails = [
            {'subject': 'Reunião projeto', 'content': ''},
            {'subject': '50% desconto', 'content': ''},
            {'subject': 'Ganhe dinheiro', 'content': ''},
            {'subject': 'Convite família', 'content': ''},
            {'subject': 'Outros assuntos', 'content': ''}
        ]
        
        result = self.processor.classify_emails(emails)
        
        categories = [email['category'] for email in result]
        assert 'importante' in categories
        assert 'promocional' in categories
        assert 'spam' in categories
        assert 'pessoal' in categories
        assert 'outros' in categories
    
    @mock.patch('pandas.DataFrame.to_csv')
    def test_save_to_csv(self, mock_to_csv):
        """Testa salvamento em CSV"""
        emails = [{'subject': 'Test', 'content': 'Test'}]
        
        result = self.processor.save_to_csv(emails, 'test.csv')
        
        assert 'salvo' in result.lower()
        mock_to_csv.assert_called_once()
    
    @mock.patch('pandas.DataFrame.to_csv')
    def test_save_to_csv_error(self, mock_to_csv):
        """Testa erro ao salvar CSV"""
        mock_to_csv.side_effect = Exception("Write error")
        emails = [{'subject': 'Test', 'content': 'Test'}]
        
        result = self.processor.save_to_csv(emails, 'test.csv')
        
        assert 'erro' in result.lower()
    
    def test_get_statistics_empty(self):
        """Testa estatísticas com lista vazia"""
        stats = self.processor.get_statistics([])
        assert stats == {}
    
    def test_get_statistics_with_emails(self):
        """Testa estatísticas com emails"""
        emails = [
            {
                'is_relevant': 1,
                'category': 'importante',
                'folder': 'INBOX',
                'timestamp': '2023-01-01T00:00:00'
            },
            {
                'is_relevant': 0,
                'category': 'spam',
                'folder': 'INBOX',
                'timestamp': '2023-01-02T00:00:00'
            }
        ]
        
        stats = self.processor.get_statistics(emails)
        
        assert stats['total'] == 2
        assert stats['relevant'] == 1
        assert stats['not_relevant'] == 1
        assert 'categories' in stats
        assert 'folders' in stats
        assert 'date_range' in stats

class TestSMTPIntegration:
    """Testes de integração SMTP"""
    
    def setup_method(self):
        """Setup para testes de integração"""
        self.fetcher = SMTPEmailFetcher()
        self.processor = EmailProcessor()
    
    def test_end_to_end_mock(self):
        """Teste end-to-end com mocks"""
        # Este teste simula o fluxo completo:
        # 1. Conectar ao servidor
        # 2. Baixar emails
        # 3. Processar emails
        # 4. Salvar resultados
        
        with mock.patch('imaplib.IMAP4_SSL') as mock_imap:
            # Mock da conexão
            mock_conn = mock.MagicMock()
            mock_conn.login.return_value = ('OK', ['Success'])
            mock_conn.select.return_value = ('OK', ['1'])
            mock_conn.search.return_value = ('OK', [b'1 2'])
            
            # Mock de emails
            email_data = b'''From: test@example.com
Subject: Test Subject
Date: Mon, 1 Jan 2023 12:00:00 +0000

Test content'''
            
            mock_conn.fetch.return_value = ('OK', [(None, email_data)])
            mock_imap.return_value = mock_conn
            
            # Executa fluxo
            success, _ = self.fetcher.connect('imap.test.com', 993, 'user@test.com', 'password')
            assert success
            
            emails = self.fetcher.fetch_emails('INBOX', 10, 30)
            assert isinstance(emails, list)
            
            processed_emails = self.processor.classify_emails(emails)
            assert isinstance(processed_emails, list)

if __name__ == '__main__':
    pytest.main([__file__])
