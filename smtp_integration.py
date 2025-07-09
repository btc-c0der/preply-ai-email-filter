import imaplib
import email
import pandas as pd
from email.header import decode_header
from datetime import datetime, timedelta
import ssl
import logging
from typing import List, Dict, Optional, Tuple
import re
import base64
import quopri

class SMTPEmailFetcher:
    """Classe para buscar emails de servidores IMAP/SMTP"""
    
    def __init__(self):
        self.connection = None
        self.server = None
        self.port = None
        self.username = None
        self.is_connected = False
        self.logger = logging.getLogger(__name__)
        
        # Configurações pré-definidas para provedores populares
        self.provider_configs = {
            'gmail': {
                'server': 'imap.gmail.com',
                'port': 993,
                'ssl': True,
                'auth_type': 'oauth2_or_app_password'
            },
            'outlook': {
                'server': 'outlook.office365.com',
                'port': 993,
                'ssl': True,
                'auth_type': 'basic'
            },
            'yahoo': {
                'server': 'imap.mail.yahoo.com',
                'port': 993,
                'ssl': True,
                'auth_type': 'app_password'
            },
            'icloud': {
                'server': 'imap.mail.me.com',
                'port': 993,
                'ssl': True,
                'auth_type': 'app_password'
            }
        }
    
    def connect(self, server: str, port: int, username: str, password: str, use_ssl: bool = True) -> Tuple[bool, str]:
        """Conecta ao servidor IMAP"""
        try:
            self.server = server
            self.port = port
            self.username = username
            
            # Cria conexão
            if use_ssl:
                self.connection = imaplib.IMAP4_SSL(server, port)
            else:
                self.connection = imaplib.IMAP4(server, port)
            
            # Faz login
            result = self.connection.login(username, password)
            
            if result[0] == 'OK':
                self.is_connected = True
                return True, "Conexão estabelecida com sucesso!"
            else:
                return False, f"Falha no login: {result[1]}"
                
        except imaplib.IMAP4.error as e:
            return False, f"Erro IMAP: {str(e)}"
        except ssl.SSLError as e:
            return False, f"Erro SSL: {str(e)}"
        except Exception as e:
            return False, f"Erro de conexão: {str(e)}"
    
    def connect_with_provider(self, provider: str, username: str, password: str) -> Tuple[bool, str]:
        """Conecta usando configuração pré-definida do provedor"""
        if provider not in self.provider_configs:
            return False, f"Provedor '{provider}' não suportado"
        
        config = self.provider_configs[provider]
        return self.connect(
            config['server'],
            config['port'],
            username,
            password,
            config['ssl']
        )
    
    def disconnect(self):
        """Desconecta do servidor"""
        if self.connection and self.is_connected:
            try:
                self.connection.close()
                self.connection.logout()
            except:
                pass
        self.is_connected = False
        self.connection = None
    
    def list_folders(self) -> List[str]:
        """Lista pastas disponíveis no servidor"""
        if not self.is_connected:
            return []
        
        try:
            result, folders = self.connection.list()
            if result == 'OK':
                folder_names = []
                for folder in folders:
                    # Decodifica nome da pasta
                    folder_info = folder.decode().split('"')
                    if len(folder_info) >= 3:
                        folder_names.append(folder_info[-2])
                return folder_names
        except Exception as e:
            self.logger.error(f"Erro ao listar pastas: {e}")
        
        return []
    
    def decode_header_value(self, header_value: str) -> str:
        """Decodifica valores de cabeçalho de email"""
        if not header_value:
            return ""
        
        try:
            decoded_parts = decode_header(header_value)
            decoded_string = ""
            
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        decoded_string += part.decode(encoding)
                    else:
                        # Tenta diferentes encodings
                        for enc in ['utf-8', 'latin-1', 'ascii']:
                            try:
                                decoded_string += part.decode(enc)
                                break
                            except:
                                continue
                        else:
                            # Se nenhum encoding funcionar, usa replace
                            decoded_string += part.decode('utf-8', errors='replace')
                else:
                    decoded_string += str(part)
            
            return decoded_string.strip()
        except Exception as e:
            self.logger.error(f"Erro ao decodificar header: {e}")
            return str(header_value)
    
    def extract_email_content(self, msg) -> str:
        """Extrai conteúdo do email"""
        content = ""
        
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        charset = part.get_content_charset() or 'utf-8'
                        payload = part.get_payload(decode=True)
                        if payload:
                            try:
                                content += payload.decode(charset, errors='replace')
                            except:
                                content += payload.decode('utf-8', errors='replace')
                    elif content_type == "text/html" and not content:
                        # Usa HTML se não tiver texto plano
                        charset = part.get_content_charset() or 'utf-8'
                        payload = part.get_payload(decode=True)
                        if payload:
                            try:
                                html_content = payload.decode(charset, errors='replace')
                                # Remove tags HTML básicas
                                content += re.sub(r'<[^>]+>', '', html_content)
                            except:
                                pass
            else:
                # Email não é multipart
                charset = msg.get_content_charset() or 'utf-8'
                payload = msg.get_payload(decode=True)
                if payload:
                    try:
                        content = payload.decode(charset, errors='replace')
                    except:
                        content = payload.decode('utf-8', errors='replace')
        except Exception as e:
            self.logger.error(f"Erro ao extrair conteúdo: {e}")
        
        return content.strip()
    
    def fetch_emails(self, folder: str = "INBOX", limit: int = 100, days_back: int = 30) -> List[Dict]:
        """Busca emails do servidor"""
        if not self.is_connected:
            return []
        
        emails = []
        
        try:
            # Seleciona pasta
            result = self.connection.select(folder)
            if result[0] != 'OK':
                self.logger.error(f"Não foi possível selecionar a pasta {folder}")
                return []
            
            # Calcula data limite
            since_date = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")
            
            # Busca emails
            search_criteria = f'(SINCE "{since_date}")'
            result, message_ids = self.connection.search(None, search_criteria)
            
            if result != 'OK':
                self.logger.error("Erro na busca de emails")
                return []
            
            # Pega IDs dos emails (limitado)
            email_ids = message_ids[0].split()
            if limit > 0:
                email_ids = email_ids[-limit:]  # Pega os mais recentes
            
            # Processa cada email
            for i, email_id in enumerate(email_ids):
                try:
                    # Busca email
                    result, msg_data = self.connection.fetch(email_id, '(RFC822)')
                    if result != 'OK':
                        continue
                    
                    # Parse do email
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    
                    # Extrai informações
                    subject = self.decode_header_value(msg.get('Subject', ''))
                    sender = self.decode_header_value(msg.get('From', ''))
                    date_str = msg.get('Date', '')
                    content = self.extract_email_content(msg)
                    
                    # Parse da data
                    try:
                        if date_str:
                            # Remove timezone info para simplificar
                            date_str = re.sub(r'\s*\([^)]*\)$', '', date_str)
                            date_str = re.sub(r'\s*[+-]\d{4}$', '', date_str)
                            email_date = email.utils.parsedate_to_datetime(date_str)
                        else:
                            email_date = datetime.now()
                    except:
                        email_date = datetime.now()
                    
                    # Limpa endereço do remetente
                    sender_clean = re.sub(r'.*<(.+?)>.*', r'\1', sender)
                    if not sender_clean or '@' not in sender_clean:
                        sender_clean = sender
                    
                    email_data = {
                        'id': f"smtp_{i+1}",
                        'sender': sender_clean,
                        'subject': subject,
                        'content': content[:1000],  # Limita tamanho do conteúdo
                        'timestamp': email_date.isoformat(),
                        'folder': folder,
                        'category': 'unknown',  # Será classificado depois
                        'is_relevant': 0  # Será classificado depois
                    }
                    
                    emails.append(email_data)
                    
                except Exception as e:
                    self.logger.error(f"Erro ao processar email {email_id}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Erro ao buscar emails: {e}")
        
        return emails
    
    def test_connection(self, server: str, port: int, username: str, password: str, use_ssl: bool = True) -> Tuple[bool, str]:
        """Testa conexão sem salvar estado"""
        try:
            if use_ssl:
                test_conn = imaplib.IMAP4_SSL(server, port)
            else:
                test_conn = imaplib.IMAP4(server, port)
            
            result = test_conn.login(username, password)
            test_conn.logout()
            
            if result[0] == 'OK':
                return True, "Conexão teste bem-sucedida!"
            else:
                return False, f"Falha no teste: {result[1]}"
                
        except Exception as e:
            return False, f"Erro no teste: {str(e)}"

class EmailProcessor:
    """Classe para processar emails baixados"""
    
    def __init__(self, classifier=None):
        self.classifier = classifier
    
    def classify_emails(self, emails: List[Dict]) -> List[Dict]:
        """Classifica emails usando o modelo de IA"""
        if not self.classifier or not self.classifier.is_trained:
            return emails
        
        for email_data in emails:
            try:
                result = self.classifier.predict(
                    email_data.get('subject', ''),
                    email_data.get('content', '')
                )
                
                email_data['is_relevant'] = 1 if result['is_relevant'] else 0
                email_data['confidence'] = result['confidence']
                
                # Determina categoria baseada no conteúdo
                content_lower = (email_data.get('subject', '') + ' ' + email_data.get('content', '')).lower()
                
                if any(word in content_lower for word in ['reunião', 'projeto', 'deadline', 'urgente', 'contrato']):
                    email_data['category'] = 'importante'
                elif any(word in content_lower for word in ['desconto', 'promoção', 'oferta', 'cupom']):
                    email_data['category'] = 'promocional'
                elif any(word in content_lower for word in ['ganhe', 'grátis', 'clique aqui', 'parabéns']):
                    email_data['category'] = 'spam'
                elif any(word in content_lower for word in ['família', 'amigo', 'convite', 'pessoal']):
                    email_data['category'] = 'pessoal'
                else:
                    email_data['category'] = 'outros'
                    
            except Exception as e:
                email_data['is_relevant'] = 0
                email_data['confidence'] = 0.5
                email_data['category'] = 'unknown'
        
        return emails
    
    def save_to_csv(self, emails: List[Dict], filename: str) -> str:
        """Salva emails em arquivo CSV"""
        try:
            df = pd.DataFrame(emails)
            df.to_csv(filename, index=False, encoding='utf-8')
            return f"Emails salvos em {filename}"
        except Exception as e:
            return f"Erro ao salvar: {str(e)}"
    
    def get_statistics(self, emails: List[Dict]) -> Dict:
        """Retorna estatísticas dos emails"""
        if not emails:
            return {}
        
        df = pd.DataFrame(emails)
        
        stats = {
            'total': len(emails),
            'relevant': sum(1 for e in emails if e.get('is_relevant', 0) == 1),
            'not_relevant': sum(1 for e in emails if e.get('is_relevant', 0) == 0),
            'categories': df['category'].value_counts().to_dict() if 'category' in df.columns else {},
            'folders': df['folder'].value_counts().to_dict() if 'folder' in df.columns else {},
            'date_range': {
                'oldest': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'newest': df['timestamp'].max() if 'timestamp' in df.columns else None
            }
        }
        
        return stats
