import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from faker import Faker

# Configure test environment
os.environ['GRADIO_SERVER_NAME'] = '127.0.0.1'
os.environ['GRADIO_SERVER_PORT'] = '7861'

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_email_data():
    """Create sample email data for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'sender': [
            'gerente@empresa.com',
            'promocoes@loja.com',
            'maria@gmail.com',
            'spam@spam.com',
            'cliente@cliente.com'
        ],
        'subject': [
            'Reunião de projeto amanhã',
            '50% OFF em tudo!',
            'Convite para aniversário',
            'PARABÉNS! Você ganhou!',
            'Proposta comercial'
        ],
        'content': [
            'Precisamos discutir o projeto. Confirme presença.',
            'Promoção imperdível! Corra para a loja virtual.',
            'Oi! Festa no sábado. Você vem?',
            'Clique aqui para resgatar seu prêmio de R$ 10.000!',
            'Segue proposta para análise e aprovação.'
        ],
        'category': ['importante', 'promocional', 'pessoal', 'spam', 'importante'],
        'is_relevant': [1, 0, 1, 0, 1],
        'timestamp': [
            '2025-07-09T10:00:00',
            '2025-07-09T11:00:00',
            '2025-07-09T12:00:00',
            '2025-07-09T13:00:00',
            '2025-07-09T14:00:00'
        ]
    })

@pytest.fixture
def mock_classifier():
    """Create a mock classifier for testing"""
    mock = Mock()
    mock.is_trained = True
    mock.train_stats = {
        'accuracy': 0.85,
        'classification_report': {
            'Não Relevante': {'precision': 0.83, 'recall': 0.87, 'f1-score': 0.85},
            'Relevante': {'precision': 0.87, 'recall': 0.83, 'f1-score': 0.85}
        },
        'confusion_matrix': [[100, 20], [15, 85]]
    }
    mock.predict.return_value = {
        'is_relevant': True,
        'confidence': 0.85,
        'probabilities': {'not_relevant': 0.15, 'relevant': 0.85}
    }
    mock.get_feature_importance.return_value = [
        ('reunião', 0.5), ('projeto', 0.4), ('urgente', 0.3)
    ]
    return mock

@pytest.fixture
def mock_faker():
    """Create a mock faker for consistent test data"""
    fake = Faker('pt_BR')
    fake.seed_instance(42)
    return fake

@pytest.fixture
def csv_file(temp_dir, sample_email_data):
    """Create a CSV file for testing"""
    filepath = os.path.join(temp_dir, 'test_emails.csv')
    sample_email_data.to_csv(filepath, index=False)
    return filepath

@pytest.fixture
def mock_gradio_interface():
    """Mock Gradio interface for testing"""
    with patch('gradio.Blocks') as mock_blocks:
        mock_interface = Mock()
        mock_blocks.return_value.__enter__.return_value = mock_interface
        mock_interface.launch = Mock()
        yield mock_interface
