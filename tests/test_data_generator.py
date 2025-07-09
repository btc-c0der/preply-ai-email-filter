import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from data_generator import EmailDataGenerator
from freezegun import freeze_time

class TestEmailDataGenerator:
    """Test suite for EmailDataGenerator class"""
    
    def test_init(self):
        """Test EmailDataGenerator initialization"""
        generator = EmailDataGenerator()
        
        # Check that categories are properly initialized
        assert 'importante' in generator.categories
        assert 'promocional' in generator.categories
        assert 'spam' in generator.categories
        assert 'pessoal' in generator.categories
        
        # Check that each category has required keys
        for category in generator.categories.values():
            assert 'keywords' in category
            assert 'subjects' in category
            assert 'senders' in category
            assert isinstance(category['keywords'], list)
            assert isinstance(category['subjects'], list)
            assert isinstance(category['senders'], list)
    
    def test_generate_email_content_importante(self):
        """Test content generation for important emails"""
        generator = EmailDataGenerator()
        subject = "Reunião de projeto"
        content = generator.generate_email_content('importante', subject)
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert 'reunião de projeto' in content.lower()
        assert any(word in content.lower() for word in ['olá', 'prezado', 'bom dia'])
    
    def test_generate_email_content_promocional(self):
        """Test content generation for promotional emails"""
        generator = EmailDataGenerator()
        subject = "50% OFF em tudo!"
        content = generator.generate_email_content('promocional', subject)
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert '50% off em tudo!' in content.lower()
        assert any(word in content.lower() for word in ['promoção', 'desconto', 'oferta'])
    
    def test_generate_email_content_spam(self):
        """Test content generation for spam emails"""
        generator = EmailDataGenerator()
        subject = "Você ganhou!"
        content = generator.generate_email_content('spam', subject)
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert 'você ganhou!' in content.lower()
        assert any(word in content.lower() for word in ['clique', 'link', 'urgente'])
    
    def test_generate_email_content_pessoal(self):
        """Test content generation for personal emails"""
        generator = EmailDataGenerator()
        subject = "Aniversário no sábado"
        content = generator.generate_email_content('pessoal', subject)
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert 'aniversário no sábado' in content.lower()
        assert any(word in content.lower() for word in ['oi', 'olá', 'querido'])
    
    def test_generate_dataset_structure(self):
        """Test dataset generation structure"""
        generator = EmailDataGenerator()
        df = generator.generate_dataset(100)
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        
        # Check required columns
        required_columns = ['id', 'sender', 'subject', 'content', 'category', 'is_relevant', 'timestamp']
        for col in required_columns:
            assert col in df.columns
        
        # Check data types
        assert df['id'].dtype == 'int64'
        assert df['is_relevant'].dtype == 'int64'
        assert df['sender'].dtype == 'object'
        assert df['subject'].dtype == 'object'
        assert df['content'].dtype == 'object'
        assert df['category'].dtype == 'object'
        assert df['timestamp'].dtype == 'object'
    
    def test_generate_dataset_categories(self):
        """Test that generated dataset has correct categories"""
        generator = EmailDataGenerator()
        df = generator.generate_dataset(100)
        
        # Check that all categories are present
        categories = df['category'].unique()
        expected_categories = ['importante', 'promocional', 'spam', 'pessoal']
        for category in expected_categories:
            assert category in categories
    
    def test_generate_dataset_relevance(self):
        """Test relevance assignment logic"""
        generator = EmailDataGenerator()
        df = generator.generate_dataset(100)
        
        # Check that importante and pessoal are marked as relevant
        important_emails = df[df['category'] == 'importante']
        personal_emails = df[df['category'] == 'pessoal']
        
        assert all(important_emails['is_relevant'] == 1)
        assert all(personal_emails['is_relevant'] == 1)
        
        # Check that promocional and spam are marked as not relevant
        promotional_emails = df[df['category'] == 'promocional']
        spam_emails = df[df['category'] == 'spam']
        
        assert all(promotional_emails['is_relevant'] == 0)
        assert all(spam_emails['is_relevant'] == 0)
    
    def test_generate_dataset_size_parameter(self):
        """Test different dataset sizes"""
        generator = EmailDataGenerator()
        
        for size in [10, 50, 100, 500]:
            df = generator.generate_dataset(size)
            assert len(df) == size
    
    def test_generate_dataset_unique_ids(self):
        """Test that generated IDs are unique and sequential"""
        generator = EmailDataGenerator()
        df = generator.generate_dataset(100)
        
        # Check that IDs are unique
        assert len(df['id'].unique()) == len(df)
        
        # Check that IDs are sequential starting from 1
        assert df['id'].min() == 1
        assert df['id'].max() == 100
        assert sorted(df['id'].tolist()) == list(range(1, 101))
    
    @freeze_time("2025-07-09")
    def test_generate_dataset_timestamps(self):
        """Test timestamp generation"""
        generator = EmailDataGenerator()
        df = generator.generate_dataset(10)
        
        # Check that timestamps are strings
        assert all(isinstance(ts, str) for ts in df['timestamp'])
        
        # Check that timestamps are in ISO format
        for ts in df['timestamp']:
            assert 'T' in ts
            assert len(ts) >= 19  # YYYY-MM-DDTHH:MM:SS minimum
    
    def test_save_dataset(self, temp_dir):
        """Test dataset saving functionality"""
        generator = EmailDataGenerator()
        df = generator.generate_dataset(10)
        
        # Save to temporary directory
        filename = os.path.join(temp_dir, 'test_dataset.csv')
        result_filename = generator.save_dataset(df, filename)
        
        # Check that file was created
        assert os.path.exists(filename)
        assert result_filename == filename
        
        # Check that saved file can be loaded and has correct data
        loaded_df = pd.read_csv(filename)
        assert len(loaded_df) == 10
        assert list(loaded_df.columns) == list(df.columns)
    
    def test_category_distribution(self):
        """Test that category distribution is reasonable"""
        generator = EmailDataGenerator()
        df = generator.generate_dataset(1000)
        
        category_counts = df['category'].value_counts()
        total = len(df)
        
        # Check that each category has reasonable representation
        for category in ['importante', 'promocional', 'spam', 'pessoal']:
            assert category in category_counts.index
            percentage = category_counts[category] / total
            assert 0.1 <= percentage <= 0.5  # Each category should be 10-50%
    
    def test_email_content_varies(self):
        """Test that generated email content varies"""
        generator = EmailDataGenerator()
        df = generator.generate_dataset(50)
        
        # Check that subjects are not all the same
        unique_subjects = df['subject'].nunique()
        assert unique_subjects > 1
        
        # Check that content is not all the same
        unique_content = df['content'].nunique()
        assert unique_content > 1
    
    def test_sender_domain_validity(self):
        """Test that sender emails have valid domain format"""
        generator = EmailDataGenerator()
        df = generator.generate_dataset(100)
        
        for sender in df['sender']:
            assert '@' in sender
            assert '.' in sender
            parts = sender.split('@')
            assert len(parts) == 2
            assert len(parts[0]) > 0  # Username part
            assert len(parts[1]) > 0  # Domain part
    
    @patch('random.random')
    def test_mixed_category_generation(self, mock_random):
        """Test mixed category generation with mocked random"""
        # Set up mock to trigger mixed category (< 0.1)
        mock_random.return_value = 0.05
        
        generator = EmailDataGenerator()
        df = generator.generate_dataset(10)
        
        # Check that some subjects might be longer (mixed)
        # This is a bit tricky to test deterministically, so we just check structure
        assert len(df) == 10
        assert all(isinstance(subject, str) for subject in df['subject'])
    
    def test_main_function_execution(self, temp_dir):
        """Test the main function execution"""
        with patch('data_generator.EmailDataGenerator.save_dataset') as mock_save:
            mock_save.return_value = 'test_dataset.csv'
            
            # Import and run main function
            from data_generator import main
            
            # Capture print output
            with patch('builtins.print') as mock_print:
                main()
                
                # Check that print was called with expected messages
                print_calls = [call.args[0] for call in mock_print.call_args_list]
                assert any('Gerando dataset' in call for call in print_calls)
                assert any('Dataset salvo' in call for call in print_calls)
    
    def test_category_keywords_not_empty(self):
        """Test that category keywords are not empty"""
        generator = EmailDataGenerator()
        
        for category_name, category_data in generator.categories.items():
            assert len(category_data['keywords']) > 0
            assert all(isinstance(keyword, str) for keyword in category_data['keywords'])
            assert all(len(keyword) > 0 for keyword in category_data['keywords'])
    
    def test_category_subjects_not_empty(self):
        """Test that category subjects are not empty"""
        generator = EmailDataGenerator()
        
        for category_name, category_data in generator.categories.items():
            assert len(category_data['subjects']) > 0
            assert all(isinstance(subject, str) for subject in category_data['subjects'])
            assert all(len(subject) > 0 for subject in category_data['subjects'])
    
    def test_category_senders_not_empty(self):
        """Test that category senders are not empty"""
        generator = EmailDataGenerator()
        
        for category_name, category_data in generator.categories.items():
            assert len(category_data['senders']) > 0
            assert all(isinstance(sender, str) for sender in category_data['senders'])
            assert all('@' in sender for sender in category_data['senders'])
    
    def test_email_content_length(self):
        """Test that generated email content has reasonable length"""
        generator = EmailDataGenerator()
        df = generator.generate_dataset(100)
        
        for content in df['content']:
            assert len(content) > 10  # Minimum reasonable length
            assert len(content) < 1000  # Maximum reasonable length
    
    def test_generate_dataset_with_seed(self):
        """Test that dataset generation is reproducible with seed"""
        generator1 = EmailDataGenerator()
        generator2 = EmailDataGenerator()
        
        # Set same seed for both generators
        np.random.seed(42)
        df1 = generator1.generate_dataset(50)
        
        np.random.seed(42)
        df2 = generator2.generate_dataset(50)
        
        # DataFrames should be identical
        pd.testing.assert_frame_equal(df1, df2)
