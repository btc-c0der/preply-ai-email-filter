import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

class TestSmoke:
    """Smoke tests to ensure basic functionality works"""
    
    @pytest.mark.smoke
    def test_data_generator_smoke(self):
        """Smoke test for data generator"""
        from data_generator import EmailDataGenerator
        
        generator = EmailDataGenerator()
        df = generator.generate_dataset(10)
        
        assert len(df) == 10
        assert 'subject' in df.columns
        assert 'content' in df.columns
        assert 'is_relevant' in df.columns
    
    @pytest.mark.smoke
    def test_email_classifier_smoke(self, sample_email_data):
        """Smoke test for email classifier"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        accuracy = classifier.train(sample_email_data)
        
        assert classifier.is_trained == True
        assert 0 <= accuracy <= 1
        
        result = classifier.predict("Test", "Test content")
        assert isinstance(result, dict)
        assert 'is_relevant' in result
    
    @pytest.mark.smoke
    def test_app_smoke(self):
        """Smoke test for app"""
        from app import EmailFilterApp
        
        with patch('gradio.Blocks'):
            with patch('os.path.exists', return_value=False):
                app = EmailFilterApp()
                
                # Test basic initialization
                assert app.is_model_loaded == False
                assert app.dataset is None
    
    @pytest.mark.smoke
    def test_imports_smoke(self):
        """Smoke test for imports"""
        # Test that all modules can be imported
        import data_generator
        import email_classifier
        import app
        
        # Test that main classes can be instantiated
        generator = data_generator.EmailDataGenerator()
        classifier = email_classifier.EmailClassifier()
        
        with patch('gradio.Blocks'):
            with patch('os.path.exists', return_value=False):
                app_instance = app.EmailFilterApp()
        
        assert generator is not None
        assert classifier is not None
        assert app_instance is not None

class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.slow
    def test_large_dataset_generation(self):
        """Test generation of large dataset"""
        from data_generator import EmailDataGenerator
        
        generator = EmailDataGenerator()
        
        import time
        start_time = time.time()
        df = generator.generate_dataset(5000)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 30  # 30 seconds
        assert len(df) == 5000
    
    @pytest.mark.slow
    def test_training_performance(self):
        """Test training performance"""
        from data_generator import EmailDataGenerator
        from email_classifier import EmailClassifier
        
        generator = EmailDataGenerator()
        df = generator.generate_dataset(1000)
        
        classifier = EmailClassifier()
        
        import time
        start_time = time.time()
        accuracy = classifier.train(df)
        end_time = time.time()
        
        # Should complete training in reasonable time
        assert end_time - start_time < 60  # 1 minute
        assert accuracy >= 0.7
    
    @pytest.mark.slow
    def test_prediction_performance(self, sample_email_data):
        """Test prediction performance"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        import time
        start_time = time.time()
        
        # Make many predictions
        for i in range(1000):
            classifier.predict(f"Subject {i}", f"Content {i}")
        
        end_time = time.time()
        
        # Should complete predictions in reasonable time
        assert end_time - start_time < 10  # 10 seconds

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_dataset(self):
        """Test with empty dataset"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        empty_df = pd.DataFrame(columns=['subject', 'content', 'is_relevant'])
        
        with pytest.raises(Exception):
            classifier.train(empty_df)
    
    def test_single_class_dataset(self):
        """Test with single class dataset"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        single_class_df = pd.DataFrame({
            'subject': ['Test 1', 'Test 2'],
            'content': ['Content 1', 'Content 2'],
            'is_relevant': [1, 1]  # Only one class
        })
        
        with pytest.raises(Exception):
            classifier.train(single_class_df)
    
    def test_very_long_text(self, sample_email_data):
        """Test with very long text"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        # Test with very long subject and content
        long_subject = "Very long subject " * 100
        long_content = "Very long content " * 1000
        
        result = classifier.predict(long_subject, long_content)
        assert isinstance(result, dict)
        assert 'is_relevant' in result
    
    def test_special_characters_text(self, sample_email_data):
        """Test with special characters"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        # Test with special characters
        special_subject = "Reunião @#$%^&*() importante"
        special_content = "Conteúdo com símbolos !@#$%^&*()_+"
        
        result = classifier.predict(special_subject, special_content)
        assert isinstance(result, dict)
        assert 'is_relevant' in result
    
    def test_unicode_text(self, sample_email_data):
        """Test with unicode characters"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        # Test with unicode characters
        unicode_subject = "Reunião com acentuação àáâãéèêíìîóòôõúùûç"
        unicode_content = "Conteúdo com caracteres especiais: ñ, ü, ç"
        
        result = classifier.predict(unicode_subject, unicode_content)
        assert isinstance(result, dict)
        assert 'is_relevant' in result
    
    def test_mixed_language_text(self, sample_email_data):
        """Test with mixed language text"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        # Test with mixed Portuguese/English
        mixed_subject = "Meeting reunião importante"
        mixed_content = "Let's discuss o projeto amanhã"
        
        result = classifier.predict(mixed_subject, mixed_content)
        assert isinstance(result, dict)
        assert 'is_relevant' in result
    
    def test_numeric_only_text(self, sample_email_data):
        """Test with numeric only text"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        # Test with numbers only
        numeric_subject = "123456789"
        numeric_content = "000 111 222 333"
        
        result = classifier.predict(numeric_subject, numeric_content)
        assert isinstance(result, dict)
        assert 'is_relevant' in result
    
    def test_minimum_dataset_size(self):
        """Test with minimum viable dataset size"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        
        # Create minimal dataset with both classes
        minimal_df = pd.DataFrame({
            'subject': ['Important', 'Spam', 'Work', 'Promotion'],
            'content': ['Work content', 'Spam content', 'Work content', 'Promo content'],
            'is_relevant': [1, 0, 1, 0]
        })
        
        # Should be able to train
        accuracy = classifier.train(minimal_df)
        assert 0 <= accuracy <= 1
    
    def test_imbalanced_dataset(self):
        """Test with highly imbalanced dataset"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        
        # Create imbalanced dataset (90% one class, 10% other)
        subjects = ['Important'] * 9 + ['Spam'] * 1
        contents = ['Work content'] * 9 + ['Spam content'] * 1
        labels = [1] * 9 + [0] * 1
        
        imbalanced_df = pd.DataFrame({
            'subject': subjects,
            'content': contents,
            'is_relevant': labels
        })
        
        # Should handle imbalanced data
        accuracy = classifier.train(imbalanced_df)
        assert 0 <= accuracy <= 1

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_missing_columns(self):
        """Test with missing required columns"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        
        # Dataset missing required columns
        bad_df = pd.DataFrame({
            'wrong_column': ['test'],
            'another_wrong': ['test']
        })
        
        with pytest.raises(Exception):
            classifier.train(bad_df)
    
    def test_none_values_in_text(self):
        """Test with None values in text columns"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        
        # Dataset with None values
        df_with_none = pd.DataFrame({
            'subject': ['Test', None, 'Another'],
            'content': [None, 'Content', 'Another content'],
            'is_relevant': [1, 0, 1]
        })
        
        # Should handle None values gracefully
        accuracy = classifier.train(df_with_none)
        assert 0 <= accuracy <= 1
    
    def test_invalid_relevance_values(self):
        """Test with invalid relevance values"""
        from email_classifier import EmailClassifier
        
        classifier = EmailClassifier()
        
        # Dataset with invalid relevance values
        bad_relevance_df = pd.DataFrame({
            'subject': ['Test 1', 'Test 2'],
            'content': ['Content 1', 'Content 2'],
            'is_relevant': [2, 3]  # Invalid values (should be 0 or 1)
        })
        
        # Should handle or raise appropriate error
        with pytest.raises(Exception):
            classifier.train(bad_relevance_df)
    
    def test_app_error_handling(self):
        """Test app error handling"""
        from app import EmailFilterApp
        
        with patch('gradio.Blocks'):
            with patch('os.path.exists', return_value=False):
                app = EmailFilterApp()
                
                # Test error handling in various methods
                assert "⚠️" in app.classify_email("", "")
                assert "⚠️" in app.get_model_stats()
                assert "⚠️" in app.get_feature_importance()
                
                # Test with invalid file
                invalid_file = Mock()
                invalid_file.name = 'nonexistent.csv'
                result = app.analyze_batch(invalid_file)
                assert "❌" in result or "⚠️" in result

class TestDataQuality:
    """Test data quality and validation"""
    
    def test_generated_data_quality(self):
        """Test quality of generated data"""
        from data_generator import EmailDataGenerator
        
        generator = EmailDataGenerator()
        df = generator.generate_dataset(100)
        
        # Check data quality
        assert df['subject'].notna().all()
        assert df['content'].notna().all()
        assert df['is_relevant'].isin([0, 1]).all()
        
        # Check that subjects are not empty
        assert (df['subject'].str.len() > 0).all()
        assert (df['content'].str.len() > 0).all()
        
        # Check email format for senders
        assert df['sender'].str.contains('@').all()
        assert df['sender'].str.contains(r'\.').all()
    
    def test_data_diversity(self):
        """Test diversity of generated data"""
        from data_generator import EmailDataGenerator
        
        generator = EmailDataGenerator()
        df = generator.generate_dataset(200)
        
        # Check diversity
        unique_subjects = df['subject'].nunique()
        unique_contents = df['content'].nunique()
        unique_senders = df['sender'].nunique()
        
        # Should have reasonable diversity
        assert unique_subjects > 10
        assert unique_contents > 10
        assert unique_senders > 3
    
    def test_category_distribution(self):
        """Test category distribution in generated data"""
        from data_generator import EmailDataGenerator
        
        generator = EmailDataGenerator()
        df = generator.generate_dataset(1000)
        
        # Check category distribution
        category_counts = df['category'].value_counts()
        
        # Each category should have reasonable representation
        for category in ['importante', 'promocional', 'spam', 'pessoal']:
            assert category in category_counts.index
            percentage = category_counts[category] / len(df)
            assert 0.1 <= percentage <= 0.5  # 10-50% for each category
