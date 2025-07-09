import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import joblib
from unittest.mock import Mock, patch, MagicMock
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from email_classifier import EmailClassifier

class TestEmailClassifier:
    """Test suite for EmailClassifier class"""
    
    def test_init(self):
        """Test EmailClassifier initialization"""
        classifier = EmailClassifier()
        
        # Check initialization
        assert isinstance(classifier.vectorizer, TfidfVectorizer)
        assert isinstance(classifier.model, MultinomialNB)
        assert classifier.is_trained == False
        assert isinstance(classifier.stop_words, set)
        assert len(classifier.stop_words) > 0
        
        # Check that common Portuguese stopwords are included
        assert 'de' in classifier.stop_words
        assert 'para' in classifier.stop_words
        assert 'com' in classifier.stop_words
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing"""
        classifier = EmailClassifier()
        
        # Test normal text
        text = "Olá, como você está? Reunião às 14h!"
        result = classifier.preprocess_text(text)
        
        assert isinstance(result, str)
        assert result.lower() == result  # Should be lowercase
        assert '!' not in result  # Special characters removed
        assert '?' not in result
        assert ',' not in result
        assert '14h' not in result  # Numbers removed
    
    def test_preprocess_text_empty_and_none(self):
        """Test preprocessing with empty and None input"""
        classifier = EmailClassifier()
        
        # Test None input
        assert classifier.preprocess_text(None) == ""
        
        # Test empty string
        assert classifier.preprocess_text("") == ""
        
        # Test pandas NaN
        assert classifier.preprocess_text(pd.NA) == ""
    
    def test_preprocess_text_stopwords_removal(self):
        """Test stopwords removal"""
        classifier = EmailClassifier()
        
        text = "de para com uma reunião importante"
        result = classifier.preprocess_text(text)
        
        # Stopwords should be removed
        assert 'de' not in result
        assert 'para' not in result
        assert 'com' not in result
        assert 'uma' not in result
        
        # Content words should remain
        assert 'reunião' in result
        assert 'importante' in result
    
    def test_preprocess_text_short_words(self):
        """Test removal of short words"""
        classifier = EmailClassifier()
        
        text = "eu tu você reunião"
        result = classifier.preprocess_text(text)
        
        # Words with 2 or fewer characters should be removed
        assert 'eu' not in result
        assert 'tu' not in result
        
        # Longer words should remain
        assert 'você' in result
        assert 'reunião' in result
    
    def test_preprocess_text_special_characters(self):
        """Test special character handling"""
        classifier = EmailClassifier()
        
        text = "Reunião@empresa.com #hashtag 50% desconto!"
        result = classifier.preprocess_text(text)
        
        # Special characters should be removed
        assert '@' not in result
        assert '.' not in result
        assert '#' not in result
        assert '%' not in result
        assert '!' not in result
        
        # Should contain only letters and spaces
        for char in result:
            assert char.isalpha() or char.isspace()
    
    def test_prepare_features(self, sample_email_data):
        """Test feature preparation"""
        classifier = EmailClassifier()
        df = sample_email_data.copy()
        
        result_df = classifier.prepare_features(df)
        
        # Check that new columns are created
        assert 'combined_text' in result_df.columns
        assert 'processed_text' in result_df.columns
        
        # Check that combined text contains both subject and content
        for idx, row in result_df.iterrows():
            assert row['subject'] in row['combined_text']
            assert row['content'] in row['combined_text']
        
        # Check that processed text is properly processed
        for processed_text in result_df['processed_text']:
            assert isinstance(processed_text, str)
            # Should not contain common stopwords
            assert 'de' not in processed_text
            assert 'para' not in processed_text
    
    def test_train_basic(self, sample_email_data):
        """Test basic training functionality"""
        classifier = EmailClassifier()
        
        accuracy = classifier.train(sample_email_data)
        
        # Check that training completed
        assert classifier.is_trained == True
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        
        # Check that train_stats is populated
        assert hasattr(classifier, 'train_stats')
        assert 'accuracy' in classifier.train_stats
        assert 'classification_report' in classifier.train_stats
        assert 'confusion_matrix' in classifier.train_stats
    
    def test_train_with_insufficient_data(self):
        """Test training with insufficient data"""
        classifier = EmailClassifier()
        
        # Create minimal dataset with only one class
        df = pd.DataFrame({
            'subject': ['Test'],
            'content': ['Test content'],
            'is_relevant': [1]
        })
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            classifier.train(df)
    
    def test_predict_untrained_model(self):
        """Test prediction with untrained model"""
        classifier = EmailClassifier()
        
        with pytest.raises(ValueError, match="Modelo não foi treinado ainda!"):
            classifier.predict("Test subject", "Test content")
    
    def test_predict_trained_model(self, sample_email_data):
        """Test prediction with trained model"""
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        result = classifier.predict("Reunião importante", "Discussão sobre projeto")
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'is_relevant' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        
        # Check data types
        assert isinstance(result['is_relevant'], bool)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['probabilities'], dict)
        
        # Check probabilities structure
        assert 'not_relevant' in result['probabilities']
        assert 'relevant' in result['probabilities']
        assert isinstance(result['probabilities']['not_relevant'], float)
        assert isinstance(result['probabilities']['relevant'], float)
        
        # Check that probabilities sum to 1 (approximately)
        prob_sum = result['probabilities']['not_relevant'] + result['probabilities']['relevant']
        assert abs(prob_sum - 1.0) < 0.001
    
    def test_predict_different_inputs(self, sample_email_data):
        """Test prediction with different input types"""
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        # Test with normal strings
        result1 = classifier.predict("Subject", "Content")
        assert isinstance(result1, dict)
        
        # Test with empty strings
        result2 = classifier.predict("", "")
        assert isinstance(result2, dict)
        
        # Test with None (should handle gracefully)
        result3 = classifier.predict(None, None)
        assert isinstance(result3, dict)
    
    def test_get_feature_importance_untrained(self):
        """Test feature importance with untrained model"""
        classifier = EmailClassifier()
        
        result = classifier.get_feature_importance()
        assert result == []
    
    def test_get_feature_importance_trained(self, sample_email_data):
        """Test feature importance with trained model"""
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        result = classifier.get_feature_importance(5)
        
        # Check result structure
        assert isinstance(result, list)
        assert len(result) <= 5
        
        # Check that each item is a tuple with feature name and importance
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)  # Feature name
            assert isinstance(item[1], float)  # Importance score
    
    def test_get_feature_importance_different_top_n(self, sample_email_data):
        """Test feature importance with different top_n values"""
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        # Test different top_n values
        for top_n in [1, 5, 10, 20]:
            result = classifier.get_feature_importance(top_n)
            assert len(result) <= top_n
    
    def test_save_model_untrained(self, temp_dir):
        """Test saving untrained model"""
        classifier = EmailClassifier()
        
        filename = os.path.join(temp_dir, 'test_model.pkl')
        
        with pytest.raises(ValueError, match="Modelo não foi treinado ainda!"):
            classifier.save_model(filename)
    
    def test_save_model_trained(self, sample_email_data, temp_dir):
        """Test saving trained model"""
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        filename = os.path.join(temp_dir, 'test_model.pkl')
        classifier.save_model(filename)
        
        # Check that file was created
        assert os.path.exists(filename)
        
        # Check that file contains expected data
        model_data = joblib.load(filename)
        assert 'vectorizer' in model_data
        assert 'model' in model_data
        assert 'train_stats' in model_data
    
    def test_load_model(self, sample_email_data, temp_dir):
        """Test loading model"""
        # First, create and save a model
        classifier1 = EmailClassifier()
        classifier1.train(sample_email_data)
        
        filename = os.path.join(temp_dir, 'test_model.pkl')
        classifier1.save_model(filename)
        
        # Now load it into a new classifier
        classifier2 = EmailClassifier()
        classifier2.load_model(filename)
        
        # Check that model was loaded correctly
        assert classifier2.is_trained == True
        assert hasattr(classifier2, 'train_stats')
        
        # Check that predictions work
        result = classifier2.predict("Test subject", "Test content")
        assert isinstance(result, dict)
        assert 'is_relevant' in result
    
    def test_load_nonexistent_model(self, temp_dir):
        """Test loading non-existent model"""
        classifier = EmailClassifier()
        
        filename = os.path.join(temp_dir, 'nonexistent_model.pkl')
        
        with pytest.raises(FileNotFoundError):
            classifier.load_model(filename)
    
    def test_plot_confusion_matrix_untrained(self):
        """Test plotting confusion matrix with untrained model"""
        classifier = EmailClassifier()
        
        result = classifier.plot_confusion_matrix()
        assert result is None
    
    def test_plot_confusion_matrix_trained(self, sample_email_data):
        """Test plotting confusion matrix with trained model"""
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            with patch('seaborn.heatmap') as mock_heatmap:
                result = classifier.plot_confusion_matrix()
                
                # Check that matplotlib functions were called
                mock_figure.assert_called_once()
                mock_heatmap.assert_called_once()
                
                # Check that result is the pyplot module
                import matplotlib.pyplot as plt
                assert result == plt
    
    def test_main_function(self, temp_dir):
        """Test main function execution"""
        # Create a test dataset file
        test_data = pd.DataFrame({
            'subject': ['Test 1', 'Test 2', 'Test 3'],
            'content': ['Content 1', 'Content 2', 'Content 3'],
            'is_relevant': [1, 0, 1]
        })
        
        dataset_file = os.path.join(temp_dir, 'emails_dataset.csv')
        test_data.to_csv(dataset_file, index=False)
        
        # Change to temp directory and run main
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            with patch('builtins.print') as mock_print:
                from email_classifier import main
                main()
                
                # Check that expected messages were printed
                print_calls = [call.args[0] for call in mock_print.call_args_list]
                assert any('Carregando dataset' in call for call in print_calls)
                assert any('Modelo treinado' in call for call in print_calls)
                
        finally:
            os.chdir(original_cwd)
    
    def test_vectorizer_parameters(self):
        """Test vectorizer initialization parameters"""
        classifier = EmailClassifier()
        
        assert classifier.vectorizer.max_features == 5000
        assert classifier.vectorizer.stop_words is None
    
    def test_model_parameters(self):
        """Test model initialization parameters"""
        classifier = EmailClassifier()
        
        assert isinstance(classifier.model, MultinomialNB)
    
    def test_train_stats_structure(self, sample_email_data):
        """Test train_stats structure after training"""
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        stats = classifier.train_stats
        
        # Check main structure
        assert 'accuracy' in stats
        assert 'classification_report' in stats
        assert 'confusion_matrix' in stats
        
        # Check accuracy
        assert isinstance(stats['accuracy'], float)
        assert 0 <= stats['accuracy'] <= 1
        
        # Check classification report structure
        report = stats['classification_report']
        assert 'Não Relevante' in report
        assert 'Relevante' in report
        
        # Check confusion matrix structure
        cm = stats['confusion_matrix']
        assert isinstance(cm, list)
        assert len(cm) == 2  # 2x2 matrix
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2
    
    def test_text_preprocessing_edge_cases(self):
        """Test text preprocessing with edge cases"""
        classifier = EmailClassifier()
        
        # Test with only special characters
        result1 = classifier.preprocess_text("!@#$%^&*()")
        assert result1 == ""
        
        # Test with only numbers
        result2 = classifier.preprocess_text("123456")
        assert result2 == ""
        
        # Test with mixed case
        result3 = classifier.preprocess_text("TeSt CaSe")
        assert result3 == "test case"
        
        # Test with Portuguese characters
        result4 = classifier.preprocess_text("reunião importância")
        assert 'reunião' in result4
        assert 'importância' in result4
    
    def test_combined_text_creation(self, sample_email_data):
        """Test combined text creation in prepare_features"""
        classifier = EmailClassifier()
        df = sample_email_data.copy()
        
        result_df = classifier.prepare_features(df)
        
        # Check that combined text is created correctly
        for idx, row in result_df.iterrows():
            expected_combined = f"{row['subject']} {row['content']}"
            assert row['combined_text'] == expected_combined
    
    def test_prediction_consistency(self, sample_email_data):
        """Test that predictions are consistent for same input"""
        classifier = EmailClassifier()
        classifier.train(sample_email_data)
        
        subject = "Reunião importante"
        content = "Discussão sobre projeto"
        
        # Make multiple predictions
        result1 = classifier.predict(subject, content)
        result2 = classifier.predict(subject, content)
        
        # Results should be identical
        assert result1['is_relevant'] == result2['is_relevant']
        assert result1['confidence'] == result2['confidence']
        assert result1['probabilities'] == result2['probabilities']
