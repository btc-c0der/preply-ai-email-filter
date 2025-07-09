import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
import gradio as gr
from app import EmailFilterApp

class TestEmailFilterApp:
    """Test suite for EmailFilterApp class"""
    
    def test_init(self):
        """Test EmailFilterApp initialization"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    
                    # Check initialization
                    assert app.is_model_loaded == False
                    assert app.dataset is None
                    mock_classifier_class.assert_called_once()
                    mock_generator_class.assert_called_once()
    
    def test_init_with_existing_model(self):
        """Test initialization with existing model"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=True):
                    mock_classifier = Mock()
                    mock_classifier_class.return_value = mock_classifier
                    
                    app = EmailFilterApp()
                    
                    # Check that model loading was attempted
                    mock_classifier.load_model.assert_called_once()
                    assert app.is_model_loaded == True
    
    def test_init_with_model_loading_error(self):
        """Test initialization with model loading error"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=True):
                    mock_classifier = Mock()
                    mock_classifier.load_model.side_effect = Exception("Load error")
                    mock_classifier_class.return_value = mock_classifier
                    
                    app = EmailFilterApp()
                    
                    # Should handle error gracefully
                    assert app.is_model_loaded == False
    
    def test_generate_and_train_success(self):
        """Test successful dataset generation and training"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    # Setup mocks
                    mock_classifier = Mock()
                    mock_classifier.train.return_value = 0.85
                    mock_classifier_class.return_value = mock_classifier
                    
                    mock_generator = Mock()
                    mock_dataset = pd.DataFrame({
                        'category': ['importante', 'spam', 'pessoal', 'promocional'],
                        'is_relevant': [1, 0, 1, 0]
                    })
                    mock_generator.generate_dataset.return_value = mock_dataset
                    mock_generator_class.return_value = mock_generator
                    
                    app = EmailFilterApp()
                    
                    with patch.object(app, 'create_category_chart') as mock_cat_chart:
                        with patch.object(app, 'create_relevance_chart') as mock_rel_chart:
                            mock_cat_chart.return_value = "category_chart"
                            mock_rel_chart.return_value = "relevance_chart"
                            
                            result_text, cat_chart, rel_chart = app.generate_and_train(100)
                            
                            # Check that methods were called
                            mock_generator.generate_dataset.assert_called_once_with(100)
                            mock_classifier.train.assert_called_once()
                            mock_classifier.save_model.assert_called_once()
                            
                            # Check results
                            assert "Modelo treinado com sucesso!" in result_text
                            assert "100" in result_text
                            assert "0.85" in result_text
                            assert cat_chart == "category_chart"
                            assert rel_chart == "relevance_chart"
                            assert app.is_model_loaded == True
    
    def test_generate_and_train_error(self):
        """Test dataset generation and training with error"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    # Setup mocks to raise error
                    mock_generator = Mock()
                    mock_generator.generate_dataset.side_effect = Exception("Generation error")
                    mock_generator_class.return_value = mock_generator
                    
                    app = EmailFilterApp()
                    
                    result_text, cat_chart, rel_chart = app.generate_and_train(100)
                    
                    # Check error handling
                    assert "❌ Erro:" in result_text
                    assert "Generation error" in result_text
                    assert cat_chart is None
                    assert rel_chart is None
    
    def test_classify_email_model_not_loaded(self):
        """Test email classification when model is not loaded"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    
                    result = app.classify_email("Test subject", "Test content")
                    
                    assert "⚠️ Modelo não carregado" in result
    
    def test_classify_email_success(self):
        """Test successful email classification"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    # Setup mock classifier
                    mock_classifier = Mock()
                    mock_classifier.predict.return_value = {
                        'is_relevant': True,
                        'confidence': 0.85,
                        'probabilities': {'not_relevant': 0.15, 'relevant': 0.85}
                    }
                    mock_classifier_class.return_value = mock_classifier
                    
                    app = EmailFilterApp()
                    app.is_model_loaded = True
                    
                    result = app.classify_email("Test subject", "Test content")
                    
                    # Check that classifier was called
                    mock_classifier.predict.assert_called_once_with("Test subject", "Test content")
                    
                    # Check result
                    assert "🟢 **RELEVANTE**" in result
                    assert "0.85" in result
    
    def test_classify_email_not_relevant(self):
        """Test email classification for non-relevant email"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    # Setup mock classifier
                    mock_classifier = Mock()
                    mock_classifier.predict.return_value = {
                        'is_relevant': False,
                        'confidence': 0.75,
                        'probabilities': {'not_relevant': 0.75, 'relevant': 0.25}
                    }
                    mock_classifier_class.return_value = mock_classifier
                    
                    app = EmailFilterApp()
                    app.is_model_loaded = True
                    
                    result = app.classify_email("Spam subject", "Spam content")
                    
                    # Check result
                    assert "🔴 **NÃO RELEVANTE**" in result
                    assert "0.75" in result
    
    def test_classify_email_error(self):
        """Test email classification with error"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    # Setup mock classifier to raise error
                    mock_classifier = Mock()
                    mock_classifier.predict.side_effect = Exception("Classification error")
                    mock_classifier_class.return_value = mock_classifier
                    
                    app = EmailFilterApp()
                    app.is_model_loaded = True
                    
                    result = app.classify_email("Test subject", "Test content")
                    
                    # Check error handling
                    assert "❌ Erro na classificação:" in result
                    assert "Classification error" in result
    
    def test_analyze_batch_model_not_loaded(self):
        """Test batch analysis when model is not loaded"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    
                    mock_file = Mock()
                    mock_file.name = 'test.csv'
                    
                    result = app.analyze_batch(mock_file)
                    
                    assert "⚠️ Modelo não carregado" in result
    
    def test_analyze_batch_success(self, csv_file):
        """Test successful batch analysis"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    # Setup mock classifier
                    mock_classifier = Mock()
                    mock_classifier.predict.return_value = {
                        'is_relevant': True,
                        'confidence': 0.85
                    }
                    mock_classifier_class.return_value = mock_classifier
                    
                    app = EmailFilterApp()
                    app.is_model_loaded = True
                    
                    mock_file = Mock()
                    mock_file.name = csv_file
                    
                    analysis, results_df = app.analyze_batch(mock_file)
                    
                    # Check that analysis was performed
                    assert "📊 **Análise do Lote:**" in analysis
                    assert "Total de emails:" in analysis
                    assert isinstance(results_df, pd.DataFrame)
    
    def test_analyze_batch_missing_columns(self, temp_dir):
        """Test batch analysis with missing columns"""
        # Create CSV with missing columns
        df = pd.DataFrame({'wrong_column': ['test']})
        csv_file = os.path.join(temp_dir, 'wrong_columns.csv')
        df.to_csv(csv_file, index=False)
        
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    app.is_model_loaded = True
                    
                    mock_file = Mock()
                    mock_file.name = csv_file
                    
                    result = app.analyze_batch(mock_file)
                    
                    assert "❌ Arquivo deve ter colunas 'subject' e 'content'" in result
    
    def test_analyze_batch_error(self):
        """Test batch analysis with file error"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    app.is_model_loaded = True
                    
                    # Mock file that causes error
                    mock_file = Mock()
                    mock_file.name = 'nonexistent.csv'
                    
                    result = app.analyze_batch(mock_file)
                    
                    assert "❌ Erro na análise:" in result
    
    def test_get_model_stats_not_loaded(self):
        """Test getting model stats when model is not loaded"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    
                    result = app.get_model_stats()
                    
                    assert "⚠️ Modelo não carregado" in result
    
    def test_get_model_stats_success(self, mock_classifier):
        """Test getting model stats successfully"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    mock_classifier_class.return_value = mock_classifier
                    
                    app = EmailFilterApp()
                    app.is_model_loaded = True
                    app.classifier = mock_classifier
                    
                    result = app.get_model_stats()
                    
                    assert "📊 **Estatísticas do Modelo:**" in result
                    assert "0.85" in result  # accuracy
                    assert "Precisão:" in result
                    assert "Recall:" in result
                    assert "F1-Score:" in result
    
    def test_get_feature_importance_not_loaded(self):
        """Test getting feature importance when model is not loaded"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    
                    result = app.get_feature_importance()
                    
                    assert "⚠️ Modelo não carregado" in result
    
    def test_get_feature_importance_success(self, mock_classifier):
        """Test getting feature importance successfully"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    mock_classifier_class.return_value = mock_classifier
                    
                    app = EmailFilterApp()
                    app.is_model_loaded = True
                    app.classifier = mock_classifier
                    
                    result = app.get_feature_importance()
                    
                    assert "🔍 **Palavras mais importantes para classificação:**" in result
                    assert "reunião" in result
                    assert "projeto" in result
                    assert "urgente" in result
    
    def test_create_category_chart_no_dataset(self):
        """Test creating category chart without dataset"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    
                    result = app.create_category_chart()
                    
                    assert result is None
    
    def test_create_category_chart_with_dataset(self, sample_email_data):
        """Test creating category chart with dataset"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    with patch('plotly.express.pie') as mock_pie:
                        mock_pie.return_value = "pie_chart"
                        
                        app = EmailFilterApp()
                        app.dataset = sample_email_data
                        
                        result = app.create_category_chart()
                        
                        assert result == "pie_chart"
                        mock_pie.assert_called_once()
    
    def test_create_relevance_chart_no_dataset(self):
        """Test creating relevance chart without dataset"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    
                    result = app.create_relevance_chart()
                    
                    assert result is None
    
    def test_create_relevance_chart_with_dataset(self, sample_email_data):
        """Test creating relevance chart with dataset"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    with patch('plotly.express.bar') as mock_bar:
                        mock_bar.return_value = "bar_chart"
                        
                        app = EmailFilterApp()
                        app.dataset = sample_email_data
                        
                        result = app.create_relevance_chart()
                        
                        assert result == "bar_chart"
                        mock_bar.assert_called_once()
    
    def test_create_interface(self, mock_gradio_interface):
        """Test creating Gradio interface"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    
                    with patch('gradio.Blocks') as mock_blocks:
                        mock_blocks.return_value.__enter__.return_value = mock_gradio_interface
                        
                        result = app.create_interface()
                        
                        assert result == mock_gradio_interface
    
    def test_main_function(self):
        """Test main function execution"""
        with patch('app.EmailFilterApp') as mock_app_class:
            mock_app = Mock()
            mock_interface = Mock()
            mock_app.create_interface.return_value = mock_interface
            mock_app_class.return_value = mock_app
            
            from app import main
            main()
            
            # Check that app was created and launched
            mock_app_class.assert_called_once()
            mock_app.create_interface.assert_called_once()
            mock_interface.launch.assert_called_once_with(
                server_name="0.0.0.0",
                server_port=7860,
                share=True,
                debug=True
            )
    
    @patch('gradio.Markdown')
    @patch('gradio.Tabs')
    @patch('gradio.TabItem')
    @patch('gradio.Row')
    @patch('gradio.Column')
    @patch('gradio.Slider')
    @patch('gradio.Button')
    @patch('gradio.Textbox')
    @patch('gradio.File')
    @patch('gradio.Dataframe')
    @patch('gradio.Plot')
    @patch('gradio.Examples')
    def test_interface_components(self, *mock_components):
        """Test that interface creates all necessary components"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    
                    with patch('gradio.Blocks') as mock_blocks:
                        mock_interface = Mock()
                        mock_blocks.return_value.__enter__.return_value = mock_interface
                        
                        app.create_interface()
                        
                        # Check that Blocks was called
                        mock_blocks.assert_called_once()
    
    def test_batch_analysis_statistics(self, csv_file):
        """Test batch analysis statistics calculation"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    # Setup mock classifier with different predictions
                    mock_classifier = Mock()
                    mock_classifier.predict.side_effect = [
                        {'is_relevant': True, 'confidence': 0.8},
                        {'is_relevant': False, 'confidence': 0.7},
                        {'is_relevant': True, 'confidence': 0.9},
                        {'is_relevant': False, 'confidence': 0.6},
                        {'is_relevant': True, 'confidence': 0.85}
                    ]
                    mock_classifier_class.return_value = mock_classifier
                    
                    app = EmailFilterApp()
                    app.is_model_loaded = True
                    
                    mock_file = Mock()
                    mock_file.name = csv_file
                    
                    analysis, results_df = app.analyze_batch(mock_file)
                    
                    # Check statistics
                    assert "Total de emails: 5" in analysis
                    assert "Relevantes: 3" in analysis
                    assert "Não relevantes: 2" in analysis
                    assert "Confiança média:" in analysis
    
    def test_error_handling_in_methods(self):
        """Test error handling in various methods"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    app = EmailFilterApp()
                    
                    # Test that methods handle errors gracefully
                    assert "⚠️" in app.classify_email("", "")
                    assert "⚠️" in app.analyze_batch(Mock())
                    assert "⚠️" in app.get_model_stats()
                    assert "⚠️" in app.get_feature_importance()
    
    def test_dataset_assignment(self):
        """Test that dataset is properly assigned after generation"""
        with patch('app.EmailClassifier') as mock_classifier_class:
            with patch('app.EmailDataGenerator') as mock_generator_class:
                with patch('os.path.exists', return_value=False):
                    mock_classifier = Mock()
                    mock_classifier.train.return_value = 0.85
                    mock_classifier_class.return_value = mock_classifier
                    
                    mock_generator = Mock()
                    mock_dataset = pd.DataFrame({'test': [1, 2, 3]})
                    mock_generator.generate_dataset.return_value = mock_dataset
                    mock_generator_class.return_value = mock_generator
                    
                    app = EmailFilterApp()
                    
                    with patch.object(app, 'create_category_chart'):
                        with patch.object(app, 'create_relevance_chart'):
                            app.generate_and_train(100)
                            
                            # Check that dataset was assigned
                            pd.testing.assert_frame_equal(app.dataset, mock_dataset)
