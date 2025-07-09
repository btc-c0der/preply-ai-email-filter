import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

class TestIntegration:
    """Integration tests for the entire email filter system"""
    
    def test_full_pipeline_integration(self, temp_dir):
        """Test the full pipeline from data generation to prediction"""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Import modules
            from data_generator import EmailDataGenerator
            from email_classifier import EmailClassifier
            
            # Step 1: Generate data
            generator = EmailDataGenerator()
            df = generator.generate_dataset(100)
            
            # Verify data generation
            assert len(df) == 100
            assert 'subject' in df.columns
            assert 'content' in df.columns
            assert 'is_relevant' in df.columns
            
            # Step 2: Train classifier
            classifier = EmailClassifier()
            accuracy = classifier.train(df)
            
            # Verify training
            assert classifier.is_trained == True
            assert 0 <= accuracy <= 1
            
            # Step 3: Make predictions
            test_cases = [
                ("Reunião importante", "Discussão sobre projeto"),
                ("50% OFF", "Promoção especial"),
                ("Convite festa", "Oi! Vem pra festa"),
                ("Ganhe dinheiro", "Clique aqui para ganhar")
            ]
            
            for subject, content in test_cases:
                result = classifier.predict(subject, content)
                
                # Verify prediction structure
                assert isinstance(result, dict)
                assert 'is_relevant' in result
                assert 'confidence' in result
                assert 'probabilities' in result
                assert isinstance(result['is_relevant'], bool)
                assert isinstance(result['confidence'], float)
                assert 0 <= result['confidence'] <= 1
                
            # Step 4: Save and load model
            classifier.save_model('test_model.pkl')
            assert os.path.exists('test_model.pkl')
            
            # Load model in new classifier
            new_classifier = EmailClassifier()
            new_classifier.load_model('test_model.pkl')
            
            # Verify loaded model works
            result = new_classifier.predict("Test", "Test content")
            assert isinstance(result, dict)
            assert 'is_relevant' in result
            
        finally:
            os.chdir(original_cwd)
    
    def test_app_integration_with_real_components(self, temp_dir):
        """Test app integration with real components"""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Create test dataset
            from data_generator import EmailDataGenerator
            generator = EmailDataGenerator()
            df = generator.generate_dataset(50)
            df.to_csv('emails_dataset.csv', index=False)
            
            # Import and test app
            from app import EmailFilterApp
            
            with patch('gradio.Blocks'):
                app = EmailFilterApp()
                
                # Test training
                with patch.object(app, 'create_category_chart', return_value=None):
                    with patch.object(app, 'create_relevance_chart', return_value=None):
                        result, _, _ = app.generate_and_train(50)
                        
                        assert "✅ Modelo treinado com sucesso!" in result
                        assert app.is_model_loaded == True
                
                # Test classification
                classification_result = app.classify_email("Reunião", "Importante")
                assert "🟢" in classification_result or "🔴" in classification_result
                
                # Test model stats
                stats_result = app.get_model_stats()
                assert "📊 **Estatísticas do Modelo:**" in stats_result
                
                # Test feature importance
                features_result = app.get_feature_importance()
                assert "🔍 **Palavras mais importantes" in features_result
                
        finally:
            os.chdir(original_cwd)
    
    def test_csv_batch_processing_integration(self, temp_dir):
        """Test CSV batch processing integration"""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Create and train model
            from data_generator import EmailDataGenerator
            from email_classifier import EmailClassifier
            
            generator = EmailDataGenerator()
            df = generator.generate_dataset(100)
            
            classifier = EmailClassifier()
            classifier.train(df)
            classifier.save_model('test_model.pkl')
            
            # Create test CSV for batch processing
            test_emails = pd.DataFrame({
                'subject': [
                    'Reunião urgente',
                    'Desconto especial',
                    'Convite aniversário',
                    'Ganhe dinheiro'
                ],
                'content': [
                    'Precisamos discutir projeto',
                    'Promoção de fim de ano',
                    'Festa no sábado',
                    'Clique aqui para ganhar'
                ]
            })
            test_emails.to_csv('test_batch.csv', index=False)
            
            # Test batch processing through app
            from app import EmailFilterApp
            
            with patch('gradio.Blocks'):
                app = EmailFilterApp()
                
                # Mock file object
                mock_file = Mock()
                mock_file.name = 'test_batch.csv'
                
                analysis, results_df = app.analyze_batch(mock_file)
                
                # Verify results
                assert "📊 **Análise do Lote:**" in analysis
                assert "Total de emails: 4" in analysis
                assert isinstance(results_df, pd.DataFrame)
                assert len(results_df) == 4
                assert 'subject' in results_df.columns
                assert 'is_relevant' in results_df.columns
                assert 'confidence' in results_df.columns
                
        finally:
            os.chdir(original_cwd)
    
    def test_data_consistency_across_modules(self, temp_dir):
        """Test data consistency across different modules"""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Generate data with generator
            from data_generator import EmailDataGenerator
            generator = EmailDataGenerator()
            df = generator.generate_dataset(100)
            
            # Check data structure
            assert 'subject' in df.columns
            assert 'content' in df.columns
            assert 'is_relevant' in df.columns
            
            # Train classifier with generated data
            from email_classifier import EmailClassifier
            classifier = EmailClassifier()
            
            # Should handle the data without errors
            accuracy = classifier.train(df)
            assert 0 <= accuracy <= 1
            
            # Test predictions on generated data
            for _, row in df.head(5).iterrows():
                result = classifier.predict(row['subject'], row['content'])
                assert isinstance(result, dict)
                assert 'is_relevant' in result
                
        finally:
            os.chdir(original_cwd)
    
    def test_model_persistence_integration(self, temp_dir):
        """Test model persistence across sessions"""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Session 1: Train and save model
            from data_generator import EmailDataGenerator
            from email_classifier import EmailClassifier
            
            generator = EmailDataGenerator()
            df = generator.generate_dataset(100)
            
            classifier1 = EmailClassifier()
            accuracy1 = classifier1.train(df)
            classifier1.save_model('persistent_model.pkl')
            
            # Make a prediction
            result1 = classifier1.predict("Test subject", "Test content")
            
            # Session 2: Load model and use it
            classifier2 = EmailClassifier()
            classifier2.load_model('persistent_model.pkl')
            
            # Should be able to make predictions
            result2 = classifier2.predict("Test subject", "Test content")
            
            # Results should be identical
            assert result1['is_relevant'] == result2['is_relevant']
            assert result1['confidence'] == result2['confidence']
            
        finally:
            os.chdir(original_cwd)
    
    def test_error_handling_integration(self, temp_dir):
        """Test error handling across the system"""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            from app import EmailFilterApp
            
            with patch('gradio.Blocks'):
                app = EmailFilterApp()
                
                # Test various error scenarios
                
                # 1. Classification without trained model
                result = app.classify_email("Test", "Test")
                assert "⚠️" in result
                
                # 2. Batch analysis without trained model
                mock_file = Mock()
                mock_file.name = 'nonexistent.csv'
                result = app.analyze_batch(mock_file)
                assert "⚠️" in result or "❌" in result
                
                # 3. Model stats without trained model
                result = app.get_model_stats()
                assert "⚠️" in result
                
                # 4. Feature importance without trained model
                result = app.get_feature_importance()
                assert "⚠️" in result
                
        finally:
            os.chdir(original_cwd)
    
    def test_portuguese_text_processing_integration(self, temp_dir):
        """Test Portuguese text processing integration"""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            from data_generator import EmailDataGenerator
            from email_classifier import EmailClassifier
            
            # Create data with Portuguese text
            generator = EmailDataGenerator()
            df = generator.generate_dataset(100)
            
            # Check that Portuguese text is properly generated
            portuguese_words = ['reunião', 'projeto', 'promoção', 'desconto']
            found_portuguese = False
            for _, row in df.iterrows():
                combined_text = f"{row['subject']} {row['content']}".lower()
                if any(word in combined_text for word in portuguese_words):
                    found_portuguese = True
                    break
            
            assert found_portuguese
            
            # Train classifier
            classifier = EmailClassifier()
            classifier.train(df)
            
            # Test with Portuguese text
            result = classifier.predict("Reunião de projeto", "Discussão sobre implementação")
            assert isinstance(result, dict)
            assert 'is_relevant' in result
            
        finally:
            os.chdir(original_cwd)
    
    def test_performance_integration(self, temp_dir):
        """Test performance with larger dataset"""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            from data_generator import EmailDataGenerator
            from email_classifier import EmailClassifier
            
            # Generate larger dataset
            generator = EmailDataGenerator()
            df = generator.generate_dataset(1000)
            
            # Train classifier
            classifier = EmailClassifier()
            accuracy = classifier.train(df)
            
            # Should achieve reasonable accuracy
            assert accuracy >= 0.7  # At least 70% accuracy
            
            # Test prediction speed
            import time
            start_time = time.time()
            
            for i in range(100):
                classifier.predict(f"Test subject {i}", f"Test content {i}")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete 100 predictions in reasonable time
            assert total_time < 10  # Less than 10 seconds
            
        finally:
            os.chdir(original_cwd)
    
    def test_chart_generation_integration(self, temp_dir):
        """Test chart generation integration"""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            from data_generator import EmailDataGenerator
            from app import EmailFilterApp
            
            # Generate data
            generator = EmailDataGenerator()
            df = generator.generate_dataset(100)
            
            with patch('gradio.Blocks'):
                app = EmailFilterApp()
                app.dataset = df
                
                # Test chart generation
                with patch('plotly.express.pie') as mock_pie:
                    mock_pie.return_value = "mock_pie_chart"
                    category_chart = app.create_category_chart()
                    assert category_chart == "mock_pie_chart"
                
                with patch('plotly.express.bar') as mock_bar:
                    mock_bar.return_value = "mock_bar_chart"
                    relevance_chart = app.create_relevance_chart()
                    assert relevance_chart == "mock_bar_chart"
                
        finally:
            os.chdir(original_cwd)
    
    def test_end_to_end_workflow(self, temp_dir):
        """Test complete end-to-end workflow"""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            from app import EmailFilterApp
            
            with patch('gradio.Blocks'):
                app = EmailFilterApp()
                
                # Step 1: Generate and train
                with patch.object(app, 'create_category_chart', return_value=None):
                    with patch.object(app, 'create_relevance_chart', return_value=None):
                        train_result, _, _ = app.generate_and_train(100)
                        assert "✅ Modelo treinado com sucesso!" in train_result
                
                # Step 2: Classify individual email
                classify_result = app.classify_email("Reunião importante", "Discussão sobre projeto")
                assert ("🟢" in classify_result or "🔴" in classify_result)
                
                # Step 3: Get model statistics
                stats_result = app.get_model_stats()
                assert "📊 **Estatísticas do Modelo:**" in stats_result
                
                # Step 4: Get feature importance
                features_result = app.get_feature_importance()
                assert "🔍 **Palavras mais importantes" in features_result
                
                # Step 5: Test batch processing
                test_csv = pd.DataFrame({
                    'subject': ['Test 1', 'Test 2'],
                    'content': ['Content 1', 'Content 2']
                })
                test_csv.to_csv('test_batch.csv', index=False)
                
                mock_file = Mock()
                mock_file.name = 'test_batch.csv'
                
                batch_result, batch_df = app.analyze_batch(mock_file)
                assert "📊 **Análise do Lote:**" in batch_result
                assert len(batch_df) == 2
                
        finally:
            os.chdir(original_cwd)
