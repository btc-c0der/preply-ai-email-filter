#!/bin/bash

# Test runner script for comprehensive testing

echo "🧪 Starting comprehensive test suite..."

# Install test dependencies
echo "📦 Installing test dependencies..."
pip install -r requirements-test.txt

# Run different test categories
echo "🔍 Running unit tests..."
pytest tests/test_data_generator.py tests/test_email_classifier.py tests/test_app.py -v --tb=short

echo "🔗 Running integration tests..."
pytest tests/test_integration.py -v --tb=short

echo "💨 Running smoke tests..."
pytest tests/test_edge_cases.py -m smoke -v --tb=short

echo "🐛 Running edge case tests..."
pytest tests/test_edge_cases.py -v --tb=short

echo "📊 Running full test suite with coverage..."
pytest --cov=. --cov-report=html --cov-report=term-missing --cov-fail-under=90 -v

echo "🎯 Running performance tests (slow)..."
pytest tests/test_edge_cases.py -m slow -v --tb=short

echo "✅ Test suite completed!"
echo "📈 Check htmlcov/index.html for detailed coverage report"
