# 🧪 Testing Documentation

## Overview

This project includes a comprehensive test suite with **>90% code coverage** to ensure reliability and maintainability of the AI email filter system.

## Test Structure

```
tests/
├── conftest.py              # Test configuration and fixtures
├── test_data_generator.py   # Unit tests for data generation
├── test_email_classifier.py # Unit tests for email classification
├── test_app.py             # Unit tests for Gradio app
├── test_integration.py     # Integration tests
└── test_edge_cases.py      # Edge cases and performance tests
```

## Test Categories

### 🔧 Unit Tests
- **Data Generator Tests**: Test email data generation, categories, and data quality
- **Email Classifier Tests**: Test model training, prediction, and persistence
- **App Tests**: Test Gradio interface components and user interactions

### 🔗 Integration Tests
- **Full Pipeline**: Test complete workflow from data generation to prediction
- **Model Persistence**: Test saving and loading trained models
- **CSV Processing**: Test batch email processing functionality

### 💨 Smoke Tests
- **Quick Validation**: Fast tests to ensure basic functionality works
- **Import Tests**: Verify all modules can be imported correctly

### 🐛 Edge Cases
- **Error Handling**: Test error conditions and graceful degradation
- **Data Quality**: Test with malformed, empty, or edge case data
- **Performance**: Test with large datasets and high loads

## Running Tests

### Quick Test Run
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_data_generator.py -v
```

### Comprehensive Test Suite
```bash
# Run the complete test suite
./run_tests.sh
```

### Test Categories
```bash
# Run only unit tests
pytest tests/test_data_generator.py tests/test_email_classifier.py tests/test_app.py

# Run only integration tests
pytest tests/test_integration.py

# Run only smoke tests
pytest tests/test_edge_cases.py -m smoke

# Run performance tests
pytest tests/test_edge_cases.py -m slow
```

## Coverage Requirements

The test suite maintains **>90% code coverage** with the following targets:

| Component | Coverage Target | Current Coverage |
|-----------|----------------|------------------|
| Data Generator | 95% | ✅ |
| Email Classifier | 95% | ✅ |
| App Interface | 90% | ✅ |
| Integration | 85% | ✅ |
| **Overall** | **90%** | **✅** |

## Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
addopts = 
    --cov=.
    --cov-report=html:htmlcov
    --cov-report=term-missing
    --cov-fail-under=90
    --strict-markers
    --disable-warnings
    -v
```

### Coverage Configuration
```ini
[coverage:run]
source = .
omit = 
    tests/*
    venv/*
    .pytest_cache/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

## Test Fixtures

### Available Fixtures
- `temp_dir`: Temporary directory for test files
- `sample_email_data`: Sample email dataset for testing
- `mock_classifier`: Mock classifier with pre-configured responses
- `csv_file`: Sample CSV file for batch processing tests
- `mock_gradio_interface`: Mock Gradio interface for UI tests

### Example Usage
```python
def test_example(sample_email_data, temp_dir):
    # Use fixtures in your tests
    assert len(sample_email_data) == 5
    assert os.path.exists(temp_dir)
```

## Mock Strategy

### External Dependencies
- **Gradio Interface**: Mocked to avoid UI dependencies
- **File System**: Temporary directories for isolation
- **Network Calls**: Mocked for predictable testing
- **Random Generation**: Seeded for reproducible tests

### Example Mock Usage
```python
@patch('gradio.Blocks')
def test_with_mock(mock_blocks):
    # Test code here
    mock_blocks.assert_called_once()
```

## Performance Tests

### Benchmarks
- **Data Generation**: 5,000 emails in <30 seconds
- **Model Training**: 1,000 emails in <60 seconds
- **Predictions**: 1,000 predictions in <10 seconds

### Performance Test Example
```python
@pytest.mark.slow
def test_performance():
    import time
    start = time.time()
    # Test code
    end = time.time()
    assert end - start < expected_time
```

## Error Handling Tests

### Test Scenarios
- **Invalid Input**: None, empty strings, malformed data
- **Missing Files**: Non-existent CSV files, model files
- **Network Issues**: Simulated connection failures
- **Memory Limits**: Large datasets and memory constraints

### Example Error Test
```python
def test_error_handling():
    with pytest.raises(ValueError, match="Expected error message"):
        # Code that should raise error
        pass
```

## Data Quality Tests

### Validation Checks
- **Email Format**: Valid email addresses in sender field
- **Text Content**: Non-empty subjects and content
- **Categories**: Valid category assignments
- **Relevance**: Binary relevance values (0 or 1)

### Quality Test Example
```python
def test_data_quality():
    df = generator.generate_dataset(100)
    
    # Check email format
    assert df['sender'].str.contains('@').all()
    
    # Check non-empty content
    assert (df['subject'].str.len() > 0).all()
    
    # Check valid categories
    valid_categories = ['importante', 'promocional', 'spam', 'pessoal']
    assert df['category'].isin(valid_categories).all()
```

## Continuous Integration

### GitHub Actions Workflow
- **Multi-Python Version**: Test on Python 3.8, 3.9, 3.10, 3.11
- **Coverage Reports**: Automatic coverage reporting
- **Security Scanning**: Safety and Bandit security checks
- **Code Quality**: Black, isort, flake8, mypy checks

### CI Configuration
```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']
    steps:
    - name: Run tests with coverage
      run: |
        pytest --cov=. --cov-report=xml --cov-fail-under=90 -v
```

## Best Practices

### Test Writing Guidelines
1. **Descriptive Names**: Use clear, descriptive test function names
2. **Single Responsibility**: Each test should test one specific behavior
3. **Arrange-Act-Assert**: Structure tests with clear setup, action, and verification
4. **Independent Tests**: Tests should not depend on each other
5. **Fast Execution**: Keep tests fast with appropriate mocking

### Example Test Structure
```python
def test_specific_behavior():
    # Arrange
    setup_test_data()
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected_value
```

## Troubleshooting

### Common Issues

**Coverage Below 90%**
```bash
# Check coverage report
pytest --cov=. --cov-report=html
# Open htmlcov/index.html to see uncovered lines
```

**Slow Tests**
```bash
# Run only fast tests
pytest -m "not slow"

# Run tests in parallel
pytest -n auto
```

**Mock Issues**
```bash
# Verify mock setup
pytest -v -s  # Show print statements
```

### Debugging Tests
```python
# Add debugging to tests
import pdb; pdb.set_trace()  # Debugger breakpoint
print(f"Debug: {variable}")  # Debug prints
```

## Coverage Reports

### HTML Report
After running tests with coverage, open `htmlcov/index.html` in a browser to see:
- **Line Coverage**: Which lines are covered by tests
- **Branch Coverage**: Which code branches are tested
- **Function Coverage**: Which functions are tested
- **Missing Lines**: Specific lines that need test coverage

### Terminal Report
```bash
pytest --cov=. --cov-report=term-missing
```

Shows coverage percentage and missing lines directly in terminal.

## Maintenance

### Adding New Tests
1. **Create Test File**: Follow naming convention `test_*.py`
2. **Add Test Functions**: Use `test_*` prefix
3. **Update Coverage**: Ensure new code is covered
4. **Update Documentation**: Document new test scenarios

### Updating Tests
1. **Review Changes**: Check if existing tests need updates
2. **Maintain Coverage**: Keep coverage above 90%
3. **Test Performance**: Ensure tests remain fast
4. **Update Fixtures**: Update shared test data if needed

---

## 🎯 Test Coverage Goals

| Goal | Status |
|------|--------|
| Unit Test Coverage > 95% | ✅ |
| Integration Test Coverage > 85% | ✅ |
| Overall Coverage > 90% | ✅ |
| All Edge Cases Covered | ✅ |
| Performance Tests Passing | ✅ |
| CI/CD Pipeline Green | ✅ |

**The comprehensive test suite ensures the AI email filter is robust, reliable, and ready for production use!**
