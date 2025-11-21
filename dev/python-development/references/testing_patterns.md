# Python Testing Patterns and Strategies

## Overview

This guide provides comprehensive testing patterns and strategies for Python projects, covering unit tests, integration tests, mocking, and best practices.

## Testing Pyramid

### 1. Unit Tests (70%)
- Fast, isolated tests for individual functions/classes
- Test business logic in isolation
- Mock external dependencies

### 2. Integration Tests (20%)
- Test interactions between components
- Test database operations, API calls
- Use test databases/fixtures

### 3. End-to-End Tests (10%)
- Test complete user workflows
- Use actual services (when feasible)
- Focus on critical user paths

## pytest Patterns

### Fixtures

```python
# conftest.py
import pytest
from my_app import create_app
from my_app.database import db
from my_app.models import User

@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app(testing=True)
    with app.app_context():
        yield app

@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()

@pytest.fixture
def database(app):
    """Create test database."""
    db.create_all()
    yield db
    db.drop_all()

@pytest.fixture
def sample_user(database):
    """Create sample user for tests."""
    user = User(username="testuser", email="test@example.com")
    db.session.add(user)
    db.session.commit()
    return user

# Factory fixtures for creating test data
@pytest.fixture
def user_factory(database):
    def create_user(**kwargs):
        defaults = {"username": "testuser", "email": "test@example.com"}
        defaults.update(kwargs)
        user = User(**defaults)
        db.session.add(user)
        db.session.commit()
        return user
    return create_user
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("", ""),
    ("123", "123"),
])
def test_uppercase(input, expected):
    assert input.upper() == expected

# Parametrizing with fixtures
@pytest.mark.parametrize("user_fixture", ["admin_user", "regular_user"])
def test_user_permissions(user_fixture, request):
    user = request.getfixturevalue(user_fixture)
    assert user.has_permission() is True
```

### Markers and Test Organization

```python
# Custom markers
pytest_plugins = ["pytest_asyncio"]

# Markers for different test types
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

# Usage in tests
@pytest.mark.integration
def test_database_connection():
    pass

@pytest.mark.slow
@pytest.mark.parametrize("size", [10, 100, 1000])
def test_large_dataset_processing(size):
    pass

# Running specific test types
# pytest -m unit                    # Only unit tests
# pytest -m "not slow"              # Skip slow tests
# pytest -m "integration or slow"   # Integration or slow tests
```

## Mocking Patterns

### Using unittest.mock

```python
from unittest.mock import Mock, patch, MagicMock
import requests

# Basic mocking
def test_api_call():
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {"key": "value"}
        mock_get.return_value.status_code = 200

        result = fetch_api_data()

        mock_get.assert_called_once_with("https://api.example.com")
        assert result == {"key": "value"}

# Mocking specific methods
class TestUserService:
    @patch('my_app.services.user_service.Database')
    def test_create_user(self, mock_db):
        mock_db.add.return_value = None
        mock_db.commit.return_value = None

        user = create_user("testuser", "test@example.com")

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        assert user.username == "testuser"

# Using MagicMock for complex objects
def test_file_processing():
    mock_file = MagicMock()
    mock_file.read.return_value = "file content"
    mock_file.name = "test.txt"

    result = process_file(mock_file)

    mock_file.read.assert_called_once()
    assert result == "processed: file content"
```

### Context Manager Mocking

```python
from unittest.mock import patch

def test_database_transaction():
    with patch('my_app.database.db.session.begin') as mock_begin:
        with patch('my_app.database.db.session.commit') as mock_commit:
            with patch('my_app.database.db.session.rollback') as mock_rollback:

                # Your transaction code here
                try:
                    perform_database_operation()
                except Exception:
                    db.session.rollback()
                    raise

                mock_begin.assert_called_once()
                mock_commit.assert_called_once()
                mock_rollback.assert_not_called()
```

## Testing Different Components

### API Testing

```python
def test_api_endpoints(client):
    """Test REST API endpoints."""
    # Test GET
    response = client.get('/api/users')
    assert response.status_code == 200
    assert isinstance(response.json, list)

    # Test POST
    user_data = {"username": "newuser", "email": "new@example.com"}
    response = client.post('/api/users', json=user_data)
    assert response.status_code == 201
    assert response.json["username"] == "newuser"

    # Test validation
    invalid_data = {"username": ""}  # Missing required fields
    response = client.post('/api/users', json=invalid_data)
    assert response.status_code == 400

def test_authentication(client, sample_user):
    """Test authentication endpoints."""
    # Test login
    response = client.post('/auth/login', json={
        "username": "testuser",
        "password": "password123"
    })
    assert response.status_code == 200
    assert "access_token" in response.json

    # Test protected endpoint
    token = response.json["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get('/api/profile', headers=headers)
    assert response.status_code == 200
```

### Database Testing

```python
def test_database_models(database):
    """Test database model operations."""
    # Create
    user = User(username="testuser", email="test@example.com")
    database.session.add(user)
    database.session.commit()

    # Read
    retrieved_user = User.query.filter_by(username="testuser").first()
    assert retrieved_user is not None
    assert retrieved_user.email == "test@example.com"

    # Update
    retrieved_user.email = "newemail@example.com"
    database.session.commit()
    updated_user = User.query.get(retrieved_user.id)
    assert updated_user.email == "newemail@example.com"

    # Delete
    database.session.delete(retrieved_user)
    database.session.commit()
    deleted_user = User.query.get(retrieved_user.id)
    assert deleted_user is None
```

### Async Testing

```python
import pytest
import asyncio
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/async-endpoint")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function("test_input")
    assert result == "expected_output"
```

## Test Organization

### Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── test_models.py
│   ├── test_services.py
│   └── test_utils.py
├── integration/             # Integration tests
│   ├── test_api.py
│   ├── test_database.py
│   └── test_external_services.py
├── e2e/                     # End-to-end tests
│   ├── test_user_workflows.py
│   └── test_admin_workflows.py
├── fixtures/                # Test data files
│   ├── sample_data.json
│   └── test_images/
└── helpers/                 # Test utilities
    ├── test_helpers.py
    └── assertions.py
```

### Test Helpers and Utilities

```python
# tests/helpers/test_helpers.py
import json
from pathlib import Path

def load_test_data(filename):
    """Load test data from JSON file."""
    data_path = Path(__file__).parent.parent / "fixtures" / filename
    with open(data_path) as f:
        return json.load(f)

def assert_valid_response(response, expected_status=200):
    """Assert response has valid structure."""
    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/json")

def create_test_user(**kwargs):
    """Create a test user with default values."""
    defaults = {
        "username": "testuser",
        "email": "test@example.com",
        "is_active": True
    }
    defaults.update(kwargs)
    return User(**defaults)

# tests/helpers/assertions.py
def assert_user_data(actual, expected):
    """Assert user data matches expected values."""
    assert actual["username"] == expected["username"]
    assert actual["email"] == expected["email"]
    assert "password" not in actual  # Password should not be in response

def assert_error_response(response, expected_error, expected_status=400):
    """Assert error response has correct format."""
    assert response.status_code == expected_status
    assert "error" in response.json
    assert expected_error in response.json["error"]
```

## Performance Testing

```python
import time
import pytest

@pytest.mark.slow
def test_performance_large_dataset():
    """Test performance with large dataset."""
    start_time = time.time()

    result = process_large_dataset(10000)

    end_time = time.time()
    execution_time = end_time - start_time

    # Should complete within 5 seconds
    assert execution_time < 5.0
    assert len(result) == 10000

# Memory testing
import tracemalloc

def test_memory_usage():
    """Test memory usage doesn't grow excessively."""
    tracemalloc.start()

    # Run operation multiple times
    for _ in range(100):
        process_data()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Should not use more than 100MB
    assert peak < 100 * 1024 * 1024  # 100MB in bytes
```

## Property-Based Testing

```python
import hypothesis
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=100))
def test_text_processing(input_text):
    """Test text processing with various inputs."""
    result = process_text(input_text)

    # Properties that should always hold
    assert isinstance(result, str)
    assert len(result) >= 0
    # If input was uppercase, result should be uppercase
    if input_text.isupper():
        assert result.isupper()

@given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
def test_calculator_operations(x, y):
    """Test calculator with property-based approach."""
    # Addition should be commutative
    assert add(x, y) == add(y, x)

    # Addition with zero should return the other number
    assert add(x, 0) == x
    assert add(0, y) == y
```

## Best Practices

### 1. Test Naming Conventions

```python
# Good: Descriptive test names
def test_user_creation_with_valid_data_succeeds():
    pass

def test_user_creation_with_duplicate_email_fails():
    pass

# Bad: Generic test names
def test_user():
    pass

def test_creation():
    pass
```

### 2. AAA Pattern (Arrange, Act, Assert)

```python
def test_user_email_validation():
    # Arrange
    user_data = {"username": "testuser", "email": "invalid-email"}

    # Act
    with pytest.raises(ValueError) as exc_info:
        create_user(user_data)

    # Assert
    assert "Invalid email format" in str(exc_info.value)
```

### 3. Test Isolation

```python
# Each test should be independent
def test_feature_a():
    # Setup specific to this test
    # Test feature A
    # Cleanup (use fixtures for automatic cleanup)

def test_feature_b():
    # Setup specific to this test
    # Test feature B
    # Don't rely on state from test_feature_a
```

### 4. Meaningful Assertions

```python
# Good: Specific assertions
assert response.status_code == 200
assert response.json["user"]["username"] == "expected_username"
assert "error" not in response.json

# Bad: Generic assertions
assert response is not None
assert result  # Too generic
```

### 5. Error Testing

```python
def test_specific_error_scenarios():
    # Test specific exceptions
    with pytest.raises(ValueError) as exc_info:
        function_with_validation(invalid_input)

    assert str(exc_info.value) == "Expected error message"

# Test error responses
def test_api_error_handling(client):
    response = client.post("/api/invalid-endpoint", json={})
    assert response.status_code == 404
    assert "not found" in response.json["error"].lower()
```

## Running Tests Effectively

### Commands

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_user.py

# Run specific test function
uv run pytest tests/test_user.py::test_user_creation

# Run tests with specific markers
uv run pytest -m unit                    # Only unit tests
uv run pytest -m "not slow"              # Skip slow tests
uv run pytest -m "integration or slow"   # Integration or slow tests

# Run tests in parallel
uv run pytest -n auto                    # Auto-detect CPU cores
uv run pytest -n 4                       # Use 4 processes

# Stop on first failure
uv run pytest -x

# Run failed tests only
uv run pytest --lf

# Show detailed output
uv run pytest -v                         # Verbose
uv run pytest -s                         # Show print statements
uv run pytest --tb=long                  # Detailed traceback
```

### Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```