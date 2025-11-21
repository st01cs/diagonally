# pytest Best Practices and Comprehensive Guide

This guide covers pytest best practices, patterns, and advanced testing techniques for Python projects.

## Table of Contents
- [Installation and Setup](#installation-and-setup)
- [Basic pytest Concepts](#basic-pytest-concepts)
- [Fixtures](#fixtures)
- [Parametrization](#parametrization)
- [Markers](#markers)
- [Mocking and Patching](#mocking-and-patching)
- [Async Testing](#async-testing)
- [Database Testing](#database-testing)
- [API Testing](#api-testing)
- [Test Organization](#test-organization)
- [Advanced pytest Features](#advanced-pytest-features)
- [Performance Testing](#performance-testing)
- [CI/CD Integration](#cicd-integration)

## Installation and Setup

### Basic Installation
```bash
# Core pytest
uv add --dev pytest

# Common plugins
uv add --dev pytest-cov      # Coverage reporting
uv add --dev pytest-asyncio  # Async testing support
uv add --dev pytest-xdist    # Parallel test execution
uv add --dev pytest-mock     # Mocking utilities
uv add --dev pytest-benchmark # Performance testing
```

### Configuration (pyproject.toml)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under=80",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "api: marks tests as API tests",
    "database: marks tests that require database",
    "external: marks tests that require external services",
    "smoke: marks critical smoke tests",
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]
```

## Basic pytest Concepts

### Test Discovery
- pytest automatically discovers test files named `test_*.py`
- Test classes should start with `Test`
- Test functions should start with `test_`

### Assertions
```python
# Simple assertions
assert result == expected_value
assert condition is True
assert len(items) > 0

# With custom messages
assert user.is_active, f"User {user.id} should be active"

# Checking for exceptions
with pytest.raises(ValueError, match="Expected error message"):
    function_that_raises_value_error()

# Checking warnings
with pytest.warns(UserWarning, match="Warning message"):
    function_that_issues_warning()
```

### Basic Test Structure
```python
import pytest
from my_module import Calculator

class TestCalculator:
    """Test suite for Calculator class."""

    def setup_method(self):
        """Setup for each test method."""
        self.calc = Calculator()

    def teardown_method(self):
        """Cleanup after each test method."""
        if hasattr(self, 'calc'):
            del self.calc

    def test_addition(self):
        """Test basic addition."""
        result = self.calc.add(2, 3)
        assert result == 5

    def test_division_by_zero(self):
        """Test division by zero raises error."""
        with pytest.raises(ZeroDivisionError):
            self.calc.divide(10, 0)
```

## Fixtures

### Basic Fixtures
```python
import pytest
from my_app.database import DatabaseConnection
from my_app.models import User

@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return User(
        id=1,
        name="Test User",
        email="test@example.com",
        is_active=True
    )

@pytest.fixture
def database_connection():
    """Create a database connection for testing."""
    conn = DatabaseConnection("sqlite:///:memory:")
    conn.create_tables()
    yield conn
    conn.close()

# Using fixtures in tests
def test_user_creation(sample_user):
    """Test user object creation."""
    assert sample_user.name == "Test User"
    assert sample_user.email == "test@example.com"

def test_database_operations(database_connection):
    """Test database operations."""
    user = User(name="DB User", email="db@example.com")
    database_connection.save(user)

    retrieved_user = database_connection.get_user(user.id)
    assert retrieved_user.name == "DB User"
```

### Fixture Scopes
```python
@pytest.fixture(scope="function")  # Default - new for each test
def function_fixture():
    pass

@pytest.fixture(scope="class")     # Once per test class
def class_fixture():
    pass

@pytest.fixture(scope="module")    # Once per module
def module_fixture():
    pass

@pytest.fixture(scope="session")   # Once per test session
def session_fixture():
    pass
```

### Parametrized Fixtures
```python
@pytest.fixture(params=[1, 2, 3])
def number_param(request):
    """Fixture that returns different numbers."""
    return request.param

def test_with_parametrized_fixture(number_param):
    """Test with different number values."""
    assert number_param in [1, 2, 3]

# Using ids for better test names
@pytest.fixture(params=[
    ("valid_email@example.com", True),
    ("invalid-email", False),
    ("", False),
], ids=["valid", "invalid_format", "empty"])
def email_validation_data(request):
    """Parametrized fixture with custom IDs."""
    return request.param

def test_email_validation(email_validation_data):
    """Test email validation with various inputs."""
    email, expected_result = email_validation_data
    assert validate_email(email) == expected_result
```

## Parametrization

### Basic Parametrization
```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (2, 4),      # 2 squared = 4
    (3, 9),      # 3 squared = 9
    (4, 16),     # 4 squared = 16
    (5, 25),     # 5 squared = 25
])
def test_square_calculation(input, expected):
    """Test square calculation with various inputs."""
    assert input ** 2 == expected

# Multiple parameters
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (2, 3, 5),
    (10, -5, 5),
    (0, 0, 0),
])
def test_addition(a, b, expected):
    """Test addition with various inputs."""
    assert a + b == expected
```

### Parametrization with IDs
```python
@pytest.mark.parametrize(
    "user_type,permissions",
    [
        ("admin", ["read", "write", "delete"]),
        ("editor", ["read", "write"]),
        ("viewer", ["read"]),
    ],
    ids=["admin_user", "editor_user", "viewer_user"]
)
def test_user_permissions(user_type, permissions):
    """Test permissions for different user types."""
    user = create_user(user_type)
    assert user.permissions == permissions
```

### Combining Parametrization
```python
@pytest.mark.parametrize("operation", ["add", "subtract", "multiply"])
@pytest.mark.parametrize("a,b", [(1, 2), (3, 4)])
def test_calculator_operations(operation, a, b):
    """Test calculator with multiple parametrizations."""
    calc = Calculator()
    if operation == "add":
        result = calc.add(a, b)
    elif operation == "subtract":
        result = calc.subtract(a, b)
    elif operation == "multiply":
        result = calc.multiply(a, b)

    # Add appropriate assertions based on operation
    assert result is not None
```

## Markers

### Built-in Markers
```python
import pytest

@pytest.mark.skip(reason="Feature not implemented yet")
def test_unimplemented_feature():
    """Skip this test."""
    pass

@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="Requires Python 3.8+"
)
def test_python_38_feature():
    """Skip on older Python versions."""
    pass

@pytest.mark.xfail
def test_known_bug():
    """Expected to fail."""
    assert False  # Known bug

@pytest.mark.xfail(
    reason="Bug #123: Known issue with database"
)
def test_database_bug():
    """Expected to fail due to known bug."""
    pass

@pytest.mark.parametrize("input", [1, 2, 3])
def test_with_marked_parametrize(input):
    """Test with parameterization."""
    assert input > 0
```

### Custom Markers
```python
import pytest

@pytest.mark.slow
def test_slow_operation():
    """Mark test as slow."""
    import time
    time.sleep(5)

@pytest.mark.integration
def test_database_integration():
    """Mark test as integration test."""
    # Database integration test
    pass

@pytest.mark.api
def test_api_endpoint():
    """Mark test as API test."""
    # API test
    pass

# Using markers in command line
# pytest -m "not slow"      # Run all tests except slow ones
# pytest -m integration     # Run only integration tests
# pytest -m "api or slow"  # Run API or slow tests
```

## Mocking and Patching

### Basic Mocking
```python
from unittest.mock import Mock, patch, MagicMock
import requests

def test_with_mock_object():
    """Test using mock objects."""
    # Create mock
    mock_service = Mock()
    mock_service.get_user.return_value = {"id": 1, "name": "Test User"}

    # Use mock in test
    user = mock_service.get_user(1)
    assert user["name"] == "Test User"

    # Assert mock was called correctly
    mock_service.get_user.assert_called_once_with(1)

def test_with_patch_decorator():
    """Test using patch decorator."""
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"result": "success"}

        response = fetch_api_data()

        assert response["result"] == "success"
        mock_get.assert_called_once_with("https://api.example.com/data")

def test_with_patch_context_manager():
    """Test using patch as context manager."""
    with patch('my_module.external_service') as mock_service:
        mock_service.process_data.return_value = "processed"

        result = process_external_data()

        assert result == "processed"
        mock_service.process_data.assert_called_once()
```

### Advanced Mocking
```python
def test_with_side_effect():
    """Test mock with side effect."""
    mock_func = Mock()
    mock_func.side_effect = [1, 2, 3]  # Returns different values

    assert mock_func() == 1
    assert mock_func() == 2
    assert mock_func() == 3

def test_with_exception_side_effect():
    """Test mock that raises exception."""
    mock_func = Mock()
    mock_func.side_effect = ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        mock_func()

def test_spec_mock():
    """Create mock that follows another object's specification."""
    from my_module import RealService

    # Mock that only has methods from RealService
    mock_service = Mock(spec=RealService)
    mock_service.process_data.return_value = "processed"

    # This works - method exists in spec
    result = mock_service.process_data({"data": "test"})

    # This would raise AttributeError - method not in spec
    # mock_service.nonexistent_method()
```

## Async Testing

### Basic Async Tests
```python
import pytest
import asyncio
from httpx import AsyncClient
from my_app.main import app

@pytest.mark.asyncio
async def test_async_function():
    """Test asynchronous function."""
    result = await async_add_numbers(2, 3)
    assert result == 5

@pytest.mark.asyncio
async def test_async_with_fixture(async_client):
    """Test async with fixture."""
    response = await async_client.post("/api/users", json={
        "name": "Test User",
        "email": "test@example.com"
    })

    assert response.status_code == 201

# Async fixture
@pytest.fixture
async def async_client():
    """Async test client fixture."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
```

### Async Context Managers
```python
@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async context manager."""
    async with AsyncDatabaseConnection() as db:
        user = await db.create_user("Test User")
        assert user.id is not None

    # Database connection is automatically closed
```

## Database Testing

### SQLAlchemy Testing
```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from my_app.models import Base, User

@pytest.fixture(scope="function")
def test_db():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

class TestUserModel:
    """Test suite for User model."""

    def test_create_user(self, test_db):
        """Test user creation in database."""
        user = User(name="Test User", email="test@example.com")
        test_db.add(user)
        test_db.commit()

        retrieved_user = test_db.query(User).filter_by(id=user.id).first()
        assert retrieved_user.name == "Test User"
        assert retrieved_user.email == "test@example.com"

    def test_user_validation(self, test_db):
        """Test user model validation."""
        # Test invalid email
        with pytest.raises(Exception):  # Depending on your validation
            user = User(name="Test", email="invalid-email")
            test_db.add(user)
            test_db.commit()
```

### Factory Pattern for Tests
```python
@pytest.fixture
def user_factory(test_db):
    """Factory fixture for creating users."""
    def create_user(**kwargs):
        defaults = {
            "name": "Test User",
            "email": "test@example.com",
            "is_active": True
        }
        defaults.update(kwargs)

        user = User(**defaults)
        test_db.add(user)
        test_db.commit()
        return user

    return create_user

def test_with_user_factory(user_factory):
    """Test using user factory."""
    # Create user with defaults
    user1 = user_factory()

    # Create user with custom values
    user2 = user_factory(name="Custom User", is_active=False)

    assert user1.name == "Test User"
    assert user2.name == "Custom User"
    assert user2.is_active is False
```

## API Testing

### FastAPI Testing
```python
from fastapi.testclient import TestClient
from my_app.main import app

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Authentication headers for testing."""
    return {"Authorization": "Bearer test_token"}

class TestUserAPI:
    """Test suite for user API endpoints."""

    def test_create_user(self, client):
        """Test user creation endpoint."""
        user_data = {
            "name": "API Test User",
            "email": "api-test@example.com"
        }
        response = client.post("/api/users", json=user_data)

        assert response.status_code == 201
        response_json = response.json()
        assert response_json["name"] == "API Test User"
        assert response_json["email"] == "api-test@example.com"

    def test_get_user(self, client, sample_user):
        """Test getting user by ID."""
        response = client.get(f"/api/users/{sample_user.id}")

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["id"] == sample_user.id

    def test_protected_endpoint(self, client, auth_headers):
        """Test protected endpoint with authentication."""
        response = client.get("/api/protected", headers=auth_headers)

        assert response.status_code == 200

    def test_unauthorized_access(self, client):
        """Test unauthorized access to protected endpoint."""
        response = client.get("/api/protected")

        assert response.status_code == 401
```

### API Testing with Mocks
```python
def test_api_with_external_service_mocked(client):
    """Test API endpoint with external service mocked."""
    with patch('my_app.services.email_service.send_email') as mock_email:
        mock_email.return_value = True

        response = client.post("/api/users", json={
            "name": "Test User",
            "email": "test@example.com"
        })

        assert response.status_code == 201
        mock_email.assert_called_once_with("test@example.com", "Welcome!")
```

## Test Organization

### Test Structure
```
tests/
├── conftest.py              # Shared fixtures and configuration
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

### conftest.py Example
```python
import pytest
from my_app.database import get_test_db
from my_app.models import User

@pytest.fixture(scope="session")
def test_db():
    """Test database for entire session."""
    db = get_test_db()
    yield db
    db.cleanup()

@pytest.fixture
def sample_user(test_db):
    """Sample user fixture."""
    user = User(name="Test User", email="test@example.com")
    test_db.add(user)
    test_db.commit()
    return user

@pytest.fixture
def authenticated_client(client, sample_user):
    """Client with authenticated user."""
    client.force_login(sample_user)
    return client
```

### Test Helpers
```python
# tests/helpers/assertions.py
def assert_valid_user_response(response_data):
    """Assert user response has valid structure."""
    assert "id" in response_data
    assert "name" in response_data
    assert "email" in response_data
    assert response_data["email"].count("@") == 1

def assert_error_response(response, expected_status, expected_message=None):
    """Assert error response format."""
    assert response.status_code == expected_status
    if expected_message:
        assert expected_message in response.json()["error"]

# tests/helpers/test_helpers.py
import json
from pathlib import Path

def load_test_data(filename):
    """Load test data from JSON file."""
    data_path = Path(__file__).parent.parent / "fixtures" / filename
    with open(data_path) as f:
        return json.load(f)
```

## Advanced pytest Features

### Parametrized Fixtures with IDs
```python
@pytest.fixture(
    params=[
        ("user", ["read", "write"]),
        ("admin", ["read", "write", "delete"]),
        ("guest", ["read"]),
    ],
    ids=["user_role", "admin_role", "guest_role"]
)
def user_with_permissions(request):
    """Fixture that returns users with different permissions."""
    role, permissions = request.param
    return create_user_with_role(role, permissions)
```

### Indirect Parametrization
```python
@pytest.fixture
def input_data(request):
    """Fixture that receives parametrized input."""
    return request.param

@pytest.mark.parametrize("input_data", [1, 2, 3], indirect=True)
def test_with_indirect_fixture(input_data):
    """Test using indirectly parametrized fixture."""
    assert input_data in [1, 2, 3]
```

### Custom Markers
```python
# Define custom marker in conftest.py
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

# Use in tests
@pytest.mark.slow
def test_slow_operation():
    """Mark test as slow."""
    pass
```

## Performance Testing

### Benchmark Testing
```python
import pytest
from my_app import expensive_function

@pytest.mark.benchmark
def test_expensive_function_performance(benchmark):
    """Benchmark expensive function."""
    result = benchmark(expensive_function, test_data)
    assert result is not None

# Custom benchmark assertion
@pytest.mark.benchmark
def test_performance_requirement(benchmark):
    """Test with performance requirements."""
    result = benchmark(expensive_function, test_data)
    assert result is not None

    # Access benchmark stats
    assert benchmark.stats.stats.mean < 1.0  # Should complete in < 1 second
```

### Memory Testing
```python
import tracemalloc

def test_memory_usage():
    """Test memory usage doesn't grow excessively."""
    tracemalloc.start()

    # Run operation multiple times
    for _ in range(100):
        result = memory_intensive_operation()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Should not use more than 100MB
    assert peak < 100 * 1024 * 1024
```

## CI/CD Integration

### GitHub Actions
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install uv
      uses: astral-sh/setup-uv@v2

    - name: Install dependencies
      run: uv sync --all-extras --dev

    - name: Run tests
      run: uv run pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Test Result Reports
```bash
# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html

# Generate JUnit XML for CI/CD
uv run pytest --junitxml=test-results.xml

# Generate detailed test report
uv run pytest --html=test-report.html --self-contained-html
```

### Running Tests in CI
```bash
# Run tests with different configurations
uv run pytest -m "not slow" --maxfail=1  # Fast tests, stop on first failure
uv run pytest -m integration --duration=10  # Integration tests with timeout
uv run pytest --dist=auto --numprocesses=auto  # Parallel execution
```

## Common pytest Commands

### Development Commands
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific file
uv run pytest tests/test_user.py

# Run specific test
uv run pytest tests/test_user.py::TestUserService::test_create_user

# Run with markers
uv run pytest -m unit                    # Only unit tests
uv run pytest -m "not slow"            # Skip slow tests
uv run pytest -m "integration or api"  # Integration or API tests

# Parallel execution
uv run pytest -n auto                   # Auto-detect CPU cores
uv run pytest -n 4                      # Use 4 processes

# Debug options
uv run pytest -v                        # Verbose output
uv run pytest -s                        # Show print statements
uv run pytest --tb=long                # Detailed traceback
uv run pytest -x                        # Stop on first failure
uv run pytest --lf                      # Run only failed tests

# Reporting
uv run pytest --html=report.html --self-contained-html
uv run pytest --junitxml=results.xml
```

### Test Performance Commands
```bash
# Profile test execution time
uv run pytest --durations=10

# Find slowest tests
uv run pytest --durations=0

# Run with coverage and generate reports
uv run pytest --cov=src --cov-report=html --cov-report=xml
```

## Best Practices Summary

### Test Structure
- Use descriptive test names that explain what is being tested
- Group related tests in classes
- Use fixtures for setup/teardown logic
- Apply appropriate markers for test categorization

### Assertions
- Be specific with assertions and messages
- Test both positive and negative cases
- Use parametrization for testing multiple scenarios
- Mock external dependencies for isolated testing

### Performance
- Use markers to identify slow tests
- Run tests in parallel for faster execution
- Use coverage reports to ensure adequate test coverage
- Regularly review and optimize test performance

### Maintenance
- Keep tests simple and focused
- Regularly update tests when code changes
- Use consistent naming conventions
- Document complex test scenarios with comments