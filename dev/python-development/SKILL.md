---
name: python-development
description: "Comprehensive Python development workflow support including project setup, package management, code quality, testing, debugging, documentation, and deployment. Use when Claude needs to work with Python projects for: (1) Setting up new Python projects with proper structure and virtual environments, (2) Managing dependencies and packages (pip, requirements.txt, poetry), (3) Implementing code quality tools and best practices (linting, formatting, type checking), (4) Creating and running tests (pytest, unittest), (5) Debugging Python code and troubleshooting common issues, (6) Setting up CI/CD pipelines and deployment configurations, (7) Performance optimization and security improvements"
---

# Python Development
Expert guidance for writing clean, efficient, and maintainable Python code following modern best practices and industry standards.

## Core Principles

### Code Style and Standards

**Follow PEP 8 style guide for code formatting**
- Use `ruff` to automatically enforce PEP 8 compliance
- Configure pre-commit hooks to catch style issues early
- Check specific PEP 8 rules: `uv run ruff check --select E,W`

**Use PEP 257 for docstring conventions**
```python
def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the Euclidean distance between two points.

    Args:
        x1: X coordinate of the first point
        y1: Y coordinate of the first point
        x2: X coordinate of the second point
        y2: Y coordinate of the second point

    Returns:
        The Euclidean distance between the two points.

    Raises:
        ValueError: If any coordinate is not a number.

    Example:
        >>> calculate_distance(0, 0, 3, 4)
        5.0
    """
    pass
```

**Implement type hints (PEP 484) for better code clarity**
```python
from typing import List, Dict, Optional, Union, Protocol
from dataclasses import dataclass
from enum import Enum

# Basic type hints
def process_data(items: List[str], config: Dict[str, Union[str, int]]) -> Optional[str]:
    """Process a list of items with configuration."""
    return None

# Protocol for interface definition
class DataProcessor(Protocol):
    def process(self, data: str) -> str: ...

# Enum for constants
class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

# Dataclass for structured data
@dataclass
class User:
    id: int
    name: str
    email: str
    status: Status = Status.PENDING
```

**Maximum line length: 88 characters (Black formatter standard)**
- Enforced by `ruff format`
- Use `uv run ruff format --check` to verify compliance
- Configure in pyproject.toml: `line-length = 88`

**Use meaningful variable and function names that convey intent**
```python
# Bad - unclear naming
def proc(d, c):
    x = d[0] + c['v']
    return x

# Good - clear, descriptive naming
def calculate_total_with_tax(item_price: float, tax_config: Dict[str, float]) -> float:
    base_price = item_price
    tax_rate = tax_config['rate']
    total_with_tax = base_price * (1 + tax_rate)
    return total_with_tax
```

## Overview

This skill provides comprehensive support for Python development workflows, from project initialization to deployment. It covers best practices, tooling, and common patterns used in professional Python development.

## Core Capabilities

### 1. Project Setup and Structure

**Use when:** Starting a new Python project or restructuring an existing one.

**Workflow:**
1. Determine project type and requirements
2. Set up virtual environment
3. Initialize project structure
4. Configure development tools

**Common project structures:**

```
# Standard Python package
my_project/
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── module.py
├── tests/
│   ├── __init__.py
│   └── test_module.py
├── pyproject.toml
├── README.md
├── requirements.txt
└── .gitignore

# FastAPI project
my_api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   ├── routes/
│   └── services/
├── tests/
├── requirements.txt
└── Dockerfile
```

**Virtual environment setup:**

```bash
# uv (recommended - modern, fast package manager)
uv init my-project
cd my-project
uv add package_name

# Python venv (legacy)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# conda environment
conda create -n myenv python=3.11
conda activate myenv
```

### 2. Package Management

**Use when:** Managing dependencies, resolving conflicts, or setting up reproducible environments.

**Package management tools:**

```bash
# uv (recommended - fast, modern package manager)
uv init my-project                    # Initialize new project
uv add package_name                   # Add main dependency
uv add --dev package_name             # Add development dependency
uv sync                              # Install dependencies
uv run command                       # Run commands in project environment
uv pip freeze > requirements.txt     # Generate requirements.txt if needed

# pip and requirements.txt (legacy)
pip install package_name
pip freeze > requirements.txt
pip install -r requirements.txt

# pip-tools for deterministic builds
pip-compile requirements.in
pip-sync requirements.txt
```

**Common essential packages:**

```bash
# Install all at once with uv
uv add python-dotenv click pydantic fastapi sqlalchemy requests
uv add --dev ruff mypy pytest pre-commit

# Development tools
ruff           # Code formatting, import sorting, and linting (all-in-one)
mypy           # Type checking
pytest         # Testing
pre-commit     # Git hooks

# Production packages
python-dotenv  # Environment variables
click          # CLI tools
pydantic       # Data validation
fastapi        # Web framework
sqlalchemy     # Database ORM
requests       # HTTP client
```

### 3. Code Quality Tools

**Use when:** Improving code quality, maintaining consistency, or setting up team standards.

**Configuration examples:**

**pyproject.toml (recommended):**
```toml
[tool.ruff]
line-length = 88
target-version = "py311"
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
]
ignore = [
    "E501",   # line too long, handled by black
    "B008",   # do not perform function calls in argument defaults
    "C901",   # too complex
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.ruff.lint.isort]
known-first-party = ["src"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

**Pre-commit hooks (.pre-commit-config.yaml):**
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

**Common ruff commands:**
```bash
# Check and fix issues
uv run ruff check src/ tests/ --fix

# Format code
uv run ruff format src/ tests/

# Run both check and format
uv run ruff check --fix . && uv run ruff format .

# Check specific rules
uv run ruff check src/ --select E,W,F
```

### 4. Testing Strategies

**Use when:** Writing tests, improving test coverage, or setting up testing infrastructure.

**pytest framework and patterns:**

```python
# pytest example - comprehensive testing patterns
import pytest
from typing import List
from my_module import calculate_sum, User, UserService

class TestCalculation:
    """Test suite for calculation functions."""

    def test_add_positive_numbers(self):
        """Test addition of positive numbers."""
        assert calculate_sum(2, 3) == 5

    def test_add_negative_numbers(self):
        """Test addition of negative numbers."""
        assert calculate_sum(-2, -3) == -5

    def test_add_mixed_numbers(self):
        """Test addition of mixed positive/negative numbers."""
        assert calculate_sum(5, -3) == 2

    @pytest.mark.parametrize("a,b,expected", [
        (1, 2, 3),
        (0, 0, 0),
        (-1, 1, 0),
        (10, -5, 5),
        (100, 200, 300),
    ])
    def test_add_various_numbers(self, a, b, expected):
        """Test addition with various number combinations."""
        assert calculate_sum(a, b) == expected

    @pytest.mark.parametrize("invalid_input", [
        None,
        "string",
        [1, 2],
        {"a": 1},
    ])
    def test_add_invalid_inputs(self, invalid_input):
        """Test that invalid inputs raise appropriate errors."""
        with pytest.raises(TypeError):
            calculate_sum(invalid_input, 1)

# Testing with fixtures
@pytest.fixture
def sample_user():
    """Fixture providing a sample user for testing."""
    return User(id=1, name="John Doe", email="john@example.com")

@pytest.fixture
def user_service():
    """Fixture providing a UserService instance."""
    return UserService()

class TestUserService:
    """Test suite for UserService class."""

    def test_create_user_success(self, user_service):
        """Test successful user creation."""
        user = user_service.create_user("Jane Smith", "jane@example.com")

        assert user.name == "Jane Smith"
        assert user.email == "jane@example.com"
        assert user.id is not None

    def test_create_duplicate_email(self, user_service, sample_user):
        """Test creating user with duplicate email raises error."""
        with pytest.raises(ValueError, match="Email already exists"):
            user_service.create_user(sample_user.name, sample_user.email)

    def test_get_user_by_id(self, user_service, sample_user):
        """Test retrieving user by ID."""
        found_user = user_service.get_user(sample_user.id)
        assert found_user.id == sample_user.id
        assert found_user.name == sample_user.name

# Async testing with pytest-asyncio
import asyncio
import httpx
from fastapi.testclient import TestClient

@pytest.mark.asyncio
async def test_async_user_creation():
    """Test asynchronous user creation."""
    result = await async_create_user("Test User", "test@example.com")
    assert result.id is not None
    assert result.email == "test@example.com"

# Testing with database
@pytest.fixture
async def test_db_session():
    """Fixture providing a test database session."""
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine
    from my_app.database import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

class TestUserDatabase:
    """Test suite for database operations."""

    def test_save_and_retrieve_user(self, test_db_session):
        """Test saving and retrieving user from database."""
        user = User(name="Test User", email="test@example.com")
        test_db_session.add(user)
        test_db_session.commit()

        retrieved_user = test_db_session.query(User).filter_by(id=user.id).first()
        assert retrieved_user is not None
        assert retrieved_user.name == "Test User"

# API testing with FastAPI
@pytest.fixture
def client():
    """FastAPI test client fixture."""
    from my_app.main import app
    return TestClient(app)

class TestUserAPI:
    """Test suite for user API endpoints."""

    def test_create_user_endpoint(self, client):
        """Test user creation API endpoint."""
        user_data = {
            "name": "API Test User",
            "email": "api-test@example.com"
        }
        response = client.post("/api/users", json=user_data)

        assert response.status_code == 201
        assert response.json()["name"] == "API Test User"
        assert response.json()["email"] == "api-test@example.com"

    def test_get_user_endpoint(self, client, sample_user):
        """Test getting user by ID via API."""
        response = client.get(f"/api/users/{sample_user.id}")

        assert response.status_code == 200
        assert response.json()["id"] == sample_user.id

# Mocking and patching
from unittest.mock import Mock, patch, MagicMock

class TestWithMocks:
    """Test suite using mocks and patches."""

    @patch('my_module.external_api_call')
    def test_with_external_service_mocked(self, mock_api_call):
        """Test function behavior with external API mocked."""
        mock_api_call.return_value = {"status": "success", "data": [1, 2, 3]}

        result = process_external_data()

        assert result == [1, 2, 3]
        mock_api_call.assert_called_once()

    def test_with_manual_mock(self):
        """Test using manual mock objects."""
        mock_service = Mock()
        mock_service.get_user.return_value = User(id=1, name="Mock User")

        handler = UserHandler(mock_service)
        user_info = handler.get_user_info(1)

        assert "Mock User" in user_info
        mock_service.get_user.assert_called_with(1)
```

**Test organization:**
```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests
├── e2e/           # End-to-end tests
├── fixtures/      # Test data
└── conftest.py    # Shared pytest configuration
```

**pytest configuration and commands:**

```bash
# Run all tests
uv run pytest

# Run with coverage (recommended)
uv run pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_user.py

# Run specific test class
uv run pytest tests/test_user.py::TestUserService

# Run specific test method
uv run pytest tests/test_user.py::TestUserService::test_create_user_success

# Run with markers for test organization
uv run pytest -m "not slow"                    # Skip slow tests
uv run pytest -m integration                   # Run only integration tests
uv run pytest -m unit                          # Run only unit tests
uv run pytest -m "unit or integration"         # Run unit and integration tests

# Run tests in parallel (faster execution)
uv run pytest -n auto                          # Auto-detect CPU cores
uv run pytest -n 4                             # Use 4 processes

# Stop on first failure (useful for debugging)
uv run pytest -x

# Run only failed tests from previous run
uv run pytest --lf

# Verbose output with detailed test names
uv run pytest -v

# Show captured output (useful for debugging)
uv run pytest -s

# Detailed traceback on errors
uv run pytest --tb=long

# Run tests with specific Python warnings
uv run pytest -W error::DeprecationWarning

# Generate JUnit XML report (for CI/CD)
uv run pytest --junitxml=test-results.xml
```

**pytest configuration in pyproject.toml:**
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
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]
```

### 5. Debugging and Troubleshooting

**Use when:** Investigating bugs, performance issues, or unexpected behavior.

**Debugging tools and techniques:**

```python
# pdb (built-in debugger)
import pdb; pdb.set_trace()

# More verbose debugging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Performance profiling
import cProfile
def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    # Your code here
    profiler.disable()
    profiler.print_stats(sort='cumulative')
```

**Common debugging scenarios:**

```bash
# Check installed packages and versions
uv pip list
uv pip freeze

# Check Python path
uv run python -c "import sys; print(sys.path)"

# Validate syntax
uv run python -m py_compile script.py

# Check imports
uv run python -c "import package_name; print(package_name.__file__)"

# Memory profiling
uv add --dev memory-profiler
uv run python -m memory_profiler script.py
```

### 6. Documentation

**Use when:** Writing documentation, setting up docstrings, or generating API docs.

**Docstring standards (Google style):**

```python
def calculate_area(radius: float) -> float:
    """Calculate the area of a circle.

    Args:
        radius: The radius of the circle in meters.

    Returns:
        The area of the circle in square meters.

    Raises:
        ValueError: If radius is negative.

    Example:
        >>> calculate_area(5)
        78.53981633974483
    """
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return 3.14159 * radius ** 2
```

**Documentation tools:**
```bash
# Sphinx for API documentation
uv add --dev sphinx
uv run sphinx-quickstart docs
make html

# MkDocs for project documentation
uv add --dev mkdocs
uv run mkdocs new docs
uv run mkdocs serve
```

### 7. Performance Optimization

**Use when:** Improving code performance, memory usage, or scalability.

**Common optimization techniques:**

```python
# Use generators for large datasets
def process_large_file(filename):
    with open(filename) as f:
        for line in f:
            yield process_line(line)

# Memoization with functools.lru_cache
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(x, y):
    # Expensive computation
    return result

# Use collections.deque for efficient appends/pops
from collections import deque
queue = deque()

# NumPy for numerical computations
import numpy as np
arr = np.array(data)  # Much faster than Python lists
```

### 8. Code Style and Standards

**Use when:** Ensuring consistent code quality, enforcing team standards, or improving code readability.

**Code Quality Tools and Commands:**

```bash
# Check all style issues
uv run ruff check .

# Fix auto-correctable issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Both check and format in one command
uv run ruff check --fix . && uv run ruff format .

# Check specific rule categories
uv run ruff check --select E,W,F  # PEP 8, pyflakes
uv run ruff check --select I       # Import sorting
uv run ruff check --select B       # Bugbear patterns
uv run ruff check --select UP      # pyupgrade

# View configuration
uv run ruff check --show-settings

# Diff formatter to see changes without modifying files
uv run ruff format --diff .
```

**Naming Conventions:**

```python
# Constants: UPPER_SNAKE_CASE
MAX_CONNECTIONS = 100
DEFAULT_TIMEOUT = 30

# Variables and functions: snake_case
user_name = "john_doe"
def calculate_total(price: float, tax_rate: float) -> float:

# Classes: PascalCase
class UserManager:
class DatabaseConnection:

# Private members: underscore prefix
class DataProcessor:
    def __init__(self):
        self._internal_data = []  # Private
        self.__very_private = 0   # Name mangled

# Modules: lowercase with optional underscores
# file: user_manager.py
# file: database_connection.py
```

**Import Organization:**

```python
# Import order (enforced by ruff isort rules)
# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Optional

# 2. Third-party imports
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# 3. Local imports
from .models import User
from .utils import format_name

# 4. Relative imports (for same package)
from . import config
from ..services import email_service
```

**Error Handling Patterns:**

```python
# Specific exception handling
def process_user_data(user_id: int) -> Dict[str, str]:
    try:
        user = get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        processed_data = transform_user_data(user)
        return processed_data

    except ValueError as e:
        logger.warning(f"Data processing error: {e}")
        raise
    except DatabaseError as e:
        logger.error(f"Database error processing user {user_id}: {e}")
        raise ServiceUnavailableError("Unable to process user data") from e
    finally:
        cleanup_resources()
```

**Code Documentation Standards:**

```python
class DataAnalyzer:
    """Analyzes data patterns and generates insights.

    This class provides methods for statistical analysis of datasets,
    including trend detection and anomaly identification.

    Attributes:
        data: The dataset to analyze
        config: Configuration parameters for analysis

    Example:
        >>> analyzer = DataAnalyzer([1, 2, 3, 4, 5])
        >>> result = analyzer.calculate_mean()
        >>> print(result)  # 3.0
    """

    def __init__(self, data: List[float], config: Optional[Dict] = None):
        """Initialize the analyzer with data and configuration.

        Args:
            data: List of numerical values to analyze
            config: Optional configuration dictionary

        Raises:
            ValueError: If data is empty or contains non-numeric values
        """
        if not data:
            raise ValueError("Data cannot be empty")
        if not all(isinstance(x, (int, float)) for x in data):
            raise ValueError("All data points must be numeric")

        self.data = data
        self.config = config or {}
```

**String Formatting:**

```python
# Preferred: f-strings (Python 3.6+)
name = "Alice"
age = 30
message = f"User {name} is {age} years old"

# For complex formatting: .format()
template = "User: {name}, Age: {age}, Status: {status}"
formatted = template.format(name="Bob", age=25, status="active")

# For logging: % formatting (performance)
logger.info("Processing user %d with status %s", user_id, status)
```

**List Comprehensions and Generators:**

```python
# Good: Simple comprehensions
squares = [x**2 for x in range(10)]
even_numbers = [x for x in numbers if x % 2 == 0]

# Good: Generator expressions for large datasets
large_squares = (x**2 for x in range(1000000))
total = sum(large_squares)

# Avoid: Complex nested comprehensions (use regular loops)
# Bad: result = [x*y for x in range(10) for y in range(5) if x*y > 10]
# Good: Write clear, nested loops instead
```

### 9. Security Best Practices

**Use when:** Reviewing code for security issues or implementing secure patterns.

**Security guidelines:**

```python
# Input validation
import re
from pydantic import BaseModel, EmailStr

class UserInput(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_]+$')

# Secure password handling
import hashlib
import secrets

def hash_password(password: str, salt: str = None) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()

# Avoid SQL injection
# Bad: cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
# Good: cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

# Environment variables for secrets
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('API_KEY')
```

## Common Development Patterns

### Error Handling

```python
# Specific exception handling
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    handle_value_error(e)
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    handle_connection_error(e)
else:
    process_result(result)
finally:
    cleanup_resources()
```

### Context Managers

```python
# Custom context manager
from contextlib import contextmanager

@contextmanager
def database_connection():
    conn = create_connection()
    try:
        yield conn
    finally:
        conn.close()

# Usage
with database_connection() as conn:
    results = conn.execute("SELECT * FROM users")
```

### Configuration Management

```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    api_key: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
```

## Resources

This skill includes practical scripts and references for common Python development tasks.

### scripts/

**Project initialization and automation scripts:**
- `setup_project.py` - Automated project structure creation
- `install_dev_tools.py` - Install and configure development tools
- `run_quality_checks.py` - Execute all code quality checks
- `deploy_project.py` - Deployment automation script

### references/

**Comprehensive guides and documentation:**
- `code_style_patterns.md` - Comprehensive code style patterns, PEP guidelines, and best practices
- `pytest_best_practices.md` - Complete pytest testing framework guide with advanced patterns
- `testing_patterns.md` - Detailed testing strategies and patterns
- `security_checklist.md` - Security best practices and checklist

**Quick reference guides:**
- pytest fixtures, parametrization, and mocking
- Import organization and naming conventions
- Error handling patterns
- Type hints best practices
- Documentation standards

### assets/

**Template files and boilerplate:**
- `project_templates/` - Starting templates for different project types
- `config_templates/` - Configuration file templates (pyproject.toml, etc.)
- `docker_templates/` - Docker configuration templates
- `github_actions/` - CI/CD workflow templates

---

**Best Practices:**
- Use uv for modern, fast package management (preferred) or traditional virtual environments
- Use ruff as an all-in-one tool for linting, formatting, and import sorting (10-100x faster than alternatives)
- Write tests before writing code (TDD) or immediately after
- Use type hints for better code clarity and IDE support
- Run code quality tools before committing (`uv run pre-commit run --all-files`)
- Use `uv run ruff check --fix . && uv run ruff format .` for quick code quality fixes
- Document public APIs and complex business logic
- Keep dependencies minimal and well-maintained
- Use `uv sync` to ensure reproducible environments
- Leverage `uv run` to execute commands in the project environment