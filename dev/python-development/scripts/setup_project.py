#!/usr/bin/env python3
"""
Python Project Setup Script

Automates the creation of a new Python project with proper structure,
virtual environment, and development tools configuration.

Usage:
    python setup_project.py <project_name> [--project-type <type>]

Project types:
    standard - Basic Python package structure (default)
    fastapi  - FastAPI web application
    django   - Django web application
    cli      - CLI tool project
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def create_directory_structure(project_path: Path, project_type: str) -> None:
    """Create the basic directory structure for the project."""
    directories = [
        "src",
        "tests",
        "docs",
        "scripts",
    ]

    if project_type == "fastapi":
        directories.extend([
            "src/app",
            "src/app/models",
            "src/app/routes",
            "src/app/services",
            "src/app/middleware",
        ])
    elif project_type == "django":
        directories.extend([
            "src/settings",
            "src/apps",
            "static",
            "templates",
        ])
    elif project_type == "cli":
        directories.extend([
            "src/cli",
            "src/commands",
        ])

    for directory in directories:
        (project_path / directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_initial_files(project_path: Path, project_name: str, project_type: str) -> None:
    """Create initial project files."""

    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "tests/__init__.py",
    ]

    if project_type == "fastapi":
        init_files.extend([
            "src/app/__init__.py",
            "src/app/models/__init__.py",
            "src/app/routes/__init__.py",
            "src/app/services/__init__.py",
            "src/app/middleware/__init__.py",
        ])
    elif project_type == "django":
        init_files.extend([
            "src/settings/__init__.py",
            "src/apps/__init__.py",
        ])
    elif project_type == "cli":
        init_files.extend([
            "src/cli/__init__.py",
            "src/commands/__init__.py",
        ])

    for init_file in init_files:
        (project_path / init_file).touch()
        print(f"✅ Created file: {init_file}")

    # Create main module
    if project_type == "standard":
        main_content = f'''"""
{project_name}

Main module for the {project_name} package.
"""

__version__ = "0.1.0"


def main():
    """Main entry point for the application."""
    print(f"Hello from {project_name}!")


if __name__ == "__main__":
    main()
'''
        (project_path / "src" / project_name.replace("-", "_") / "__init__.py").write_text(main_content)

    elif project_type == "fastapi":
        main_content = f'''"""
FastAPI Application for {project_name}
"""

from fastapi import FastAPI

app = FastAPI(
    title="{project_name.replace("-", " ").title()}",
    description="FastAPI application",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {{"message": "Welcome to {project_name}"}}


@app.get("/health")
async def health_check():
    return {{"status": "healthy"}}
'''
        (project_path / "src" / "app" / "main.py").write_text(main_content)

    elif project_type == "cli":
        main_content = f'''"""
CLI application for {project_name}
"""

import click


@click.group()
def cli():
    """Main CLI interface for {project_name}."""
    pass


@cli.command()
def hello():
    """Say hello."""
    click.echo("Hello from {project_name}!")


@cli.command()
@click.option("--name", default="World", help="Name to greet")
def greet(name):
    """Greet someone."""
    click.echo(f"Hello, {{name}}!")


if __name__ == "__main__":
    cli()
'''
        (project_path / "src" / "cli" / "main.py").write_text(main_content)


def create_pyproject_toml(project_path: Path, project_name: str, project_type: str) -> None:
    """Create pyproject.toml configuration file."""

    base_config = f'''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "A Python project"
readme = "README.md"
requires-python = ">=3.8"
license = {{text = "MIT"}}
authors = [
    {{name = "Your Name", email = "your.email@example.com"}}
]
keywords = []
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

[project.urls]
Homepage = "https://github.com/yourusername/{project_name}"
Repository = "https://github.com/yourusername/{project_name}"
Issues = "https://github.com/yourusername/{project_name}/issues"

[project.scripts]
{project_name} = "{project_name.replace("-", "_")}:main"
'''

    if project_type == "fastapi":
        dependencies = [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "pydantic>=2.0.0",
        ]
    elif project_type == "django":
        dependencies = [
            "django>=4.2.0",
            "django-environ>=0.11.0",
        ]
    elif project_type == "cli":
        dependencies = [
            "click>=8.0.0",
        ]
    else:
        dependencies = []

    dev_dependencies = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-xdist>=3.0.0",  # Parallel execution
        "pytest-mock>=3.10.0",  # Mocking utilities
        "ruff>=0.1.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ]

    if dependencies:
        base_config += f'\n[project.dependencies]\n'
        for dep in dependencies:
            base_config += f'"{dep}"\n'

    base_config += f'\n[project.optional-dependencies]\ndev = [\n'
    for dep in dev_dependencies:
        base_config += f'    "{dep}",\n'
    base_config += ']\n'

    # Tool configurations
    tool_config = '''
[tool.ruff]
line-length = 88
target-version = "py38"
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "S",      # flake8-bandit (security)
]
ignore = [
    "E501",   # line too long, handled by formatter
    "B008",   # do not perform function calls in argument defaults
    "C901",   # too complex
    "S101",   # use of assert detected
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.ruff.lint.isort]
known-first-party = ["src"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = [
    "S",      # flake8-bandit (security)
    "ARG",    # flake8-unused-arguments
]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

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
    "--cov-fail-under=80",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]
'''

    base_config += tool_config
    (project_path / "pyproject.toml").write_text(base_config)
    print("✅ Created pyproject.toml")


def create_gitignore(project_path: Path) -> None:
    """Create .gitignore file."""
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
"""

    (project_path / ".gitignore").write_text(gitignore_content)
    print("✅ Created .gitignore")


def create_precommit_config(project_path: Path) -> None:
    """Create pre-commit configuration."""
    precommit_config = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
"""

    (project_path / ".pre-commit-config.yaml").write_text(precommit_config)
    print("✅ Created .pre-commit-config.yaml")


def create_readme(project_path: Path, project_name: str, project_type: str) -> None:
    """Create README.md file."""

    readme_content = f"""# {project_name.replace('-', ' ').title()}

A Python project.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/{project_name}.git
cd {project_name}

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -e ".[dev]"
```

## Usage

"""

    if project_type == "standard":
        readme_content += """```python
from {project_name.replace('-', '_')} import main

main()
```
"""
    elif project_type == "fastapi":
        readme_content += """```bash
# Run the development server
uvicorn src.app.main:app --reload

# Or using the script
python -m src.app.main
```

The API will be available at `http://localhost:8000`
"""
    elif project_type == "cli":
        readme_content += """```bash
# Run the CLI
{project_name} --help

# Example commands
{project_name} hello
{project_name} greet --name "Alice"
```
"""

    readme_content += """
## Development

```bash
# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Code formatting
black src tests

# Import sorting
isort src tests

# Type checking
mypy src
```

## License

MIT License
"""

    (project_path / "README.md").write_text(readme_content)
    print("✅ Created README.md")


def create_basic_tests(project_path: Path, project_name: str, project_type: str) -> None:
    """Create basic test files."""

    if project_type == "standard":
        test_content = f'''"""
Tests for {project_name} using pytest.
"""

import pytest
from {project_name.replace("-", "_")} import main


class TestMainModule:
    """Test suite for the main module."""

    def test_main_function_output(self, capsys):
        """Test that main function produces expected output."""
        main()
        captured = capsys.readouterr()
        assert "Hello from {project_name}" in captured.out

    def test_main_function_complete_output(self, capsys):
        """Test complete output from main function."""
        main()
        captured = capsys.readouterr()
        expected_message = "Hello from {project_name}!"
        assert expected_message in captured.out

    @pytest.mark.parametrize("scenario", [
        "first_run",
        "subsequent_run",
    ])
    def test_main_function_different_scenarios(self, capsys, scenario):
        """Test main function under different scenarios."""
        # Add scenario-specific setup if needed
        main()
        captured = capsys.readouterr()
        assert captured.out is not None
        assert len(captured.out) > 0
'''
    elif project_type == "fastapi":
        test_content = '''"""
Tests for FastAPI application using pytest.
"""

from fastapi.testclient import TestClient
from src.app.main import app

# Fixture for test client
@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


class TestRootEndpoints:
    """Test suite for root API endpoints."""

    def test_root_endpoint_success(self, client):
        """Test the root endpoint returns success response."""
        response = client.get("/")

        assert response.status_code == 200
        response_json = response.json()
        assert "message" in response_json
        assert "Welcome" in response_json["message"]

    def test_root_endpoint_response_structure(self, client):
        """Test that root endpoint has correct response structure."""
        response = client.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_health_check_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "healthy"

    @pytest.mark.parametrize("endpoint", ["/", "/health"])
    def test_endpoints_return_json(self, client, endpoint):
        """Test that endpoints return JSON responses."""
        response = client.get(endpoint)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
'''
    elif project_type == "cli":
        test_content = '''"""
Tests for CLI application using pytest.
"""

import pytest
from click.testing import CliRunner
from src.cli.main import cli


class TestCLICommands:
    """Test suite for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Click test runner fixture."""
        return CliRunner()

    def test_cli_help_command(self, runner):
        """Test that CLI help command works."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_hello_command_success(self, runner):
        """Test hello command executes successfully."""
        result = runner.invoke(cli, ["hello"])

        assert result.exit_code == 0
        assert "Hello" in result.output

    def test_greet_command_with_name(self, runner):
        """Test greet command with custom name."""
        result = runner.invoke(cli, ["greet", "--name", "Alice"])

        assert result.exit_code == 0
        assert "Alice" in result.output

    @pytest.mark.parametrize("name", ["Alice", "Bob", "Charlie"])
    def test_greet_command_multiple_names(self, runner, name):
        """Test greet command with multiple different names."""
        result = runner.invoke(cli, ["greet", "--name", name])

        assert result.exit_code == 0
        assert name in result.output

    def test_greet_command_default_name(self, runner):
        """Test greet command with default name."""
        result = runner.invoke(cli, ["greet"])

        assert result.exit_code == 0
        assert "World" in result.output
'''
    else:
        test_content = '''"""
Basic pytest tests and examples.
"""

import pytest


class TestBasicFunctionality:
    """Test suite with basic pytest examples."""

    def test_basic_assertion(self):
        """Basic assertion test."""
        assert True

    def test_math_operations(self):
        """Test basic math operations."""
        assert 1 + 1 == 2
        assert 10 - 5 == 5
        assert 3 * 4 == 12
        assert 8 / 2 == 4

    @pytest.mark.parametrize("input_val,expected", [
        (2, 4),
        (3, 9),
        (4, 16),
        (5, 25),
    ])
    def test_square_function(self, input_val, expected):
        """Test square calculation with parameterized inputs."""
        assert input_val ** 2 == expected

    def test_exception_raising(self):
        """Test that expected exceptions are raised."""
        with pytest.raises(ValueError):
            raise ValueError("Test error")

    def test_warning_capture(self, recwarn):
        """Test that warnings are captured properly."""
        import warnings
        warnings.warn("Test warning", UserWarning)

        assert len(recwarn) == 1
        assert issubclass(recwarn[0].category, UserWarning)
'''

    (project_path / "tests" / f"test_{project_name.replace('-', '_')}.py").write_text(test_content)
    print(f"✅ Created tests/test_{project_name.replace('-', '_')}.py")


def setup_project_with_uv(project_path: Path) -> bool:
    """Set up project using uv."""
    print("\\n🔧 Setting up project with uv...")

    try:
        # Initialize with uv
        subprocess.run(["uv", "init"], cwd=project_path, check=True)
        print("✅ Initialized project with uv")

        # Install dependencies from pyproject.toml
        print("📦 Installing dependencies...")
        result = subprocess.run(["uv", "sync"],
                               cwd=project_path, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Installed dependencies")
        else:
            print("⚠️  Dependency installation failed, you may need to install manually:")
            print(f"   {result.stderr}")

        # Install pre-commit hooks
        print("🔗 Installing pre-commit hooks...")
        result = subprocess.run(["uv", "run", "pre-commit", "install"],
                               cwd=project_path, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Installed pre-commit hooks")
        else:
            print("⚠️  Pre-commit installation failed, run 'uv run pre-commit install' manually")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error setting up project with uv: {e}")
        return False


def setup_virtual_environment(project_path: Path) -> bool:
    """Set up virtual environment and install dependencies (legacy method)."""
    print("\\n🔧 Setting up virtual environment...")

    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", "venv"],
                      cwd=project_path, check=True)
        print("✅ Created virtual environment")

        # Install dependencies (this might fail if the project has complex dependencies)
        print("📦 Installing dependencies...")
        result = subprocess.run(["venv/bin/pip", "install", "-e", ".[dev]"],
                               cwd=project_path, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Installed dependencies")
        else:
            print("⚠️  Dependency installation failed, you may need to install manually:")
            print(f"   {result.stderr}")

        # Install pre-commit hooks
        print("🔗 Installing pre-commit hooks...")
        result = subprocess.run(["venv/bin/pre-commit", "install"],
                               cwd=project_path, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Installed pre-commit hooks")
        else:
            print("⚠️  Pre-commit installation failed, run 'pre-commit install' manually")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error setting up environment: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Set up a new Python project")
    parser.add_argument("project_name", help="Name of the project (kebab-case)")
    parser.add_argument("--project-type", choices=["standard", "fastapi", "django", "cli"],
                       default="standard", help="Type of project to create")
    parser.add_argument("--no-venv", action="store_true",
                       help="Skip virtual environment setup")
    parser.add_argument("--use-uv", action="store_true", default=True,
                       help="Use uv package manager (recommended)")

    args = parser.parse_args()

    project_path = Path(args.project_name)

    if project_path.exists():
        print(f"❌ Directory '{args.project_name}' already exists!")
        sys.exit(1)

    print(f"🚀 Creating Python project: {args.project_name}")
    print(f"   Type: {args.project_type}")
    print(f"   Path: {project_path.absolute()}")
    print(f"   Package manager: {'uv' if args.use_uv else 'pip/venv'}")
    print()

    # Create project structure
    create_directory_structure(project_path, args.project_type)
    create_initial_files(project_path, args.project_name, args.project_type)

    # Create configuration files
    create_pyproject_toml(project_path, args.project_name, args.project_type)
    create_gitignore(project_path)
    create_precommit_config(project_path)
    create_readme(project_path, args.project_name, args.project_type)
    create_basic_tests(project_path, args.project_name, args.project_type)

    print(f"\\n✅ Project '{args.project_name}' created successfully!")

    # Set up environment if requested
    if not args.no_venv:
        if args.use_uv:
            setup_project_with_uv(project_path)
        else:
            setup_virtual_environment(project_path)

    print("\\nNext steps:")
    print(f"1. cd {args.project_name}")

    if args.use_uv:
        print("2. uv sync                              # Install dependencies")
        print("3. uv run pytest                         # Run tests")
        print("4. uv run pre-commit run --all-files     # Check code quality")
        print("5. uv run python -m src.main             # Run your project")
    else:
        if args.no_venv:
            print("2. python -m venv venv && source venv/bin/activate  # Linux/Mac")
            print("   venv\\Scripts\\activate  # Windows")
            print("3. pip install -e '.[dev]'")
        else:
            print("2. source venv/bin/activate  # Linux/Mac")
            print("   venv\\Scripts\\activate  # Windows")
        print("4. pytest")
        print("5. pre-commit run --all-files")

    print("\\nDevelopment commands:")
    if args.use_uv:
        print("  uv add package_name              # Add dependency")
        print("  uv add --dev package_name        # Add dev dependency")
        print("  uv run command                   # Run command in project env")
        print("  uv sync                          # Update dependencies")
    else:
        print("  pip install package_name         # Install dependency")
        print("  pip install -e '.[dev]'          # Install in development mode")


if __name__ == "__main__":
    main()