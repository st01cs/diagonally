# Python Code Style Patterns and Best Practices

This guide provides comprehensive patterns for writing clean, maintainable Python code following industry best practices.

## Table of Contents
- [PEP Guidelines Overview](#pep-guidelines-overview)
- [Naming Conventions](#naming-conventions)
- [Import Organization](#import-organization)
- [Code Structure](#code-structure)
- [Documentation Standards](#documentation-standards)
- [Type Hints Best Practices](#type-hints-best-practices)
- [Error Handling Patterns](#error-handling-patterns)
- [Code Organization](#code-organization)
- [Performance Considerations](#performance-considerations)

## PEP Guidelines Overview

### PEP 8 (Style Guide)
- **Line length**: 88 characters (Black standard)
- **Indentation**: 4 spaces per level
- **Blank lines**: 2 blank lines before top-level functions, 1 before method definitions
- **Imports**: At top of file, grouped by type
- **Whitespace**: Around operators and after commas

### PEP 257 (Docstring Conventions)
- **One-liners**: For simple functions
- **Multi-line docstrings**: For complex functions and classes
- **Sections**: Use Args, Returns, Raises, Example sections

### PEP 484 (Type Hints)
- **Basic types**: `int`, `str`, `float`, `bool`
- **Collections**: `List[T]`, `Dict[K, V]`, `Set[T]`
- **Optional**: `Optional[T]` for nullable types
- **Union**: `Union[T1, T2]` for multiple types

## Naming Conventions

### Variable and Function Naming
```python
# Good: Descriptive, snake_case
user_name = "john_doe"
calculate_total_price(item_price, tax_rate)
process_user_registration_data(user_input)

# Bad: Unclear or abbreviated
un = "jd"
calc_tp(ip, tr)
proc_usr_reg_dt(uid)

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"

# Boolean variables: Start with is_, has_, can_, should_
is_authenticated = True
has_permission = False
can_edit = True
should_retry = True
```

### Class and Module Naming
```python
# Classes: PascalCase
class UserManager:
    pass

class DatabaseConnectionPool:
    pass

class HTTPRequestHandler:
    pass

# Modules: lowercase_with_underscores
# file: user_manager.py
# file: database_connection.py
# file: http_request_handler.py

# Packages: lowercase
# directory: mypackage/
```

### Private and Protected Members
```python
class DataProcessor:
    def __init__(self):
        self.public_data = []      # Public
        self._protected_data = []   # Protected (convention)
        self.__private_data = []    # Private (name mangled)

    def public_method(self):
        """Public method - accessible to everyone."""
        pass

    def _protected_method(self):
        """Protected method - for internal use, subclasses can access."""
        pass

    def __private_method(self):
        """Private method - name mangled, only accessible within class."""
        pass
```

## Import Organization

### Standard Import Order
```python
# 1. Standard library imports
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Union

# 2. Third-party imports
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String

# 3. Local imports
from .models import User, Product
from .utils import format_name, validate_email
from .config import get_settings

# 4. Relative imports (for same package)
from . import database
from ..services import email_service
from ..utils.decorators import timing_decorator
```

### Import Best Practices
```python
# Good: Import specific items
from typing import List, Dict
from requests import get, post

# Avoid: Wildcard imports (except in __init__.py)
# Bad: from module import *

# Good: Use alias for long names
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection

# Good: Group related imports
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
```

## Code Structure

### Function Structure
```python
def process_user_data(
    user_id: int,
    data: Dict[str, any],
    config: Optional[Dict[str, str]] = None,
) -> Dict[str, any]:
    """
    Process and validate user data according to configuration.

    Args:
        user_id: Unique identifier for the user
        data: Raw user data dictionary
        config: Optional processing configuration

    Returns:
        Processed and validated user data

    Raises:
        ValueError: If user data is invalid
        KeyError: If required fields are missing
    """
    # Validate input
    if not user_id:
        raise ValueError("User ID cannot be empty")

    required_fields = ["name", "email"]
    for field in required_fields:
        if field not in data:
            raise KeyError(f"Missing required field: {field}")

    # Process data
    processed_data = {
        "user_id": user_id,
        "name": data["name"].strip().title(),
        "email": data["email"].lower(),
    }

    # Apply configuration if provided
    if config:
        processed_data.update(_apply_config(processed_data, config))

    return processed_data
```

### Class Structure
```python
class OrderProcessor:
    """
    Processes e-commerce orders with validation and business logic.

    Handles order creation, validation, payment processing, and notification.
    """

    def __init__(self, database_client: DatabaseClient, email_service: EmailService):
        """Initialize with required dependencies."""
        self.db = database_client
        self.email = email_service
        self.logger = logging.getLogger(__name__)

    def create_order(self, user_id: int, items: List[OrderItem]) -> Order:
        """
        Create a new order for a user with the specified items.

        Args:
            user_id: ID of the user placing the order
            items: List of items to include in the order

        Returns:
            Created order object

        Raises:
            InsufficientStockError: If items are out of stock
            InvalidOrderError: If order data is invalid
        """
        # Validate order
        self._validate_items(items)
        self._validate_user(user_id)

        # Calculate totals
        subtotal = sum(item.total_price for item in items)
        tax = self._calculate_tax(subtotal)
        total = subtotal + tax

        # Create order
        order = Order(
            user_id=user_id,
            items=items,
            subtotal=subtotal,
            tax=tax,
            total=total,
            status=OrderStatus.PENDING,
        )

        # Save to database
        saved_order = self.db.save_order(order)

        # Send confirmation
        self.email.send_order_confirmation(saved_order)

        self.logger.info(f"Created order {saved_order.id} for user {user_id}")
        return saved_order

    def _validate_items(self, items: List[OrderItem]) -> None:
        """Validate that all items are in stock and valid."""
        for item in items:
            if not item.is_available():
                raise InsufficientStockError(f"Item {item.product_id} out of stock")

    def _validate_user(self, user_id: int) -> None:
        """Validate that the user exists and is active."""
        user = self.db.get_user(user_id)
        if not user or not user.is_active:
            raise InvalidOrderError(f"Invalid user: {user_id}")

    def _calculate_tax(self, amount: float) -> float:
        """Calculate tax based on current tax rates."""
        tax_rate = self.db.get_tax_rate()
        return amount * tax_rate
```

## Documentation Standards

### Function Documentation
```python
def calculate_compound_interest(
    principal: float,
    annual_rate: float,
    times_compounded: int,
    years: int,
) -> float:
    """
    Calculate compound interest for an investment.

    Uses the formula: A = P(1 + r/n)^(nt)
    where A = final amount, P = principal, r = annual rate,
    n = times compounded per year, t = time in years.

    Args:
        principal: Initial investment amount (must be positive)
        annual_rate: Annual interest rate as decimal (e.g., 0.05 for 5%)
        times_compounded: Number of times interest is compounded per year
        years: Number of years to calculate interest for

    Returns:
        Final amount after compound interest

    Raises:
        ValueError: If any parameter is negative or zero

    Example:
        >>> calculate_compound_interest(1000, 0.05, 12, 5)
        1283.36
        >>> calculate_compound_interest(500, 0.07, 4, 10)
        983.58
    """
    if principal <= 0:
        raise ValueError("Principal must be positive")
    if annual_rate <= 0:
        raise ValueError("Annual rate must be positive")
    if times_compounded <= 0:
        raise ValueError("Times compounded must be positive")
    if years <= 0:
        raise ValueError("Years must be positive")

    final_amount = principal * (1 + annual_rate / times_compounded) ** (times_compounded * years)
    return round(final_amount, 2)
```

### Class Documentation
```python
class InventoryManager:
    """
    Manages product inventory with tracking and alerts.

    Provides functionality for tracking stock levels, handling stock updates,
    generating inventory reports, and sending low stock alerts.

    Attributes:
        database: Database connection for persistence
        alert_service: Service for sending notifications
        logger: Logger instance for recording events

    Example:
        >>> db = DatabaseConnection()
        >>> alerts = EmailAlertService()
        >>> inventory = InventoryManager(db, alerts)
        >>> inventory.add_stock("SKU123", 100)
        >>> current_level = inventory.get_stock_level("SKU123")
        >>> print(current_level)  # 100
    """

    def __init__(self, database: DatabaseConnection, alert_service: AlertService):
        """Initialize inventory manager with required services."""
        self.database = database
        self.alert_service = alert_service
        self.logger = logging.getLogger(__name__)
```

## Type Hints Best Practices

### Basic Type Annotations
```python
from typing import List, Dict, Set, Tuple, Optional, Union, Callable, Any

def process_data(
    items: List[str],
    config: Dict[str, Union[str, int, bool]],
    callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Process items with configuration and optional callback."""
    pass

# Type aliases for complex types
from typing import NewType, TypedDict

UserID = NewType("UserID", int)
Email = NewType("Email", str)

class UserProfile(TypedDict):
    name: str
    email: Email
    age: int
    preferences: Dict[str, Any]
```

### Advanced Type Hints
```python
from typing import Protocol, Generic, TypeVar
from abc import ABC, abstractmethod

# Protocol for interface definitions
class DataProcessor(Protocol):
    """Protocol for data processing implementations."""
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process input data and return processed data."""
        ...

# Generic classes
T = TypeVar("T")

class Repository(Generic[T], ABC):
    """Generic repository pattern implementation."""

    @abstractmethod
    def save(self, entity: T) -> T:
        """Save an entity to the repository."""
        pass

    @abstractmethod
    def find_by_id(self, entity_id: int) -> Optional[T]:
        """Find an entity by its ID."""
        pass

# Usage
class UserRepository(Repository[User]):
    def save(self, entity: User) -> User:
        # Implementation
        pass

    def find_by_id(self, entity_id: int) -> Optional[User]:
        # Implementation
        pass
```

## Error Handling Patterns

### Exception Hierarchy
```python
class ApplicationError(Exception):
    """Base exception for application-specific errors."""
    pass

class ValidationError(ApplicationError):
    """Raised when data validation fails."""
    pass

class ResourceNotFoundError(ApplicationError):
    """Raised when a requested resource is not found."""
    pass

class PermissionDeniedError(ApplicationError):
    """Raised when user lacks required permissions."""
    pass

class ExternalServiceError(ApplicationError):
    """Raised when external services are unavailable."""
    pass
```

### Exception Handling Best Practices
```python
def process_order(order_data: Dict[str, Any]) -> Order:
    """Process order with comprehensive error handling."""
    try:
        # Validate order data
        if not order_data.get("user_id"):
            raise ValidationError("User ID is required")

        if not order_data.get("items"):
            raise ValidationError("Order must contain at least one item")

        # Create order
        order = create_order_from_data(order_data)

        # Process payment
        try:
            payment_result = process_payment(order)
        except PaymentGatewayError as e:
            # Log payment failure but don't expose details to user
            logger.error(f"Payment failed for order {order.id}: {e}")
            raise ExternalServiceError("Payment processing failed") from e

        # Update inventory
        try:
            update_inventory(order.items)
        except InsufficientStockError as e:
            # Rollback payment
            refund_payment(payment_result.transaction_id)
            raise ValidationError(f"Insufficient stock: {e}")

        # Send confirmation
        send_order_confirmation(order)

        return order

    except ValidationError as e:
        logger.warning(f"Order validation failed: {e}")
        raise
    except ExternalServiceError as e:
        logger.error(f"External service error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing order: {e}")
        raise ApplicationError("Unable to process order") from e
```

## Code Organization

### Module Organization
```python
# module_name.py

"""
Module docstring explaining the purpose and functionality.

This module provides utilities for data processing and validation.
"""

# 1. Standard library imports
import os
import sys
import json
import logging
from typing import List, Dict, Optional

# 2. Third-party imports
import pandas as pd
import requests
from pydantic import BaseModel

# 3. Local imports
from .utils import format_data
from .exceptions import DataProcessingError

# 4. Module-level constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_TIMEOUT = 30
logger = logging.getLogger(__name__)

# 5. Private helper functions (prefixed with _)
def _validate_file_size(file_path: str) -> None:
    """Validate that file size is within limits."""
    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        raise DataProcessingError(f"File size exceeds maximum of {MAX_FILE_SIZE} bytes")

# 6. Public functions and classes
class DataProcessor:
    """Main class for data processing operations."""
    pass

def process_csv_file(file_path: str) -> pd.DataFrame:
    """Process CSV file and return DataFrame."""
    pass

# 7. Module execution (if applicable)
if __name__ == "__main__":
    # Code to run when module is executed directly
    pass
```

### Package Structure
```
my_package/
├── __init__.py           # Package initialization, exports
├── exceptions.py         # Custom exceptions
├── models.py             # Data models
├── services/
│   ├── __init__.py
│   ├── auth_service.py
│   └── data_service.py
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   └── validators.py
└── cli.py                # Command-line interface
```

## Performance Considerations

### List Comprehensions vs Loops
```python
# Good: Simple list comprehensions
squares = [x**2 for x in range(1000)]
filtered_data = [item for item in data if item.is_active()]

# For complex logic, use regular loops
result = []
for item in data:
    if item.is_active() and item.is_valid():
        processed = complex_processing(item)
        if processed.meets_criteria():
            result.append(processed)

# Use generator expressions for large datasets
large_data = (process_item(item) for item in huge_dataset)
total = sum(large_data)  # Processes items one at a time
```

### String Concatenation
```python
# Good: f-strings (Python 3.6+)
message = f"User {user.name} (ID: {user.id}) has {len(user.orders)} orders"

# Good: join() for multiple strings
items = ["item1", "item2", "item3"]
result = ", ".join(items)

# Good: StringIO for many concatenations
from io import StringIO
buffer = StringIO()
for item in large_list:
    buffer.write(f"Process item: {item}\n")
result = buffer.getvalue()
```

### Efficient Data Structures
```python
# Use sets for membership testing (O(1) vs O(n) for lists)
valid_ids = {1, 2, 3, 4, 5}  # Fast: item in valid_ids
user_id in valid_ids

# Use dictionaries for fast lookups
user_map = {user.id: user for user in users}
user = user_map[user_id]  # O(1) lookup

# Use collections.deque for queue operations
from collections import deque
queue = deque()
queue.append(item)        # O(1)
item = queue.popleft()    # O(1)
```

## Additional Resources

### Style Guides and Tools
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [PEP 257 Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 Type Hints](https://peps.python.org/pep-0484/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Ruff Linter and Formatter](https://docs.astral.sh/ruff/)
- [Mypy Type Checker](https://mypy.readthedocs.io/)

### Useful ruff Rules
- `E`, `W`: PEP 8 style violations
- `F`: Pyflakes error detection
- `I`: Import sorting
- `B`: flake8-bugbear (common bugs)
- `C4`: flake8-comprehensions (comprehension improvements)
- `UP`: pyupgrade (Python 2 to 3 upgrades)
- `S`: flake8-bandit (security issues)