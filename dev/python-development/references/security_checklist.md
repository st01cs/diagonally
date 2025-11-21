# Python Security Checklist and Best Practices

## Overview

This comprehensive security guide covers common security vulnerabilities in Python applications and provides best practices for writing secure code.

## Input Validation and Sanitization

### 1. Validate All Input

```python
# Good: Use validation libraries
from pydantic import BaseModel, EmailStr, constr, Field
from typing import Optional
import re

class UserInput(BaseModel):
    username: constr(min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_]+$')
    email: EmailStr
    age: Field(ge=0, le=150)  # Greater than or equal to 0, less than or equal to 150
    bio: Optional[constr(max_length=500)] = None

# Bad: No validation
def create_user(username, email, age):
    # Direct use without validation
    return User(username=username, email=email, age=age)
```

### 2. Sanitize Output

```python
import html
import bleach

# Good: Escape HTML output
def render_user_input(user_input):
    escaped_input = html.escape(user_input)
    return f"<div>{escaped_input}</div>"

# For rich content, use bleach to allow safe tags
def render_safe_html(user_html):
    allowed_tags = ['p', 'b', 'i', 'u', 'strong', 'em']
    allowed_attributes = {'*': ['class']}
    clean_html = bleach.clean(user_html, tags=allowed_tags, attributes=allowed_attributes)
    return clean_html
```

### 3. File Upload Security

```python
import os
from pathlib import Path
import magic  # python-magic library

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.pdf'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'application/pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file_upload(file_data, filename):
    """Validate uploaded file for security."""
    # Check file size
    if len(file_data) > MAX_FILE_SIZE:
        raise ValueError("File too large")

    # Check file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type {file_ext} not allowed")

    # Check MIME type
    mime_type = magic.from_buffer(file_data, mime=True)
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"File MIME type {mime_type} not allowed")

    return True

def safe_filename(filename):
    """Generate a safe filename."""
    # Remove path traversal attempts
    filename = os.path.basename(filename)
    # Remove dangerous characters
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    return filename
```

## Authentication and Authorization

### 1. Secure Password Handling

```python
import secrets
import hashlib
from typing import Optional

class PasswordManager:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with salt."""
        salt = secrets.token_bytes(32)
        pwdhash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000,  # Number of iterations
        )
        return salt + pwdhash

    @staticmethod
    def verify_password(stored_password: str, provided_password: str) -> bool:
        """Verify password against stored hash."""
        salt = stored_password[:32]
        stored_hash = stored_password[32:]
        pwdhash = hashlib.pbkdf2_hmac(
            'sha256',
            provided_password.encode('utf-8'),
            salt,
            100000,
        )
        return pwdhash == stored_hash

# Or use a proven library like bcrypt
import bcrypt

def hash_password_bcrypt(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt)

def verify_password_bcrypt(hashed: str, password: str) -> bool:
    """Verify password using bcrypt."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)
```

### 2. Session Management

```python
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def create_session(self, user_id: int) -> str:
        """Create secure session token."""
        session_token = secrets.token_urlsafe(32)
        self.sessions[session_token] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(hours=24),
            'ip_address': None  # Store for additional security
        }
        return session_token

    def validate_session(self, session_token: str, ip_address: str = None) -> Optional[Dict]:
        """Validate session token."""
        if session_token not in self.sessions:
            return None

        session = self.sessions[session_token]

        # Check expiration
        if datetime.utcnow() > session['expires_at']:
            del self.sessions[session_token]
            return None

        # Check IP address (optional additional security)
        if ip_address and session.get('ip_address') and session['ip_address'] != ip_address:
            del self.sessions[session_token]
            return None

        return session

    def revoke_session(self, session_token: str) -> None:
        """Revoke session token."""
        if session_token in self.sessions:
            del self.sessions[session_token]
```

### 3. Authorization

```python
from functools import wraps
from typing import List, Set
from enum import Enum

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

def require_permissions(permissions: List[Permission]):
    """Decorator to require specific permissions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = get_current_user()  # Your user retrieval logic

            if not user:
                raise PermissionError("User not authenticated")

            user_permissions = get_user_permissions(user.id)

            if not all(perm in user_permissions for perm in permissions):
                raise PermissionError(f"Insufficient permissions. Required: {permissions}")

            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@require_permissions([Permission.READ, Permission.WRITE])
def update_user_data(user_id: int, data: dict):
    """Update user data with permission checks."""
    pass
```

## Database Security

### 1. SQL Injection Prevention

```python
import sqlite3
from contextlib import contextmanager

# Bad: Vulnerable to SQL injection
def get_user_unsafe(username):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # NEVER do this!
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()

# Good: Using parameterized queries
def get_user_safe(username: str):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Safe parameterized query
    query = "SELECT * FROM users WHERE username = ?"
    cursor.execute(query, (username,))
    return cursor.fetchone()

# Even better: Using ORM like SQLAlchemy
from sqlalchemy.orm import Session
from sqlalchemy import text

def get_user_orm(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

# For raw SQL with SQLAlchemy
def get_user_raw_sql(db: Session, username: str):
    result = db.execute(
        text("SELECT * FROM users WHERE username = :username"),
        {"username": username}
    )
    return result.fetchone()
```

### 2. Database Connection Security

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Good: Use environment variables for credentials
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'sslmode': 'require'  # Enforce SSL
}

# Use connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

def create_secure_engine():
    database_url = (
        f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
        f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
        f"?sslmode={DATABASE_CONFIG['sslmode']}"
    )

    return create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600
    )
```

## API Security

### 1. API Key Management

```python
import os
import secrets
from typing import Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

class APIKeyManager:
    def __init__(self):
        self.api_keys = {}
        self.load_api_keys()

    def load_api_keys(self):
        """Load API keys from environment or secure storage."""
        # In production, use a proper key management system
        keys = os.getenv('API_KEYS', '').split(',')
        for key in keys:
            if key.strip():
                self.api_keys[key.strip()] = {'active': True}

    def generate_api_key(self) -> str:
        """Generate new API key."""
        return secrets.token_urlsafe(32)

    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key."""
        return api_key in self.api_keys and self.api_keys[api_key]['active']

# FastAPI dependency
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for FastAPI endpoints."""
    api_key = credentials.credentials
    if not api_key_manager.validate_api_key(api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid or inactive API key"
        )
    return api_key
```

### 2. Rate Limiting

```python
import time
from collections import defaultdict
from typing import Dict

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]

        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True

        return False

# FastAPI middleware
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host

    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"}
        )

    response = await call_next(request)
    return response
```

## Cryptography and Secrets Management

### 1. Encryption

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class EncryptionManager:
    def __init__(self, password: str):
        self.password = password.encode()
        self.salt = os.getenv('ENCRYPTION_SALT', b'default_salt_change_in_production')
        self.key = self._derive_key()
        self.fernet = Fernet(self.key)

    def _derive_key(self) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        return key

    def encrypt(self, data: str) -> str:
        """Encrypt data."""
        encrypted_data = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.fernet.decrypt(encrypted_bytes)
        return decrypted_data.decode()
```

### 2. Environment Variable Management

```python
import os
from dotenv import load_dotenv
from typing import Optional

class Config:
    def __init__(self):
        load_dotenv()
        self.validate_required_vars()

    def validate_required_vars(self):
        """Validate that required environment variables are set."""
        required_vars = ['DATABASE_URL', 'SECRET_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    @property
    def database_url(self) -> str:
        return os.getenv('DATABASE_URL')

    @property
    def secret_key(self) -> str:
        return os.getenv('SECRET_KEY')

    @property
    def debug(self) -> bool:
        return os.getenv('DEBUG', 'False').lower() == 'true'

config = Config()
```

## Security Headers and Web Security

### 1. Security Headers

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    return response

# CORS configuration (be restrictive)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains only
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Specific methods only
    allow_headers=["Authorization", "Content-Type"],  # Specific headers only
)
```

### 2. CSRF Protection

```python
import secrets
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class CSRFProtection:
    def __init__(self):
        self.tokens = {}

    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session."""
        token = secrets.token_urlsafe(32)
        self.tokens[session_id] = token
        return token

    def validate_token(self, session_id: str, token: str) -> bool:
        """Validate CSRF token."""
        return self.tokens.get(session_id) == token

csrf_protection = CSRFProtection()

# In your routes
@app.post("/protected-endpoint")
async def protected_endpoint(request: Request, csrf_token: str = Form(...)):
    session_id = get_session_id(request)  # Your session management

    if not csrf_protection.validate_token(session_id, csrf_token):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    # Process request
    pass
```

## Logging and Monitoring

### 1. Security Logging

```python
import logging
from datetime import datetime
from typing import Optional

# Configure security logging
security_logger = logging.getLogger('security')
security_logger.setLevel(logging.INFO)
handler = logging.FileHandler('security.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
security_logger.addHandler(handler)

def log_security_event(event_type: str, user_id: Optional[int] = None,
                      ip_address: Optional[str] = None, details: Optional[str] = None):
    """Log security events."""
    log_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'event_type': event_type,
        'user_id': user_id,
        'ip_address': ip_address,
        'details': details
    }
    security_logger.info(f"Security Event: {log_data}")

# Usage examples
log_security_event("LOGIN_SUCCESS", user_id=123, ip_address="192.168.1.1")
log_security_event("FAILED_LOGIN_ATTEMPT", ip_address="192.168.1.1", details="Invalid password")
log_security_event("PERMISSION_DENIED", user_id=456, ip_address="192.168.1.2", details="Attempted to access admin area")
```

### 2. Anomaly Detection

```python
from collections import defaultdict, deque
import time
from typing import Dict, List

class SecurityMonitor:
    def __init__(self):
        self.failed_attempts = defaultdict(deque)
        self.suspicious_ips = set()
        self.blocked_ips = set()

    def record_failed_login(self, ip_address: str):
        """Record failed login attempt."""
        now = time.time()
        self.failed_attempts[ip_address].append(now)

        # Clean old attempts (older than 1 hour)
        cutoff = now - 3600
        self.failed_attempts[ip_address] = deque(
            [attempt for attempt in self.failed_attempts[ip_address] if attempt > cutoff]
        )

        # Check for suspicious activity
        if len(self.failed_attempts[ip_address]) > 10:
            self.suspicious_ips.add(ip_address)
            log_security_event("SUSPICIOUS_ACTIVITY", ip_address=ip_address,
                             details=f"Multiple failed login attempts: {len(self.failed_attempts[ip_address])}")

        # Block IP after many attempts
        if len(self.failed_attempts[ip_address]) > 50:
            self.blocked_ips.add(ip_address)
            log_security_event("IP_BLOCKED", ip_address=ip_address)

    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked."""
        return ip_address in self.blocked_ips

security_monitor = SecurityMonitor()
```

## Security Checklist

### Development Phase
- [ ] All user inputs are validated and sanitized
- [ ] SQL queries use parameterized statements
- [ ] Passwords are properly hashed (bcrypt, Argon2)
- [ ] Sensitive data is encrypted at rest and in transit
- [ ] Environment variables used for configuration
- [ ] API rate limiting implemented
- [ ] Proper authentication and authorization
- [ ] Security headers configured
- [ ] CSRF protection for state-changing operations
- [ ] Error messages don't leak sensitive information

### Deployment Phase
- [ ] HTTPS enforced everywhere
- [ ] Regular security updates applied
- [ ] Database access restricted
- [ ] File upload restrictions in place
- [ ] Logging and monitoring configured
- [ ] Backup and recovery procedures tested
- [ ] Dependency vulnerability scanning
- [ ] Security testing performed
- [ ] Secrets management system implemented

### Operational Phase
- [ ] Regular security audits
- [ ] Monitor for suspicious activity
- [ ] Incident response plan in place
- [ ] User access regularly reviewed
- [ ] Security patches applied promptly
- [ ] Backup integrity verified
- [ ] Log analysis and retention policies
- [ ] Employee security training

### Common Vulnerabilities to Check

#### OWASP Top 10 for Python
1. **Injection (SQL, Command, LDAP)**
   - Use parameterized queries
   - Avoid eval() and exec()
   - Validate all input

2. **Broken Authentication**
   - Implement proper session management
   - Use secure password policies
   - Multi-factor authentication

3. **Sensitive Data Exposure**
   - Encrypt sensitive data
   - Use HTTPS everywhere
   - Secure storage of secrets

4. **XML External Entities (XXE)**
   - Disable XML external entities
   - Use safer XML parsers

5. **Broken Access Control**
   - Implement proper authorization
   - Principle of least privilege
   - Regular access reviews

6. **Security Misconfiguration**
   - Secure default configurations
   - Remove unnecessary features
   - Keep software updated

7. **Cross-Site Scripting (XSS)**
   - Input validation and output encoding
   - Content Security Policy
   - Safe HTML frameworks

8. **Insecure Deserialization**
   - Avoid unsafe deserialization
   - Validate serialized data

9. **Using Components with Known Vulnerabilities**
   - Regular dependency updates
   - Vulnerability scanning

10. **Insufficient Logging & Monitoring**
    - Comprehensive logging
    - Real-time monitoring
    - Incident response