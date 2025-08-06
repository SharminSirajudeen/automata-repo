"""
Pytest configuration and fixtures for the Automata Learning Platform tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.main import app
from app.database import get_db, Base, User
from app.config import settings
from app.auth import create_access_token, hash_password
from app.security import API_KEYS, hash_api_key

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override the database dependency
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def setup_database():
    """Set up test database."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(setup_database):
    """Create a database session for testing."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    """Create a test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def test_user_data():
    """Test user data for registration/login."""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }


@pytest.fixture
def test_user(db_session, test_user_data):
    """Create a test user in the database."""
    user = User(
        email=test_user_data["email"],
        full_name=test_user_data["full_name"],
        hashed_password=hash_password(test_user_data["password"])
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers for testing."""
    access_token = create_access_token(
        data={"sub": test_user.email, "user_id": str(test_user.id)}
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def api_key_headers():
    """Create API key headers for testing."""
    # Use a test API key
    test_api_key = "test_api_key_123"
    API_KEYS["test_key"] = {
        "key_hash": hash_api_key(test_api_key),
        "scopes": ["ai:read", "ai:write"],
        "rate_limit": "1000/minute",
        "created_at": "2025-08-05T16:27:32Z",
        "last_used": None,
        "usage_count": 0
    }
    return {"Authorization": f"Bearer {test_api_key}"}


@pytest.fixture
def sample_dfa():
    """Sample DFA for testing."""
    return {
        "states": [
            {"id": "q0", "x": 100, "y": 100, "is_start": True, "is_accept": False},
            {"id": "q1", "x": 200, "y": 100, "is_start": False, "is_accept": True}
        ],
        "transitions": [
            {"from_state": "q0", "to_state": "q1", "symbol": "a"},
            {"from_state": "q1", "to_state": "q1", "symbol": "a"},
            {"from_state": "q0", "to_state": "q0", "symbol": "b"},
            {"from_state": "q1", "to_state": "q0", "symbol": "b"}
        ],
        "alphabet": ["a", "b"]
    }


@pytest.fixture
def sample_nfa():
    """Sample NFA for testing."""
    return {
        "states": [
            {"id": "q0", "x": 100, "y": 100, "is_start": True, "is_accept": False},
            {"id": "q1", "x": 200, "y": 100, "is_start": False, "is_accept": False},
            {"id": "q2", "x": 300, "y": 100, "is_start": False, "is_accept": True}
        ],
        "transitions": [
            {"from_state": "q0", "to_state": "q1", "symbol": "a"},
            {"from_state": "q1", "to_state": "q2", "symbol": "b"},
            {"from_state": "q0", "to_state": "q2", "symbol": "ε"}
        ],
        "alphabet": ["a", "b"]
    }


@pytest.fixture
def sample_grammar():
    """Sample context-free grammar for testing."""
    return {
        "start_symbol": "S",
        "terminals": ["a", "b"],
        "non_terminals": ["S", "A", "B"],
        "productions": [
            {"lhs": "S", "rhs": "AB"},
            {"lhs": "A", "rhs": "a"},
            {"lhs": "B", "rhs": "b"}
        ]
    }


@pytest.fixture
def sample_problem():
    """Sample problem for testing."""
    return {
        "id": "test_dfa_problem",
        "type": "dfa",
        "title": "Test DFA Problem",
        "description": "Construct a DFA that accepts strings ending with 'a'",
        "language_description": "All strings ending with 'a'",
        "alphabet": ["a", "b"],
        "difficulty": "beginner",
        "category": "Basic Patterns",
        "test_strings": [
            {"string": "a", "should_accept": True},
            {"string": "aa", "should_accept": True},
            {"string": "ba", "should_accept": True},
            {"string": "b", "should_accept": False},
            {"string": "ab", "should_accept": False}
        ],
        "hints": [
            "Think about the last character",
            "You need to track whether the last character is 'a'"
        ]
    }


@pytest.fixture
def test_performance_data():
    """Test performance data for adaptive learning."""
    return {
        "student_id": "test_student_123",
        "problem_id": "test_problem_456",
        "problem_type": "dfa",
        "score": 0.85,
        "time_spent": 120,  # seconds
        "hints_used": 2,
        "attempts": 3,
        "difficulty_level": "intermediate"
    }


@pytest.fixture
def mock_ai_response():
    """Mock AI response for testing."""
    return {
        "response": "This is a test AI response",
        "confidence": 0.95,
        "model_used": "test_model",
        "processing_time": 1.23
    }


@pytest.fixture
def mock_jflap_conversion():
    """Mock JFLAP conversion result for testing."""
    return {
        "original": "test_nfa",
        "converted": "test_dfa",
        "algorithm": "subset_construction",
        "steps": [
            {"step": 1, "description": "Initialize with start state"},
            {"step": 2, "description": "Process symbol 'a'"},
            {"step": 3, "description": "Process symbol 'b'"}
        ]
    }


@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Reset rate limits before each test."""
    # This would reset rate limit counters
    # Implementation depends on your rate limiting solution
    yield
    # Cleanup after test


@pytest.fixture
def disable_rate_limiting(monkeypatch):
    """Disable rate limiting for tests that need it."""
    monkeypatch.setattr("app.security.limiter", lambda *args, **kwargs: lambda f: f)


class MockOllamaClient:
    """Mock Ollama client for testing AI functionality."""
    
    async def generate(self, model: str, prompt: str, **kwargs):
        """Mock generate method."""
        return {
            "response": f"Mock response for prompt: {prompt[:50]}...",
            "model": model,
            "done": True
        }
    
    async def list(self):
        """Mock list models method."""
        return {
            "models": [
                {"name": "test_model:latest"},
                {"name": "codellama:7b"}
            ]
        }


@pytest.fixture
def mock_ollama_client(monkeypatch):
    """Mock Ollama client for AI tests."""
    client = MockOllamaClient()
    monkeypatch.setattr("app.agents.ollama_client", client)
    return client


# Test data factories
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_user(email: str = "test@example.com", **kwargs):
        """Create test user data."""
        return {
            "email": email,
            "password": "testpassword123",
            "full_name": "Test User",
            **kwargs
        }
    
    @staticmethod
    def create_problem(problem_type: str = "dfa", **kwargs):
        """Create test problem data."""
        return {
            "id": f"test_{problem_type}_problem",
            "type": problem_type,
            "title": f"Test {problem_type.upper()} Problem",
            "description": f"Test {problem_type} problem description",
            "alphabet": ["a", "b"],
            "difficulty": "beginner",
            "category": "Test Category",
            "test_strings": [
                {"string": "a", "should_accept": True},
                {"string": "b", "should_accept": False}
            ],
            "hints": ["Test hint 1", "Test hint 2"],
            **kwargs
        }
    
    @staticmethod
    def create_automaton(automaton_type: str = "dfa", **kwargs):
        """Create test automaton data."""
        base_automaton = {
            "states": [
                {"id": "q0", "x": 100, "y": 100, "is_start": True, "is_accept": False},
                {"id": "q1", "x": 200, "y": 100, "is_start": False, "is_accept": True}
            ],
            "transitions": [
                {"from_state": "q0", "to_state": "q1", "symbol": "a"},
                {"from_state": "q1", "to_state": "q1", "symbol": "a"}
            ],
            "alphabet": ["a", "b"]
        }
        base_automaton.update(kwargs)
        return base_automaton


@pytest.fixture
def test_factory():
    """Provide test data factory."""
    return TestDataFactory