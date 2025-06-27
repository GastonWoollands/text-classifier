import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import TextOutput
from unittest.mock import MagicMock

@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)

@pytest.fixture
def positive_text():
    """Sample positive text for testing"""
    return "I love this amazing movie! It's fantastic!"

@pytest.fixture
def negative_text():
    """Sample negative text for testing"""
    return "I hate this terrible movie! It's awful!"

@pytest.fixture
def mock_positive():
    """Mock positive prediction response"""
    return TextOutput(label="POSITIVE", score=0.95)

@pytest.fixture
def mock_negative():
    """Mock negative prediction response"""
    return TextOutput(label="NEGATIVE", score=0.87)

@pytest.fixture
def mock_model_response():
    """Mock model response for testing"""
    return {
        "logits": MagicMock(),
        "config": MagicMock(id2label={0: "NEGATIVE", 1: "POSITIVE"})
    }

# Pytest configuration
def pytest_addoption(parser):
    """Add custom pytest options"""
    parser.addoption(
        "--run-model-tests",
        action="store_true",
        default=False,
        help="Run tests that require full model loading"
    ) 

@pytest.fixture
def run_model_tests(request):
    """Fixture to check if --run-model-tests flag was passed."""
    return request.config.getoption("--run-model-tests")