import pytest
from unittest.mock import patch

def test_health_check(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch('app.main.predict_sentiment')
def test_predict_success_label_only(mock_predict, client, mock_positive):
    """Test prediction without score"""
    mock_predict.return_value = mock_positive
    response = client.post("/predict_sentiment", json={"text": "I love this!"})
    assert response.status_code == 200
    assert response.json() == {"label": "POSITIVE"}

@patch('app.main.predict_sentiment')
def test_predict_success_with_score(mock_predict, client, mock_negative):
    """Test prediction with score"""
    mock_predict.return_value = mock_negative
    response = client.post("/predict_sentiment?add_score=true", json={"text": "I hate this!"})
    assert response.status_code == 200
    assert response.json() == {"label": "NEGATIVE", "score": 0.87}

def test_predict_empty_text(client):
    """Test empty text validation"""
    response = client.post("/predict_sentiment", json={"text": ""})
    assert response.status_code == 400

def test_predict_missing_text(client):
    """Test missing text validation"""
    response = client.post("/predict_sentiment", json={})
    assert response.status_code == 422

@patch('app.main.predict_sentiment')
def test_predict_model_error(mock_predict, client):
    """Test model error handling"""
    mock_predict.side_effect = Exception("Model failed")
    response = client.post("/predict_sentiment", json={"text": "test"})
    assert response.status_code == 500

def test_real_model_prediction(client, run_model_tests):
    if not run_model_tests:
        pytest.skip("Skipping real model test because --run-model-tests not set")

    response = client.post("/predict_sentiment?add_score=true", json={"text": "This is great!"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data and "score" in data
    assert data["label"] in ["POSITIVE", "NEGATIVE"]
