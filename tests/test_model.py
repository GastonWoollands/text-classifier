import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
import torch
import torch.nn.functional as F
from app.model import predict_sentiment, MODEL_NAME
from app.schemas import TextOutput

class TestModelPredictions:
    """Test model prediction functionality"""
    
    @patch('app.model.AutoTokenizer.from_pretrained')
    @patch('app.model.AutoModelForSequenceClassification.from_pretrained')
    def test_predict_sentiment_positive(self, mock_model, mock_tokenizer):
        """Test positive sentiment prediction"""
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_model_instance.return_value.logits = torch.tensor([[0.1, 0.9]])
        mock_model.return_value = mock_model_instance
        
        # Mock softmax
        with patch('app.model.F.softmax') as mock_softmax:
            mock_softmax.return_value = torch.tensor([[0.1, 0.9]])
            
            result = predict_sentiment("I love this!", mock_model_instance, mock_tokenizer_instance)
            
            assert isinstance(result, TextOutput)
            assert result.label == "POSITIVE"
            assert result.score == 0.9
    
    def test_predict_sentiment_empty_text(self):
        """Test empty text validation"""
        with pytest.raises(ValueError, match="empty"):
            predict_sentiment("")
    
    def test_predict_sentiment_none_text(self):
        """Test None text validation"""
        with pytest.raises(ValueError, match="empty"):
            predict_sentiment(None)
    
    def test_predict_sentiment_whitespace_text(self):
        """Test prediction with whitespace-only text raises ValueError"""
        with pytest.raises(ValueError, match="empty"):
            predict_sentiment("   ")
    
    def test_predict_sentiment_model_error(self):
        """Test model error handling"""
        # Simula que el tokenizer falla cuando se lo llama
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = Exception("Tokenizer failed")

        mock_model_instance = MagicMock()
        
        with pytest.raises(Exception, match="Tokenizer failed"):
            predict_sentiment("Test text", model=mock_model_instance, tokenizer=mock_tokenizer_instance)

class TestModelConfiguration:
    """Test model configuration and constants"""
    
    def test_model_name_constant(self):
        """Test MODEL_NAME constant is set correctly"""
        assert MODEL_NAME == "distilbert-base-uncased-finetuned-sst-2-english"
    
    def test_model_loading(self, request):
        """Test that model can be loaded (integration test)"""
        if not request.config.getoption("--run-model-tests"):
            pytest.skip("Model tests require --run-model-tests flag")

        from app.model import model, tokenizer

        assert model is not None
        assert tokenizer is not None
        assert hasattr(model, 'config')
        assert hasattr(tokenizer, 'encode')

@patch('app.model.AutoTokenizer.from_pretrained')
@patch('app.model.AutoModelForSequenceClassification.from_pretrained')
def test_predict_sentiment_success(mock_model, mock_tokenizer):
    """Test successful prediction"""
    # Mock setup
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
    mock_tokenizer.return_value = mock_tokenizer_instance
    
    mock_model_instance = MagicMock()
    mock_model_instance.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    mock_model_instance.return_value.logits = torch.tensor([[0.1, 0.9]])
    mock_model.return_value = mock_model_instance
    
    with patch('app.model.F.softmax') as mock_softmax:
        mock_softmax.return_value = torch.tensor([[0.1, 0.9]])
        result = predict_sentiment("I love this!", mock_model_instance, mock_tokenizer_instance)
        assert result.label == "POSITIVE"
        assert result.score == 0.9