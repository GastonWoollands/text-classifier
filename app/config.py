"""
Configuration settings for the text classifier application.
"""

# Model Configuration
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# API Configuration
API_TITLE = "Text Sentiment Classifier API"
API_DESCRIPTION = "A FastAPI service for sentiment analysis using DistilBERT"
API_VERSION = "1.0.0"

# Model Parameters
MAX_LENGTH = 512
TRUNCATION = True

# Logging Configuration
LOG_LEVEL = "INFO"

# Health Check Configuration
HEALTH_STATUS = "healthy" 