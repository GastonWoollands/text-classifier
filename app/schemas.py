from pydantic import BaseModel, Field
from typing import Union

class TextInput(BaseModel):
    """Input schema for text sentiment analysis"""
    text: str = Field(..., description="Text to analyze for sentiment", example="I love this movie!")

class LabelOnlyResponse(BaseModel):
    """Response schema for sentiment prediction without score"""
    label: str = Field(..., description="Predicted sentiment label", example="POSITIVE")

class TextOutput(BaseModel):
    """Response schema for sentiment prediction with score"""
    label: str = Field(..., description="Predicted sentiment label (POSITIVE or NEGATIVE)")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

class HealthResponse(BaseModel):
    """Response schema for health check endpoint"""
    status: str = Field(..., description="Service status", example="healthy")

# Union type for flexible response
SentimentResponse = Union[LabelOnlyResponse, TextOutput] 