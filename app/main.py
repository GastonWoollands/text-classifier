from fastapi import FastAPI, Query, HTTPException
from app.model import predict_sentiment
from app.schemas import (
    TextInput, 
    LabelOnlyResponse,
    HealthResponse, 
    SentimentResponse
)
from app.config import API_TITLE, API_DESCRIPTION, API_VERSION, HEALTH_STATUS

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint to verify the service is running and the model is loaded.
    """
    return HealthResponse(
        status=HEALTH_STATUS
    )

@app.post("/predict_sentiment", response_model=SentimentResponse)
def predict(
    input: TextInput, 
    add_score: bool = Query(default=False, description="Include prediction score in response")
) -> SentimentResponse:
    """
    Predict sentiment of input text.
    
    Args:
        input: TextInput containing the text to analyze
        add_score: If True, returns both label and score. If False, returns only label.
    """
    try:
        result = predict_sentiment(input.text)
        
        if add_score:
            return result
        else:
            return LabelOnlyResponse(label=result.label)
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")