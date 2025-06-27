from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import logging
from typing import Optional
from app.schemas import TextOutput
from app.config import MODEL_NAME, MAX_LENGTH, TRUNCATION, LOG_LEVEL

# Initialize logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def predict_sentiment(
    text: str, 
    model: Optional[AutoModelForSequenceClassification] = None, 
    tokenizer: Optional[AutoTokenizer] = None
) -> TextOutput:
    """
    Classify the sentiment of the text as POSITIVE or NEGATIVE.

    Args: 
        text (str): Text input to classify. Must not be empty.
        model (AutoModelForSequenceClassification, optional): Model to use. Defaults to pre-loaded model.
        tokenizer (AutoTokenizer, optional): Tokenizer to use. Defaults to pre-loaded tokenizer.

    Returns: 
        TextOutput: Object containing label and confidence score
    """
    # Input validation
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty or None")
    
    # Use default model and tokenizer if not provided
    if model is None:
        model = globals()['model']
    if tokenizer is None:
        tokenizer = globals()['tokenizer']
    
    logger.info(f"Predicting sentiment for text: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=TRUNCATION, 
            max_length=MAX_LENGTH
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            score = probs[0][pred_id].item()
            label = model.config.id2label[pred_id].upper()

            logger.info(f"Predicted label: {label}")
            logger.info(f"Predicted score: {score:.4f}")
        
        return TextOutput(label=label, score=round(score, 4))
        
    except Exception as e:
        logger.error(f"Error during sentiment prediction: {str(e)}")
        raise Exception(f"Failed to predict sentiment: {str(e)}")
