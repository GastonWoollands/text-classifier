# Text Sentiment Classifier API

A FastAPI-based sentiment analysis service using DistilBERT transformers model, deployed on Kubernetes with automated CI/CD pipeline.

## Project Overview

InfoJobs technical test for API deployment using transformers model for sentiment analysis in Kubernetes environment.

## Model

- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Task**: Binary sentiment classification (POSITIVE/NEGATIVE)
- **Framework**: Hugging Face Transformers with PyTorch
- **Max Input Length**: 512 tokens

## API Endpoints

### Health Check
```http
GET /health
```
Returns service status to verify the model is loaded and ready.

### Sentiment Analysis
```http
POST /predict_sentiment
```

**Request Body:**
```json
{
  "text": "I love this movie!"
}
```

**Query Parameters:**
- `add_score` (boolean, default: false): Include confidence score in response

**Response (without score):**
```json
{
  "label": "POSITIVE"
}
```

**Response (with score):**
```json
{
  "label": "POSITIVE",
  "score": 0.9876
}
```

## Architecture

### Tech Stack
- **Backend**: FastAPI (Python 3.10+)
- **ML**: Transformers, PyTorch
- **Container**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Package Management**: Poetry

### Kubernetes Deployment
- **Replicas**: 2 (autoscaling 1-10)
- **Resources**: 500Mi-2Gi memory, 400m-1 CPU
- **Health Checks**: Readiness and liveness probes
- **Autoscaling**: HPA based on CPU/Memory (80% threshold)

## CI/CD Pipeline

### GitHub Actions Workflow
- **Trigger**: Push to master, Pull requests
- **Steps**:
  1. Poetry dependency management
  2. Version management with Commitizen
  3. AWS ECR authentication
  4. Docker build and push
  5. Kubernetes deployment

### Required Secrets
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

### Next Steps for Robustness
- Implement Flux for GitOps
- Use Sealed Secrets for secure secret management
- Add monitoring and logging (Prometheus/Grafana)

## Deployment

### Prerequisites
- AWS EKS cluster
- ECR repository
- kubectl configured

### Local Docker Build & Run
```bash
# Install Poetry and export plugin
pip install poetry poetry-plugin-export

# Export requirements.txt
poetry export -f requirements.txt --without-hashes -o requirements.txt

# Get version from Commitizen
export version=$(cz version --project)

# Build Docker image
docker build -t text-classifier:$version .

# Run Docker container
docker run -p 8080:8080 --name text-classifier text-classifier:$version
```

### Testing with curl
```bash
# Health check
curl http://localhost:8080/health

# Sentiment analysis (label only)
curl -X POST http://localhost:8080/predict_sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this movie!"}'

# Sentiment analysis (with score)
curl -X POST "http://localhost:8080/predict_sentiment?add_score=true" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is terrible!"}'

# Stop and remove container
docker stop text-classifier
docker rm text-classifier
```

### Kubernetes Deployment
```bash
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Testing

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app
```

## Next Steps

### Kubernetes Validation
- Deploy to real Kubernetes instance
- Validate resource usage and autoscaling
- Test load balancing and high availability

### Production Readiness
- Implement proper monitoring and alerting
- Add rate limiting and authentication
- Set up proper logging and tracing
- Configure backup and disaster recovery

## Project Structure

### Key Files Description

- **`app/main.py`**: FastAPI API with health check and sentiment analysis endpoints
- **`app/model.py`**: Sentiment analysis model loading and prediction logic
- **`app/schemas.py`**: Pydantic models for request/response validation
- **`app/config.py`**: Configuration constants and settings
- **`k8s/deployment.yaml`**: Kubernetes deployment
- **`k8s/service.yaml`**: Service configuration
- **`.github/workflows/build.yml`**: CI/CD pipeline for automated deployment
- **`Dockerfile`**: Multi-stage Docker build for production image
- **`pyproject.toml`**: Poetry project configuration and dependencies