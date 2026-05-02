from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import TitanicPredictionRequest, TitanicPredictionResponse
from app.services.model_service import model_service
from app.core.config import get_logger

# Initialize logger and router
logger = get_logger(__name__)
router = APIRouter()

@router.post(
    "/predict",
    response_model=TitanicPredictionResponse,
    summary="Predict Titanic passenger survival",
    description="Accepts passenger features and returns a survival prediction with probability scores."
)
async def predict_survival(request: TitanicPredictionRequest):
    """
    API endpoint to perform survival predictions using the loaded MLflow model.
    
    Args:
        request (TitanicPredictionRequest): Passenger data features.
    
    Returns:
        TitanicPredictionResponse: Prediction label and probability.
    
    Raises:
        HTTPException: If model inference fails.
    """
    try:
        logger.info(f"Received prediction request for features: {request.dict()}")
        
        # Perform prediction via the injected model_service
        result = model_service.predict(request)
        
        if result.get("status") == "error":
            logger.error(f"Inference error: {result.get('message')}")
            raise HTTPException(status_code=500, detail=result.get("message"))

        # Map result to response schema
        # probability is [prob_class_0, prob_class_1], survival is index 1
        probs = result.get("probability", [0.0, 0.0])
        survival_prob = probs[1] if len(probs) > 1 else 0.0
        
        response = TitanicPredictionResponse(
            survival_probability=float(survival_prob),
            prediction=int(result.get("survival_prediction", 0)),
            model_version="v1.0.0" # Matches model registry convention
        )
        
        logger.info(f"Prediction completed successfully: {response.prediction}")
        return response

    except Exception as e:
        logger.exception("An unexpected error occurred during prediction")
        raise HTTPException(status_code=500, detail="Internal server error during model inference")

@router.get("/health")
async def health_check():
    """
    Basic health check for the prediction service.
    """
    return {"status": "online", "model_ready": model_service.model is not None}