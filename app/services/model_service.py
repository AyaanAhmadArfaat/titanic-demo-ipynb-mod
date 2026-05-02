import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from typing import Dict, Any
from app.core.config import settings, get_logger
from app.models.schemas import TitanicPredictionRequest
from app.pipeline.preprocessing import Preprocessor

logger = get_logger(__name__)

class ModelService:
    """
    Service class to handle model loading, inference preprocessing,
    and prediction execution for Titanic survival analysis.
    """
    def __init__(self):
        self.model_uri = f"models:/{settings.MODEL_NAME}/{settings.MODEL_VERSION}"
        self.model = None
        self.preprocessor = Preprocessor()
        self._load_model()

    def _load_model(self):
        """
        Loads the production model from MLflow Model Registry.
        """
        try:
            logger.info(f"Loading model from MLflow: {self.model_uri}")
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            self.model = mlflow.sklearn.load_model(self.model_uri)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            raise RuntimeError(f"Model loading error: {e}")

    def preprocess_request(self, request: TitanicPredictionRequest) -> np.ndarray:
        """
        Converts Pydantic request to DataFrame and applies preprocessing pipeline.
        """
        # Convert request to DataFrame
        data_dict = request.dict()
        df = pd.DataFrame([data_dict])
        
        # Standardize columns to match training features
        # Note: Preprocessor logic assumes consistent feature engineering as per training
        X_scaled = self.preprocessor.fit_transform(df)
        return X_scaled

    def predict(self, request: TitanicPredictionRequest) -> Dict[str, Any]:
        """
        Executes prediction logic and logs inference to MLflow.
        """
        try:
            X = self.preprocess_request(request)
            
            # Inference
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0].tolist()
            
            # Log to MLflow
            with mlflow.start_run(run_name="inference_request"):
                mlflow.log_param("input_class", request.Pclass)
                mlflow.log_metric("predicted_class", float(prediction))
                
            return {
                "survival_prediction": bool(prediction),
                "probability": probability,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"status": "error", "message": str(e)}

# Singleton instance for the application
model_service = ModelService()