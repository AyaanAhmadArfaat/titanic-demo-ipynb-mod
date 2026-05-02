import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging to integrate with system monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Core application configuration managed via environment variables.
    Environment variables can be set in .env or exported in the host system.
    """
    # API Settings
    APP_NAME: str = "Titanic Survival Prediction API"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"

    # MLflow Monitoring
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = "titanic_survival_prediction"

    # AWS Configuration for EC2 Deployment
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")

    # Model Metadata
    MODEL_NAME: str = "titanic_random_forest"
    MODEL_VERSION: str = "1"
    MODEL_PATH: str = "models/model.pkl"

    # Database/Data Persistence
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "sqlite:///./titanic_predictions.db"
    )

    # Environment configuration
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    def validate_settings(self):
        """
        Validate critical configuration paths and connectivity requirements.
        """
        if not self.MLFLOW_TRACKING_URI:
            logger.warning("MLFLOW_TRACKING_URI is not set. Tracking will be disabled.")
        
        logger.info(f"Configuration loaded for: {self.APP_NAME}")

# Initialize settings
settings = Settings()
settings.validate_settings()

# Expose logger for application-wide use
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)