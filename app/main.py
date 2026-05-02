import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import mlflow

from app.api.routes.predict import router as predict_router
from app.core.config import settings, get_logger

# Configure Logger
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for application startup and shutdown.
    Configures MLflow tracking and verifies connection to the model registry.
    """
    logger.info("Starting up Titanic Survival Prediction API...")
    
    # Configure MLflow tracking
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        logger.info(f"MLflow tracking configured at: {settings.MLFLOW_TRACKING_URI}")
    except Exception as e:
        logger.error(f"Failed to configure MLflow tracking: {str(e)}")

    yield

    # Shutdown logic
    logger.info("Shutting down application...")

# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Production-grade API for predicting Titanic passenger survival using Random Forest models.",
    lifespan=lifespan,
    debug=settings.DEBUG
)

# Include API Routers
app.include_router(predict_router, prefix=settings.API_V1_STR, tags=["Prediction"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to ensure all errors are logged and return standardized JSON responses.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred. Please contact the administrator."}
    )

@app.get("/")
async def root():
    """
    Root endpoint providing basic API status.
    """
    return {
        "message": "Welcome to the Titanic Survival Prediction API",
        "documentation": "/docs",
        "status": "online"
    }

if __name__ == "__main__":
    # This block is for local development testing purposes only.
    # Production deployment uses Gunicorn/Uvicorn workers.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)