import pandas as pd
import numpy as np
import mlflow
import os
from typing import Tuple
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

# Configure MLflow tracking for ingestion monitoring
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("titanic_data_ingestion")

def fetch_titanic_data(url: str = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv') -> pd.DataFrame:
    """
    Downloads and loads the Titanic dataset from the specified source.
    Logs ingestion metadata to MLflow.
    """
    try:
        logger.info(f"Fetching dataset from {url}")
        df = pd.read_csv(url)
        
        # Track ingestion metadata
        mlflow.log_param("dataset_url", url)
        mlflow.log_metric("num_rows", df.shape[0])
        mlflow.log_metric("num_columns", df.shape[1])
        
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise RuntimeError(f"Could not load data: {e}")

def analyze_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic data quality checks and returns a summary of missing values.
    """
    logger.info("=== Dataset Info ===")
    # Captured as strings in logs as file output is not desired
    logger.info(df.describe().to_dict())

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).query('`Missing Count` > 0').sort_values('Missing %', ascending=False)

    logger.info(f"Missing values identified: {missing_df.to_dict()}")
    return missing_df

def run_ingestion_pipeline() -> pd.DataFrame:
    """
    Executes the ingestion and basic validation pipeline.
    """
    with mlflow.start_run(run_name="ingestion_step"):
        df = fetch_titanic_data()
        missing_stats = analyze_data_quality(df)
        
        # Log some specific missing value stats to MLflow
        for col, row in missing_stats.iterrows():
            mlflow.log_metric(f"missing_pct_{col}", row['Missing %'])
            
        return df

if __name__ == "__main__":
    # Example of running the module directly
    run_ingestion_pipeline()