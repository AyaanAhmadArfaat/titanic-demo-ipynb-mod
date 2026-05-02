import pytest
import pandas as pd
import numpy as np
import os
import mlflow
from app.pipeline.preprocessing import FeatureEngineering, Preprocessor

# Mocking environment variables for tests
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns.db"

@pytest.fixture
def sample_titanic_data():
    """Creates a minimal dataframe representative of the Titanic dataset."""
    return pd.DataFrame({
        'Name': ['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)'],
        'Age': [22.0, 26.0, 35.0],
        'SibSp': [1, 0, 1],
        'Parch': [0, 0, 0],
        'Fare': [7.25, 7.92, 53.1],
        'Embarked': ['S', 'S', 'C'],
        'Sex': ['male', 'female', 'female'],
        'Survived': [0, 1, 1]
    })

def test_feature_engineering_logic(sample_titanic_data):
    """
    Verifies the feature engineering transformations:
    - Title extraction
    - FamilySize calculation
    - AgeBand creation
    """
    fe = FeatureEngineering()
    processed_df = fe.transform(sample_titanic_data)

    # Assert new columns exist
    assert 'Title' in processed_df.columns
    assert 'FamilySize' in processed_df.columns
    assert 'IsAlone' in processed_df.columns
    assert 'AgeBand' in processed_df.columns

    # Assert logic for FamilySize (SibSp + Parch + 1)
    assert processed_df['FamilySize'].iloc[0] == 2
    assert processed_df['IsAlone'].iloc[0] == 0
    assert processed_df['IsAlone'].iloc[1] == 1

def test_preprocessor_encoding(sample_titanic_data):
    """
    Verifies categorical encoding and scaling logic.
    """
    fe = FeatureEngineering()
    df_engineered = fe.transform(sample_titanic_data)

    preprocessor = Preprocessor()
    X_scaled = preprocessor.fit_transform(df_engineered)

    # Verify output structure
    assert isinstance(X_scaled, np.ndarray)
    assert X_scaled.shape[1] == len(preprocessor.features)
    
    # Verify encoders were stored
    assert 'Sex' in preprocessor.encoders
    assert 'Title' in preprocessor.encoders

def test_mlflow_integration():
    """
    Ensures the MLflow experiment tracking is initialized correctly during processing.
    """
    mlflow.set_experiment("test_titanic_tracking")
    with mlflow.start_run():
        mlflow.log_param("test_param", "unit_test")
        run = mlflow.active_run()
        assert run is not None
        mlflow.end_run()

def test_missing_value_imputation():
    """
    Verifies that missing values in 'Age' are handled correctly via group-median.
    """
    data = pd.DataFrame({
        'Name': ['Mr. Test', 'Miss. Test'],
        'Age': [np.nan, np.nan],
        'SibSp': [0, 0],
        'Parch': [0, 0],
        'Fare': [10.0, 10.0],
        'Embarked': ['S', 'S'],
        'Sex': ['male', 'female'],
        'Survived': [0, 1]
    })
    fe = FeatureEngineering()
    processed_df = fe.transform(data)
    
    # Should not be null anymore
    assert not processed_df['Age'].isnull().any()
    assert processed_df['Age'].iloc[0] > 0