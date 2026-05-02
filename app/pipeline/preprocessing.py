import pandas as pd
import numpy as np
import mlflow
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.core.logging import get_logger

logger = get_logger(__name__)

# Configure MLflow tracking
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("titanic_preprocessing")

class FeatureEngineering:
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering transformations."""
        df_clean = df.copy()
        
        # Extract Title
        df_clean['Title'] = df_clean['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df_clean['Title'] = df_clean['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
        )
        df_clean['Title'] = df_clean['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        
        # Missing value imputation
        df_clean['Age'] = df_clean.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
        df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
        df_clean['Fare'] = df_clean['Fare'].fillna(df_clean['Fare'].median())
        
        # Create features
        df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
        df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)
        df_clean['FarePerPerson'] = df_clean['Fare'] / df_clean['FamilySize']
        df_clean['AgeBand'] = pd.cut(df_clean['Age'], bins=[0, 12, 18, 35, 60, 100],
                                      labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
        
        return df_clean

class Preprocessor:
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                         'Embarked', 'Title', 'FamilySize', 'IsAlone', 'FarePerPerson', 'AgeBand']

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Encodes categories and scales numerical features."""
        df_work = df.copy()
        
        # Label Encoding
        cat_cols = ['Sex', 'Embarked', 'Title', 'AgeBand']
        for col in cat_cols:
            le = LabelEncoder()
            df_work[col] = le.fit_transform(df_work[col].astype(str))
            self.encoders[col] = le
            
        X = df_work[self.features]
        return self.scaler.fit_transform(X)

def run_preprocessing_pipeline(df: pd.DataFrame) -> tuple:
    """
    Executes the full preprocessing pipeline including Feature Engineering and Encoding.
    """
    with mlflow.start_run(run_name="preprocessing_step"):
        logger.info("Starting feature engineering...")
        feature_eng = FeatureEngineering()
        df_engineered = feature_eng.transform(df)
        
        logger.info("Starting categorical encoding and scaling...")
        preprocessor = Preprocessor()
        X_scaled = preprocessor.fit_transform(df_engineered)
        y = df_engineered['Survived'].values
        
        mlflow.log_param("num_features", len(preprocessor.features))
        mlflow.log_metric("dataset_rows", len(df))
        
        logger.info("Preprocessing complete.")
        return X_scaled, y

if __name__ == "__main__":
    # This block allows module testing
    from app.pipeline.data_ingestion import fetch_titanic_data
    data = fetch_titanic_data()
    X, y = run_preprocessing_pipeline(data)
    print(f"Pipeline completed. Feature matrix shape: {X.shape}")