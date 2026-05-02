import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from app.core.config import settings, get_logger
from typing import Dict, Any

logger = get_logger(__name__)

# Setup MLflow
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Trains multiple models and logs performance metrics and model artifacts to MLflow.
        """
        results = {}
        
        for name, model in self.models.items():
            with mlflow.start_run(run_name=f"train_{name.replace(' ', '_')}"):
                logger.info(f"Training {name}...")
                
                # Enable autologging for sklearn models
                mlflow.sklearn.autolog()
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Final Fit
                model.fit(X_train, y_train)
                
                # Evaluation
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                test_acc = accuracy_score(y_test, y_pred)
                test_auc = roc_auc_score(y_test, y_proba)
                
                # Log metrics
                mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("test_roc_auc", test_auc)
                
                results[name] = {
                    'model': model,
                    'test_acc': test_acc,
                    'auc': test_auc
                }
                
                logger.info(f"{name} - Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
        
        return results

def run_training_pipeline(X_scaled: np.ndarray, y: np.ndarray):
    """
    Wrapper function to execute training pipeline given preprocessed features.
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("Splitting data for model training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    best_model_name = max(results, key=lambda x: results[x]['test_acc'])
    logger.info(f"Best performing model: {best_model_name}")
    
    return results[best_model_name]['model']

if __name__ == "__main__":
    # Example integration test
    from app.pipeline.data_ingestion import fetch_titanic_data
    from app.pipeline.preprocessing import run_preprocessing_pipeline
    
    data = fetch_titanic_data()
    X, y = run_preprocessing_pipeline(data)
    best_model = run_training_pipeline(X, y)
    print("Training pipeline executed successfully.")