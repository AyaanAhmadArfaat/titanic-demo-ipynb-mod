from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum

class PassengerClass(int, Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3

class SexEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"

class EmbarkedEnum(str, Enum):
    C = "C"
    Q = "Q"
    S = "S"

class TitanicPredictionRequest(BaseModel):
    """
    Pydantic schema for Titanic survival prediction input.
    Features align with the preprocessing pipeline defined in the training notebook.
    """
    pclass: PassengerClass = Field(..., description="Ticket class (1=1st, 2=2nd, 3=3rd)")
    sex: SexEnum = Field(..., description="Gender of the passenger")
    age: float = Field(..., gt=0, lt=100, description="Age of the passenger")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare")
    embarked: EmbarkedEnum = Field(..., description="Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)")
    title: str = Field(..., description="Title derived from name (e.g., Mr, Mrs, Miss, Rare)")
    family_size: int = Field(..., ge=1, description="Total number of family members aboard")
    is_alone: int = Field(..., ge=0, le=1, description="1 if passenger is alone, 0 otherwise")
    fare_per_person: float = Field(..., ge=0, description="Fare divided by family size")
    age_band: int = Field(..., ge=0, le=4, description="Encoded age category")

    class Config:
        schema_extra = {
            "example": {
                "pclass": 3,
                "sex": "male",
                "age": 22.0,
                "sibsp": 1,
                "parch": 0,
                "fare": 7.25,
                "embarked": "S",
                "title": "Mr",
                "family_size": 2,
                "is_alone": 0,
                "fare_per_person": 3.625,
                "age_band": 2
            }
        }

class TitanicPredictionResponse(BaseModel):
    """
    Pydantic schema for model prediction output.
    """
    survival_probability: float = Field(..., ge=0, le=1)
    prediction: int = Field(..., description="0: Did Not Survive, 1: Survived")
    model_version: str = Field(..., description="Version of the ML model used for inference")

class MLflowTrackingResponse(BaseModel):
    """
    Schema for reporting successful MLflow tracking operations.
    """
    status: str
    run_id: str
    experiment_id: str
