# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Union

# --- 1. Load the Saved Model and Scaler ---
# Make sure 'best_iris_model_and_scaler.joblib' is in the same directory as this script
try:
    loaded_artifacts = joblib.load('best_iris_model_and_scaler.joblib')
    model = loaded_artifacts['model']
    scaler = loaded_artifacts['scaler'] # This might be None if RF was best model
    target_names = loaded_artifacts['target_names']
    model_type = loaded_artifacts['model_type']
    print(f"Model ({model_type}) and scaler loaded successfully.")
    print(f"Target species: {target_names}")
except FileNotFoundError:
    print("Error: 'best_iris_model_and_scaler.joblib' not found.")
    print("Please ensure the model file is in the same directory as 'main.py'.")
    exit(1) # Exit if model not found
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    exit(1) # Exit on other loading errors

# --- 2. Initialize FastAPI App ---
app = FastAPI(
    title="Iris Species Prediction API",
    description="A simple REST API to predict Iris flower species using a trained ML model.",
    version="1.0.0"
)

# --- 3. Define Input Data Model using Pydantic ---
class IrisFeatures(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

# --- 4. Define Output Data Model (Optional but good practice) ---
class PredictionResponse(BaseModel):
    predicted_species: str
    predicted_label: int
    probabilities: List[float]

# --- 5. Create Endpoints ---

@app.get("/health", summary="Check API Health Status")
async def health():
    """
    Returns the health status of the API.
    """
    return {"status": "ok", "message": "API is healthy and ready to predict!"}

@app.post("/predict", response_model=PredictionResponse, summary="Predict Iris Species")
async def predict(features: IrisFeatures):
    """
    Accepts Iris flower measurements as JSON input and returns the predicted species,
    its numerical label, and prediction probabilities.

    **Input:**
    - `sepal_length_cm`: Sepal length in centimeters (float)
    - `sepal_width_cm`: Sepal width in centimeters (float)
    - `petal_length_cm`: Petal length in centimeters (float)
    - `petal_width_cm`: Petal width in centimeters (float)

    **Output:**
    - `predicted_species`: The predicted name of the Iris species (e.g., "setosa", "versicolor", "virginica")
    - `predicted_label`: The numerical label of the predicted species (0, 1, or 2)
    - `probabilities`: A list of probabilities for each species
    """
    try:
        # Convert input Pydantic model to a NumPy array
        input_data = np.array([
            features.sepal_length_cm,
            features.sepal_width_cm,
            features.petal_length_cm,
            features.petal_width_cm
        ]).reshape(1, -1) # Reshape to 2D array for single prediction

        # Apply scaler if the best model is Logistic Regression
        if model_type == 'logistic_regression' and scaler is not None:
            processed_input_data = scaler.transform(input_data)
        else:
            # For Random Forest, or if scaler is None, use original input
            processed_input_data = input_data

        # Make prediction
        prediction_label = model.predict(processed_input_data)[0]
        prediction_proba = model.predict_proba(processed_input_data)[0].tolist()

        # Get the species name from the label
        predicted_species_name = target_names[prediction_label]

        return PredictionResponse(
            predicted_species=predicted_species_name,
            predicted_label=int(prediction_label),
            probabilities=prediction_proba
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# To run this app locally, save this code as 'main.py'
# and then run 'uvicorn main:app --reload' in your terminal