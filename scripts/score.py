# scripts/score.py
import json
import os
import joblib
import pandas as pd
from azureml.core.model import Model

def init():
    global model_a, model_b
    # Load Model A
    model_a_path = Model.get_model_path("model_a")
    model_a = joblib.load(model_a_path)
    print(f"Loaded Model A from {model_a_path}")

    # Load Model B
    model_b_path = Model.get_model_path("model_b")
    model_b = joblib.load(model_b_path)
    print(f"Loaded Model B from {model_b_path}")

def run(data):
    try:
        # Parse input JSON data
        input_data = json.loads(data)
        df = pd.DataFrame(input_data)

        # Step 1: Model A Inference
        df['prediction_a'] = model_a.predict(df)

        # Step 2: Model B Inference (Uses Model A's output)
        df['final_prediction'] = model_b.predict(df[['prediction_a']])

        # Return results
        return df[['final_prediction']].to_dict(orient="records")

    except Exception as e:
        return f"Error: {str(e)}"
