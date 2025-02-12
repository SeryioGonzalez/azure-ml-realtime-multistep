# scripts/model_a_inference.py

import argparse
import os
import pandas as pd
from azureml.core import Model
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to input dataset")
    parser.add_argument("--output_data", type=str, help="Directory where output will be written")
    args = parser.parse_args()

    print(f"Reading input data from: {args.input_data}")

    # Load input data (assumed to be CSV)
    input_csv = os.path.join(args.input_data, "sample_data.csv")  # Adjust filename if needed
    df = pd.read_csv(input_csv)

    # Load Model A from Azure ML Model Registry
    model_path = Model.get_model_path("model_a")
    model_a = joblib.load(model_path)
    print(f"Loaded Model A from {model_path}")

    # Perform inference
    features = df.drop(columns=['target'], errors='ignore')  # Ensure features match model
    df['prediction_a'] = model_a.predict(features)

    # Save predictions to output directory
    os.makedirs(args.output_data, exist_ok=True)
    output_csv = os.path.join(args.output_data, "model_a_output.csv")
    df.to_csv(output_csv, index=False)
    print(f"Model A predictions saved to {output_csv}")

if __name__ == "__main__":
    main()
