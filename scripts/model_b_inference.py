# scripts/model_b_inference.py

import argparse
import os
import pandas as pd
from azureml.core import Model
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Directory with output from Model A")
    parser.add_argument("--output_result", type=str, help="Filename for final results")
    args = parser.parse_args()

    print(f"Reading intermediate data from: {args.input_data}")

    # Read input CSV from previous step
    input_csv = os.path.join(args.input_data, "model_a_output.csv")
    df = pd.read_csv(input_csv)

    # Load Model B from Azure ML Model Registry
    model_path = Model.get_model_path("model_b")
    model_b = joblib.load(model_path)
    print(f"Loaded Model B from {model_path}")

    # Perform inference using Model B on Model A's predictions
    df['final_prediction'] = model_b.predict(df[['prediction_a']])

    # Save final results
    df.to_csv(args.output_result, index=False)
    print(f"Final predictions saved to {args.output_result}")

if __name__ == "__main__":
    main()
