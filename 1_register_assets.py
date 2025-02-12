# prepare_artifacts.py

import os
import pandas as pd
import joblib
import sys

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from azureml.core import Workspace, Dataset, Model
from azureml.core.authentication import InteractiveLoginAuthentication


# Define required config file
config_file = "config.json"

# Check if config.json exists
if not os.path.exists(config_file):
    print(f"❌ Error: Missing required file '{config_file}'.")
    print("➡️  Please ensure 'config.json' is present in the project root.")
    sys.exit(1)  # Abort execution

auth = InteractiveLoginAuthentication()
ws = Workspace.from_config(auth=auth)





# --------------------------
# 1. Generate and Register Dataset
# --------------------------

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Save dataset to CSV
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, 'sample_data.csv')
df.to_csv(data_path, index=False)
print("-- Sample dataset saved to:", data_path)

# Register the dataset in your workspace
datastore = ws.get_default_datastore()



# Upload the file to the datastore (optional, for remote access)
datastore.upload_files(
    files=[data_path],
    target_path='sample-data/',
    overwrite=True,
    show_progress=True
)
print("-- Uploaded files:", data_path)

# Create a TabularDataset from the uploaded CSV file
dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'sample-data/sample_data.csv'))
dataset = dataset.register(workspace=ws, name='sample_data', create_new_version=True)
print("-- Dataset 'sample_data' registered.")

# --------------------------
# 2. Train and Register Model A
# --------------------------

# Train Model A: simple linear regression on the synthetic dataset
model_a = LinearRegression()
model_a.fit(X, y)
os.makedirs("outputs", exist_ok=True)
model_a_path = os.path.join("outputs", "model_a.pkl")
joblib.dump(model_a, model_a_path)
print("-- Model A trained and saved to:", model_a_path)

# Register Model A in Azure ML
model_a_registered = Model.register(workspace=ws,
                                    model_path=model_a_path,
                                    model_name="model_a")
print("-- Model A registered with name 'model_a'.")


# --------------------------
# 3. Train and Register Model B
# --------------------------

# Generate intermediate predictions using Model A
predictions = model_a.predict(X)
# For Model B, define a final target as 1.1 times Model A's predictions (simulate calibration)
final_target = 1.1 * predictions

# Train Model B: using Model A's predictions as the sole feature
model_b = LinearRegression()
model_b.fit(predictions.reshape(-1, 1), final_target)
model_b_path = os.path.join("outputs", "model_b.pkl")
joblib.dump(model_b, model_b_path)
print("-- Model B trained and saved to:", model_b_path)

# Register Model B in Azure ML
model_b_registered = Model.register(workspace=ws,
                                    model_path=model_b_path,
                                    model_name="model_b")
print("-- Model B registered with name 'model_b'.")
