import requests
import json

# Replace with your deployed endpoint's URL
endpoint_url = "http://9ff3c83b-860e-44c0-b5b2-358ee45d77ad.westeurope.azurecontainer.io/score"

# Define the input data (must match the model's expected features)
input_data = {
    "feature_0": [1.2],
    "feature_1": [0.5],
    "feature_2": [3.1],
    "feature_3": [2.3],
    "feature_4": [0.8]
}

# Set the headers for the request
headers = {
    "Content-Type": "application/json"
}

# Send a POST request to the endpoint
response = requests.post(endpoint_url, headers=headers, data=json.dumps(input_data))

# Print the response
print("Response Status Code:", response.status_code)
print("Response Body:", response.json())
