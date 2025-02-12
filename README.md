# Azure ML Realtime Endpoint - Multistep Inference Demo

This project demonstrates how to deploy and use a **real-time inference endpoint** in **Azure Machine Learning** for a **multistep inference model**. It covers asset registration, model deployment, and inference using an Azure-hosted endpoint.

## 🚀 Overview

The project follows these steps:

1. **Register Assets**: Prepare and register the dataset and models in Azure ML.
2. **Deploy Inference Model**: Deploy a real-time inference endpoint using Azure ML.
3. **Use Realtime Endpoint**: Send inference requests to the deployed model.

## 📁 Project Structure

./ │── 1_register_assets.py # Registers dataset and models in Azure ML │── 2_deploy_inference_model.py # Deploys the inference models as a real-time endpoint │── 3_use_realtime_endpoint.py # Sends requests to the deployed endpoint │── env.yml # Conda environment file for dependencies │── config.json # Azure ML workspace configuration │── data/ │ └── sample_data.csv # Sample dataset │── outputs/ │ ├── model_a.pkl # Trained Model A │ ├── model_b.pkl # Trained Model B │── scripts/ │ ├── model_a_inference.py # Inference logic for Model A │ ├── model_b_inference.py # Inference logic for Model B │ ├── score.py # Entry script for Azure ML inference │── old/ │ └── 2_pipeline.py # Previous version of the deployment pipeline │── .gitignore # Git ignore file