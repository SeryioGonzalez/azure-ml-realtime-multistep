from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig  # ✅ Correct import path
from azureml.core.webservice import AciWebservice, Webservice

# Connect to Azure ML workspace
ws = Workspace.from_config()

# Define the inference environment
print("-- Creating inference environment...")
env = Environment.from_conda_specification("sergio-pipeline-env", "env.yml")

# Register environment in Azure ML
env.register(workspace=ws)
print(f"-- Environment '{env.name}' registered in Azure ML.")


print("-- Defining inference config...")
# Define inference configuration
inference_config = InferenceConfig(
    entry_script="scripts/score.py",
    environment=env
)

# Define deployment configuration (using ACI for simplicity)
print("-- Defining deployment config...")
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=2)

# Get the existing service
service_name = "pipeline-inference-service"
existing_services = [s.name for s in Webservice.list(ws)]

if service_name in existing_services:
    service = Webservice(ws, name=service_name)
    service.delete()
    print(f"-- Deleted existing service: {service_name}")
else:
    print(f"-- Service {service_name} not found.")


print(f"-- Deploying service {service_name}...")
# Deploy the service
service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[Model(ws, "model_a"), Model(ws, "model_b")],  # Ensure models are registered
    inference_config=inference_config,
    deployment_config=deployment_config
)
# Wait for deployment to complete
service.wait_for_deployment(show_output=True)
print(f"Service deployed at: {service.scoring_uri}")
