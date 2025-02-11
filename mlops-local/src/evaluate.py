import json
import os
import pickle

import boto3
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error

# AWS Credentials - Load from Environment Variables (Best Practice)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")  # Default region

print("AWS_ACCESS_KEY_ID:", AWS_ACCESS_KEY)
print("AWS_SECRET_ACCESS_KEY:", AWS_SECRET_KEY)
# Ensure AWS credentials are available
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError(
        "❌ AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
    )

# Initialize boto3 session (Optional, ensures credentials work)
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)
s3_bucket = "dvc-ml"
s3_client = session.client("s3")


# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "http://0.0.0.0:8080"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("outlet_sales_experiment")

# Load preprocessed data
df = pd.read_csv("data/preprocessed.csv")
X = df.drop(columns=["Item_Outlet_Sales"])
y = df["Item_Outlet_Sales"]

# Load previous best MSE
previous_metrics_path = "results/metrics.json"
previous_mse = float("inf")  # Default high value
# load file from S3
try:
    s3_client.download_file("dvc-ml", "results/metrics.json", previous_metrics_path)
except:
    print("No previous metrics found.")
if os.path.exists(previous_metrics_path):
    with open(previous_metrics_path, "r") as f:
        previous_mse = json.load(f).get("mse", float("inf"))


# Load new model
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions(
    filter_string="name='linear_regression_model'",
    max_results=1,
    order_by=["creation_timestamp DESC"],
)

new_mse = float("inf")  # Default high value
best_model_uri = None

# Test the new model
for version in versions:
    run_id = version.run_id
    model_uri = f"runs:/{run_id}/linear_regression_model"

    model = mlflow.pyfunc.load_model(model_uri)

    with mlflow.start_run():
        # Evaluate new model
        y_pred = model.predict(X)
        new_mse = mean_squared_error(y, y_pred)

        # Log to MLflow
        mlflow.log_metric("mse", new_mse)
        mlflow.sklearn.log_model(model, "linear_regression_model")

        # Compare and register the best model
        # if new_mse < previous_mse:
        print(
            f"✅ New model is better (MSE: {new_mse} < {previous_mse}). Registering it."
        )

        # Save new metrics to S3
        json_file = json.dumps({"mse": new_mse})
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=previous_metrics_path,
            Body=json_file,
        )
        # Save model to S3 -
        model_path = "models/model.pkl"
        model_bytes = pickle.dumps(model)
        s3_client.put_object(Bucket=s3_bucket, Key=model_path, Body=model_bytes)
        print(f"🚀 Model saved to s3://{s3_bucket}/{model_path}")

        # Transition latest model to production
        client.transition_model_version_stage(
            name="linear_regression_model",
            version=version.version,
            stage="Production",
        )
        print(f"🚀 Model version {version.version} is now in Production!")
