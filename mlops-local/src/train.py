import os

import boto3
import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load environment variables from .env file
load_dotenv()

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
s3_client = session.client("s3")
s3_bucket = "dvc-ml"
s3_artifact_path = "s3://dvc-ml/mlflow-artifacts"


# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "http://0.0.0.0:8080"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Start MLflow experiment
mlflow.set_experiment("outlet_sales_experiment")

with mlflow.start_run():
    # Load data
    df = pd.read_csv("data/preprocessed.csv")
    X = df.drop(columns=["Item_Outlet_Sales"])
    y = df["Item_Outlet_Sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Log to MLflow
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, f"linear_regression_model")
    run = mlflow.active_run()
    best_run_id = run.info.run_id
    mlflow.register_model(f"runs:/{best_run_id}/model", "linear_regression_model")

    print("TEST S3 MODEL")
    # By run ID
    model_uri = f"runs:/{best_run_id}/linear_regression_model"
    print("best_run_id:", best_run_id)
    model = mlflow.sklearn.load_model(model_uri)
