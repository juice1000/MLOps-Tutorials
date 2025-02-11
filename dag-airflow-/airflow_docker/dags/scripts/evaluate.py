import json
import os

import boto3
import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error


def test_eval(**kwargs):
    ti = kwargs["ti"]
    run_id = ti.xcom_pull(key="run_id", task_ids="train")

    try:
        mlflow.set_tracking_uri("http://mlflow-server:8083")
        mlflow.set_experiment("mlops-pipeline-training")
        with mlflow.start_run():
            # load the model
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            print("model", model)

            # TOOD: load validation data and metrics json
            # TODO: evaluate the model
            # TODO: update model if better than previous one

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Push pkl to S3
    # model_path = "models/model.pkl"
    # s3_bucket = "sagemaker-ap-southeast-1-619071320705"
    # s3 = boto3.client("s3")
    # s3_model_path = f"s3://{s3_bucket}/{model_path}"
    # s3.upload_file(model_path, s3_bucket, model_path)
    # print(f"Model saved to {s3_model_path}")


# def evaluate():
#     # Set MLflow tracking URI
#     MLFLOW_TRACKING_URI = "http://mlflow-server:8083"
#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#     mlflow.set_experiment("mlops-pipeline-training")

#     # Load preprocessed data
#     df = pd.read_csv("data/iris_preprocessed.csv")
#     X = df.drop(columns=["target"])
#     y = df["target"]

#     # Load previous best MSE
#     previous_metrics_path = "results/metrics.json"
#     previous_mse = float("inf")  # Default high value
#     print("previous_metrics_path", previous_metrics_path)
#     # load metrics file
#     try:
#         with open(previous_metrics_path, "r") as f:
#             previous_mse = json.load(f).get("mse", float("inf"))
#     except:
#         print("No previous metrics found.")

#     # Load new model
#     client = MlflowClient()
#     versions = client.search_model_versions(
#         filter_string="name='lin_reg_model'",
#         max_results=1,
#         order_by=["creation_timestamp DESC"],
#     )

#     new_mse = float("inf")  # Default high value
#     best_model_uri = None
#     version = versions[0]
#     # Test the new model
#     run_id = version.run_id
#     print("run_id", run_id)
#     model_uri = f"runs:/{run_id}/lin_reg_model"

#     model = mlflow.pyfunc.load_model(model_uri)

#     with mlflow.start_run():
#         # Evaluate new model
#         y_pred = model.predict(X)
#         new_mse = mean_squared_error(y, y_pred)

#         # Log to MLflow
#         mlflow.log_metric("mse", new_mse)
#         mlflow.sklearn.log_model(model, "linear_regression_model")

#         # Compare and register the best model
#         # if new_mse < previous_mse:
#         print(
#             f"✅ New model is better (MSE: {new_mse} < {previous_mse}). Registering it."
#         )

#         # Save new metrics to S3
#         json_file = json.dumps({"mse": new_mse})
#         os.makedirs("results", exist_ok=True)
#         with open(previous_metrics_path, "w") as f:
#             f.write(json_file)

#         # Save model
#         model_path = "models/model.pkl"
#         joblib.dump(model, model_path)

#         # model_bytes = pickle.dumps(model)
#         # print(f"🚀 Model saved to s3://{s3_bucket}/{model_path}")

#         # Transition latest model to production
#         # TODO- debug
#         client.transition_model_version_stage(
#             name="linear_regression_model",
#             version=version,
#             stage="Production",
#         )
#         print(f"🚀 Model version {version} is now in Production!")


# if __name__ == "__main__":
#     evaluate()
#     print("Evaluation completed")
