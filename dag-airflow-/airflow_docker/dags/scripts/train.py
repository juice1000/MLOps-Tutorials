import pickle

import boto3
import mlflow
from scripts.load_and_dump import load_csv_from_s3
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# AWS Configuration
S3_BUCKET = "sagemaker-ap-southeast-1-619071320705"
FILE_KEY = "data/iris_consolidated.csv"  # Previous dataset


def train(**kwargs):
    # Initialize MLflow
    mlflow.set_tracking_uri("http://mlflow-server:8083")
    # Check if experiment exists, create it if missing
    experiment_name = "mlops-pipeline-training"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # initialize S3 client
    s3 = boto3.client("s3")

    # Load data from S3 bucket
    data = load_csv_from_s3(FILE_KEY, s3)
    X = data.drop("target", axis=1)
    y = data["target"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        print("starting run")
        # Linear Regression model
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

        # Register model
        run_id = mlflow.active_run().info.run_id
        print("active run_id", run_id)
        # m = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        # print("model", m)
        # mlflow.register_model(f"runs:/{run_id}/model", "lin_reg_model")
        # print("Model saved as lin_reg_model")
        kwargs["ti"].xcom_push(key="run_id", value=run_id)
        print("run_id pushed to XCom")

        # Push pkl to S3
        model_path = "models/model.pkl"
        s3_bucket = "sagemaker-ap-southeast-1-619071320705"
        s3_model_path = f"s3://{s3_bucket}/{model_path}"
        model_binary = pickle.dumps(model)
        s3.put_object(Bucket=s3_bucket, Key=model_path, Body=model_binary)
        print(f"Model saved to {s3_model_path}")
        # mlflow.register_model(f"runs:/{run_id}/model", "lin_reg_model")
