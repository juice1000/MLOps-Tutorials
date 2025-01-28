"""
inference.py

Loads the "best" saved model (or any model file) and performs inference on a sample data point.
You can adapt this to load a model from MLflow's model registry if you want to skip local artifacts entirely.
"""

import mlflow.pyfunc
import numpy as np
from mlflow.tracking import MlflowClient


def downgrade_model_to_stage(client: MlflowClient, model_name, version, stage):
    client.transition_model_version_stage(name=model_name, version=version, stage=stage)


def evaluate_model_stages(client: MlflowClient, model_name, stage="Staging"):

    if stage == "Staging":
        model_versions = client.get_latest_versions(
            model_name, stages=["None", "Staging"]
        )
    else:  # Production
        model_versions = client.get_latest_versions(
            model_name, stages=["Staging", "Production"]
        )

    # TODO: Get sample inputs
    sample_input = np.array(
        [[12.7, 3.43, 2.36, 21.0, 111.0, 1.19, 1.61, 0.48, 0.99, 3.13, 1.27, 2.4, 463]]
    )

    best_model = None
    best_prediction = None
    best_accuracy = -1

    for version in model_versions:
        model_uri = f"models:/{model_name}/{version.version}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        prediction = model.predict(sample_input)
        accuracy = evaluate_model(prediction)  # Define your evaluation metric

        print(
            f"Model version: {version.version}, Prediction: {prediction[0]}, Accuracy: {accuracy}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = version
            best_prediction = prediction

    if best_model:
        client.transition_model_version_stage(
            name=model_name, version=best_model.version, stage="Production"
        )
        print(f"Deployed model version {best_model.version} to Production")

    print(f"Sample input: {sample_input}")
    print(f"Best Predicted Wine Class: {best_prediction[0]}")


def main():
    # Load the models from MLflow
    model_name = "WineRandomForest"
    client = MlflowClient()
    evaluate_model_stages(client, model_name, stage="Production")


def evaluate_model(prediction):
    # Implement your evaluation logic here
    # For example, return a random accuracy for demonstration purposes
    return np.random.rand()


if __name__ == "__main__":
    main()
