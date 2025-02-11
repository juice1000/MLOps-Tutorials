"""
train.py

Trains a Random Forest classifier on the Wine dataset, performs a simple hyperparameter sweep,
logs results to MLflow, and registers the best model to the MLflow Model Registry.
"""

import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split


def train_and_log_model(X_train, y_train, X_val, y_val, max_depth, n_estimators):
    """
    Trains a Random Forest classifier with the given hyperparameters,
    logs metrics and artifacts to MLflow, and returns the trained model plus run info.
    """
    with mlflow.start_run(nested=True) as child_run:
        # Log hyperparameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        # Initialize and train the model
        model = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators, random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        mlflow.log_metric("accuracy", accuracy)

        # Create and log confusion matrix plot
        disp = ConfusionMatrixDisplay.from_estimator(
            model, X_val, y_val, cmap=plt.cm.Blues
        )
        disp.plot()
        plt.title(f"RF (max_depth={max_depth}, n_estimators={n_estimators})")

        # Create plot directory if it doesn't exist
        base_dir = os.path.dirname(__file__)
        plot_dir = os.path.join(base_dir, "plot")
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = os.path.join(
            plot_dir, f"confusion_matrix_d{max_depth}_e{n_estimators}.png"
        )

        plt.savefig(plot_filename)
        plt.close()
        mlflow.log_artifact(plot_filename)

        # Save model locally and log to MLflow
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        local_model_path = os.path.join(
            models_dir, f"random_forest_d{max_depth}_e{n_estimators}.pkl"
        )
        joblib.dump(model, local_model_path)

        # Log the model artifact (named "model" within the run)
        mlflow.sklearn.log_model(model, artifact_path="model")

        return model, accuracy, child_run.info.run_id


def main():
    # Set or create an MLflow experiment
    mlflow.set_experiment("mlops-random-forest-demo")

    # Load dataset
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Simple hyperparameter sweep
    max_depth_values = [5, 10]
    n_estimators_values = [50, 100]

    best_accuracy = 0.0
    best_model = None
    best_run_id = None
    best_params = {}

    # Start a parent run for neat grouping
    with mlflow.start_run(run_name="random_forest_sweep") as parent_run:
        for max_depth in max_depth_values:
            for n_estimators in n_estimators_values:
                print(
                    f"Training RF: max_depth={max_depth}, n_estimators={n_estimators}"
                )
                model, accuracy, run_id = train_and_log_model(
                    X_train, y_train, X_val, y_val, max_depth, n_estimators
                )
                print(f" - Accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_run_id = run_id
                    best_params = {"max_depth": max_depth, "n_estimators": n_estimators}

        # Log the best result to the parent run
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.log_param("best_max_depth", best_params["max_depth"])
        mlflow.log_param("best_n_estimators", best_params["n_estimators"])

    # =============================
    # Model Registry Registration
    # =============================

    # 1. Construct the URI of the "model" artifact from the best run
    best_model_uri = f"runs:/{best_run_id}/model"

    # 2. Register the model with the Model Registry under a chosen name
    #    The name can be an existing or new model name in your Registry
    model_name = "WineRandomForest"
    print(
        f"\nRegistering the best model (run_id={best_run_id}) to MLflow Model Registry '{model_name}'"
    )

    client = MlflowClient()
    registered_model = mlflow.register_model(model_uri=best_model_uri, name=model_name)

    # (Optional) Transition the newly registered model to "Staging" or "Production"
    model_version = registered_model.version
    print(f"Registered model version: {model_version}")

    # Move the version from "None" to "Staging"
    client.transition_model_version_stage(
        name=model_name, version=model_version, stage="Staging"
    )
    print(f"Model version {model_version} transitioned to Staging.")

    # Alternatively, you can transition directly to Production if you wish:
    # client.transition_model_version_stage(
    #     name=model_name,
    #     version=model_version,
    #     stage="Production"
    # )

    print(
        f"\nBest model: max_depth={best_params['max_depth']}, "
        f"n_estimators={best_params['n_estimators']} with accuracy={best_accuracy:.4f}\n"
    )

    print(
        "Done. You can now serve the model from the registry or examine in MLflow UI."
    )


if __name__ == "__main__":
    main()
