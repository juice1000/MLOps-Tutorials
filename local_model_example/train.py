import os

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Optional: If you have local data, read it using pandas
# import pandas as pd
# data = pd.read_csv("data/sample.csv")


def main():
    # Optional: Start MLflow run for logging
    mlflow.set_experiment("local-mlops-demo")  # Creates or uses existing experiment

    with mlflow.start_run():
        # Load data (Iris for simplicity, or replace with your dataset)
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train a simple model
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")

        # Log metric to MLflow
        mlflow.log_metric("accuracy", acc)

        # Save model artifact
        os.makedirs("models", exist_ok=True)
        model_path = "models/logreg_model.pkl"
        joblib.dump(model, model_path)

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
