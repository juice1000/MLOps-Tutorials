import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

# Load the Wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define TPOT parameters
tpot_config = {
    "generations": 5,
    "population_size": 20,
    "verbosity": 2,
    "random_state": 42,
}

# Start an MLflow experiment
mlflow.set_experiment("TPOT_Wine_Classification")

with mlflow.start_run():
    # Initialize and train TPOT
    tpot = TPOTClassifier(**tpot_config)
    tpot.fit(X_train, y_train)

    # Evaluate the best model
    accuracy = tpot.score(X_test, y_test)
    print(f"Best model accuracy: {accuracy:.4f}")

    # Log TPOT parameters
    mlflow.log_params(tpot_config)

    # Log accuracy as a metric
    mlflow.log_metric("accuracy", accuracy)

    # Log the best found pipeline
    best_pipeline = str(tpot.fitted_pipeline_)
    mlflow.log_text(best_pipeline, "best_pipeline.txt")

    # Save the best model
    mlflow.sklearn.log_model(tpot.fitted_pipeline_, "tpot_best_model")

    # Export the optimized pipeline as a Python script
    tpot.export("tpot_wine_pipeline.py")
    mlflow.log_artifact("tpot_wine_pipeline.py")

    print("\nBest pipeline logged to MLflow.")

# End of MLflow run
