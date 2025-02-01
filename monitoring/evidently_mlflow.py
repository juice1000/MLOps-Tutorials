import joblib
import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from train import get_new_train_test_set, train

# --------------------- Step 1: Load & Prepare Data --------------------- #
iris = load_iris()
iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_frame["target"] = iris.target

reference_data = iris_frame.iloc[:2]  # Reference data (past)
current_data = iris_frame.iloc[2:]  # New data (current)

X_train, X_test, y_train, y_test = train_test_split(
    reference_data[iris.feature_names],
    reference_data["target"],
    test_size=0.2,
    random_state=42,
)

# --------------------- Step 2: Set up MLflow --------------------- #
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("mlops_pipeline")

with mlflow.start_run(run_name="iris_rf_model") as run:
    print("MLflow run started successfully.")

    # --------------------- Step 3: Detect Data Drift --------------------- #
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_data, current_data=current_data)

    # Get drift results as a dictionary
    drift_results = data_drift_report.as_dict()

    # Extract drift percentage
    drift_score = drift_results["metrics"][0]["result"]["share_of_drifted_columns"]
    print(f"Data Drift Score: {drift_score:.2f}")

    # Define thresholds
    LOW_DRIFT_THRESHOLD = 0.3  # Below this → No retraining needed
    HIGH_DRIFT_THRESHOLD = 0.7  # Above this → Retrain immediately

    if drift_score < LOW_DRIFT_THRESHOLD:
        print("✅ Data drift is low, no retraining required.")
        mlflow.log_metric("drift_score", drift_score)
        mlflow.set_tag("status", "no_retrain_needed")

    elif LOW_DRIFT_THRESHOLD <= drift_score < HIGH_DRIFT_THRESHOLD:
        print("⚠️ Moderate drift detected, monitoring required.")
        mlflow.log_metric("drift_score", drift_score)
        mlflow.set_tag("status", "monitoring_drift")

        # Run a prediction to to test the new incoming data against the current model
        # Load the model from the 'models' directory
        model_path = "monitoring/models/random_forest.pkl"
        model = joblib.load(model_path)
        X_train_dummy, X_test, y_train_dummy, y_test = train_test_split(
            current_data[iris.feature_names],
            current_data["target"],
            train_size=0.01,
            random_state=42,
        )
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        if acc < 0.95:
            # If accuracy is below 95%, retrain the model
            X_train, y_train, X_test, y_test = get_new_train_test_set(
                current_data, iris, X_train, y_train
            )
            model, acc, model_path = train(X_train, y_train, X_test, y_test)

            mlflow.sklearn.log_model(model, "random_forest_model")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_artifact(model_path)
        else:
            print("✅ Model accuracy is above 95%, no retraining needed.")
    else:
        print("🚨 Severe drift detected! Retraining model.")
        mlflow.log_metric("drift_score", drift_score)
        mlflow.set_tag("status", "retraining_triggered")

        # --------------------- Step 4: Train & Log Model --------------------- #
        # Use all the reference data + 70% of the new data for retraining
        X_train, y_train, X_test, y_test = get_new_train_test_set(
            current_data, iris, X_train, y_train
        )
        model, acc, model_path = train(X_train, y_train, X_test, y_test)

        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact(model_path)

    # Save drift report to MLflow
    report_path = "data_drift_report.html"
    data_drift_report.save_html(report_path)
    mlflow.log_artifact(report_path)
    mlflow.log_dict(drift_results, "data_drift_summary.json")

    print("✅ Pipeline completed successfully!")


# We can also save our data drift report to workspace and view it in the evidently UI
