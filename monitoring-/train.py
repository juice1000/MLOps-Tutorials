import os

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def dep_get_new_train_test_set(current_data, iris, X_train, y_train):
    fraction_current_data = 0.7
    split_index = int(fraction_current_data * len(current_data))
    current_data_train = current_data.iloc[:split_index]
    current_data_test = current_data.iloc[split_index:]

    X_train = pd.concat([X_train, current_data_train[iris.feature_names]])
    y_train = pd.concat([y_train, current_data_train["target"]])
    X_test = current_data_test[iris.feature_names]
    y_test = current_data_test["target"]
    return X_train, y_train, X_test, y_test


def get_new_train_test_set(current_data, reference_data):
    data_concat = pd.concat([reference_data, current_data])
    data_concat.to_csv("data/iris_consolidated.csv", index=False)
    X = data_concat.drop(columns=["target"])
    y = data_concat["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_test, y_test


def train(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # TODO: Build it mlflow style!
    model_path = "models/random_forest.pkl"
    joblib.dump(model, model_path)

    print(f"✅ Model trained successfully with accuracy: {acc:.4f}")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(model, "random_forest_model")
