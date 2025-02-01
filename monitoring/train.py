import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def get_new_train_test_set(current_data, iris, X_train, y_train):
    fraction_current_data = 0.7
    split_index = int(fraction_current_data * len(current_data))
    current_data_train = current_data.iloc[:split_index]
    current_data_test = current_data.iloc[split_index:]

    X_train = pd.concat([X_train, current_data_train[iris.feature_names]])
    y_train = pd.concat([y_train, current_data_train["target"]])
    X_test = current_data_test[iris.feature_names]
    y_test = current_data_test["target"]
    return X_train, y_train, X_test, y_test


def train(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    model_path = "monitoring/models/random_forest.pkl"
    joblib.dump(model, model_path)

    print(f"✅ Model trained successfully with accuracy: {acc:.4f}")
    return model, acc, model_path
