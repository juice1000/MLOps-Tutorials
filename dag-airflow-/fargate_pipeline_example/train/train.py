import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train():
    # Create dummy data
    X = np.arange(100).reshape(-1, 1)
    y = 3 * X.squeeze() + np.random.randn(100) * 10

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    # Log metrics with MLflow
    mlflow.log_param("coef", model.coef_[0])
    mlflow.log_metric("mse", mse)
    print(f"Trained model with coef: {model.coef_[0]}, mse: {mse}")


if __name__ == "__main__":
    mlflow.set_experiment("demo_experiment")
    with mlflow.start_run():
        train()
