import joblib
import numpy as np


def main():
    # Load the trained model
    model = joblib.load("models/logreg_model.pkl")

    # Example input (Iris-like features: sepal length, sepal width, etc.)
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

    # Predict
    prediction = model.predict(sample_input)
    print(f"Predicted class: {prediction[0]}")


if __name__ == "__main__":
    main()
