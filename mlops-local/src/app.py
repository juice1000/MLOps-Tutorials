import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# Load the latest model from MLflow Model Registry
MODEL_NAME = "MyBestModel"
MODEL_STAGE = "Production"  # Load only the production-ready model

print(f"Loading model {MODEL_NAME} from MLflow...")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")


@app.get("/")
def home():
    return {"message": "ML Model API is running!"}


@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
