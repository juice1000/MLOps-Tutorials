
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

mlflow models serve \
  --model-uri "models:/WineRandomForest/Staging" \
  --port 8080 \
  --no-conda

