
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

mlflow models build-docker \
  --model-uri "models:/WineRandomForest/Staging" \
  -n my-docker-image --enable-mlserver
