export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# mlflow server has to run!

mlflow models build-docker \
    -m "models:/WineRandomForest/Staging" \
    -n "test-mlflow" \
    --enable-mlserver

