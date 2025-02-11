run_dvc_pipeline:
	@dvc pull --force
	@echo "🚀 Starting MLflow..."
	@export AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) && \
	export AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) && \
	export AWS_REGION="us-east-1" && \
	@nohup mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://dvc-ml/mlflow-artifacts/ > mlflow.log 2>&1 &
	@sleep 10
	@curl -s http://127.0.0.1:5000/api/2.0/mlflow/experiments/list || (echo "MLflow did not start!" && exit 1)
	@echo "🔄 Running DVC pipeline..."
	@dvc repro
	@dvc push
	@lsof -t -i:5000 | xargs -r kill