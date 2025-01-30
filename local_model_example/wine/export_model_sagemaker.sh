export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
PORT=8080

# Testing out a sagemaker container locally
mlflow deployments run-local \
    --target sagemaker \
    --model-uri "models:/WineRandomForest/Staging" \
    --name "wine-model-test" \
    -C image="test-mlflow" \
    -C port=$PORT

# Build and push the container to ECR
mlflow sagemaker build-and-push-container \
    --container "test-mlflow" 
    # above one is a misleading command name, it's actually the image name


# Create new model - choose the image you just created from ECR
# Deploy the model on Sagemaker
# Create endpoint
# CFN-SM-IM-Lambda-catalog-SageMakerExecutionRole-EGtbxKglVqp1 = Sagemakerexecutionrole
mlflow deployments create -t sagemaker --name mlflow-direct \
    -m "models:/WineRandomForest/Staging" \
    --config region_name="ap-southeast-1" \
    --config execution_role_arn="arn:aws:iam::619071320705:role/CFN-SM-IM-Lambda-catalog-SageMakerExecutionRole-EGtbxKglVqp1" \
    --config image_url="619071320705.dkr.ecr.ap-southeast-1.amazonaws.com/test-mlflow:2.20.0"


# Create & Test endpoint on Sagemaker
# aws sagemaker create-endpoint-config \
#     --endpoint-config-name mlflow-endpoint-config \
#     --production-variants VariantName=AllTraffic,ModelName="mlflow-direct-model-ce9313e23040440c9248",InstanceType=ml.m5.large,InitialInstanceCount=1 \
#     --region ap-southeast-1

# aws sagemaker create-endpoint \
#     --endpoint-name mlflow-endpoint \
#     --endpoint-config-name mlflow-endpoint-config \
#     --region ap-southeast-1

# aws sagemaker describe-endpoint \
#     --endpoint-name mlflow-endpoint \
#     --region ap-southeast-1