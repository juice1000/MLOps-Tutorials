import os

import boto3
import sagemaker
from sagemaker.deserializers import JSONDeserializer
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer


def deploy_model_sagemaker(
    model_path,
    bucket_name,
    model_name,
    role_arn,
    instance_type="ml.m5.large",
    endpoint_name="lin_reg_model",
):
    """
    Deploys a model.pkl file to AWS SageMaker as a real-time endpoint.

    Parameters:
    - model_path (str): Local path to model.pkl
    - bucket_name (str): S3 bucket to upload the model
    - model_name (str): Name for the SageMaker model and endpoint
    - role_arn (str): AWS IAM role ARN with SageMaker permissions
    - instance_type (str): EC2 instance type for deployment (default: "ml.m5.large")

    Returns:
    - SageMaker Predictor object for making real-time inferences.
    """

    s3 = boto3.client("s3")
    sm_client = boto3.client("sagemaker")
    sagemaker_session = sagemaker.Session()

    # Upload model to S3
    model_s3_path = f"s3://{bucket_name}/{model_name}/model.tar.gz"
    os.system(f"tar -czf model.tar.gz {model_path}")  # Compress model.pkl
    s3.upload_file("model.tar.gz", bucket_name, f"{model_name}/model.tar.gz")

    # Define model container (using a prebuilt Scikit-learn image)
    container = sagemaker.image_uris.retrieve(
        "sklearn", boto3.Session().region_name, version="1.2-1"
    )
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
    except sm_client.exceptions.ClientError:
        endpoint_exists = False

    # Create SageMaker model
    model = Model(
        image_uri=container,
        model_data=model_s3_path,
        role=role_arn,
        sagemaker_session=sagemaker_session,
    )

    if endpoint_exists:
        print(f"✅ Updating existing endpoint: {endpoint_name}")
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            update_endpoint=True,  # Overwrites the existing endpoint
            wait=True,  # Wait until the endpoint is deployed
        )
    else:
        print(f"🚀 Creating new endpoint: {endpoint_name}")
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,  # Creates a new endpoint
            wait=True,  # Wait until the endpoint is deployed
        )

    print(f"Model {model_name} deployed at endpoint: {predictor.endpoint_name}")
    return predictor


def deploy():
    deploy_model_sagemaker(
        model_path="model.pkl",
        bucket_name="aws-glue-assets-619071320705-ap-southeast-1",
        model_name="lin_reg_model",
        role_arn="arn:aws:iam::619071320705:role/service-role/AmazonSageMaker-ExecutionRole-20250204T103093",
        instance_type="ml.m5.large",
        endpoint_name="lin_reg_model",
    )
