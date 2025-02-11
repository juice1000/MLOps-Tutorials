import json

import boto3

# Create a Boto3 client for SageMaker
client = boto3.client("sagemaker-runtime")


def lambda_handler(event, context):
    # Set the name of the SageMaker endpoint to invoke
    endpoint_name = "mlflow-direct"

    # Set the content type of the input data
    content_type = "application/json"

    body = json.loads(event["body"])
    # Extract dataframe_split data
    dataframe_split = body.get("dataframe_split", {})
    columns = dataframe_split.get("columns", [])
    data = dataframe_split.get("data", [])

    # Convert back to JSON for SageMaker
    payload = json.dumps({"dataframe_split": {"columns": columns, "data": data}})

    # Invoke the SageMaker endpoint
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType=content_type, Body=payload
    )

    # Get the response from the SageMaker endpoint
    result = response["Body"].read().decode("utf-8")

    # Return the result
    return {"statusCode": 200, "body": result}
