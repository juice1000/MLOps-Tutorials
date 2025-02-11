import json

import boto3

runtime = boto3.client("sagemaker-runtime", region_name="ap-southeast-1")

endpoint_name = "mlflow-direct"
payload = json.dumps(
    {
        "dataframe_split": {
            "columns": [
                "feat_0",
                "feat_1",
                "feat_2",
                "feat_3",
                "feat_4",
                "feat_5",
                "feat_6",
                "feat_7",
                "feat_8",
                "feat_9",
                "feat_10",
                "feat_11",
                "feat_12",
            ],
            "data": [
                [
                    14.2,
                    1.76,
                    2.45,
                    15.2,
                    112,
                    3.27,
                    3.39,
                    0.34,
                    1.97,
                    6.75,
                    1.05,
                    2.85,
                    1450,
                ]
            ],
        }
    }
)

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name, ContentType="application/json", Body=payload
)

print(response["Body"].read().decode())
