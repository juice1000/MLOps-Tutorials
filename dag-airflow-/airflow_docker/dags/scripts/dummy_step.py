import boto3
import evidently
import pandas as pd

# AWS Configuration
S3_BUCKET = "sagemaker-ap-southeast-1-619071320705"
OLD_FILE_KEY = "data/iris.csv"  # Previous dataset
NEW_FILE_KEY = "tmp/iris_new.csv"  # Updated dataset
REPORT_JSON_KEY = "tmp/iris_drift_report.json"
REPORT_HTML_KEY = "tmp/iris_drift_report.html"


def hi():
    print(evidently.__version__)
    print(pd.__version__)
    print("HI")
    return "HI"


def test_boto3():
    print(boto3.__version__)
    print("S3_BUCKET:", S3_BUCKET)
    # Set profile explicitly
    # boto3.setup_default_session(profile_name="default")
    client = boto3.client("s3")
    try:
        response = client.list_buckets()
        print("List of S3 buckets:", response["Buckets"])
        response = client.get_object(Bucket=S3_BUCKET, Key=OLD_FILE_KEY)
        print("Loaded", OLD_FILE_KEY)

    except Exception as e:
        print("Error:", e)

    return "Boto3 version printed"
