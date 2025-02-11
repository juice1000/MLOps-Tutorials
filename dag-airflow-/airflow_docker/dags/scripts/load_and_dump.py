from io import StringIO

import pandas as pd

S3_BUCKET = "sagemaker-ap-southeast-1-619071320705"


def load_csv_from_s3(file_key, s3):
    """Load CSV file from S3"""
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=file_key)
        df = pd.read_csv(response["Body"])
        print(f"Loaded {file_key} from S3.")
        return df
    except s3.exceptions.NoSuchKey:
        print(f"File {file_key} not found in S3.")
        return None


def dump_to_s3(df, file_key, s3):
    """Dump DataFrame to S3"""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=S3_BUCKET, Key=file_key, Body=csv_buffer.getvalue())
    print(f"Saved {file_key} to S3.")
