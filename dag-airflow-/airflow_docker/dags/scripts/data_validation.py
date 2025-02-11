from io import StringIO

import boto3
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from scripts.load_and_dump import dump_to_s3, load_csv_from_s3

# AWS Configuration
S3_BUCKET = "sagemaker-ap-southeast-1-619071320705"
OLD_FILE_KEY = "data/iris.csv"  # Previous dataset
NEW_FILE_KEY = "tmp/iris_new.csv"  # Updated dataset
REPORT_JSON_KEY = "tmp/iris_drift_report.json"
REPORT_HTML_KEY = "tmp/iris_drift_report.html"


def compare():
    # Load both datasets
    # Initialize S3 client
    s3 = boto3.client("s3")
    df_old = load_csv_from_s3(OLD_FILE_KEY, s3)
    df_new = load_csv_from_s3(NEW_FILE_KEY, s3)

    # Check if both files exist
    if df_old is None or df_new is None:
        print("One or both files are missing. Cannot proceed with comparison.")
        raise Exception("Missing files")

    # Generate Data Drift Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_old, current_data=df_new)

    # Save JSON report
    json_report = report.as_dict()
    json_buffer = StringIO()
    pd.json_normalize(json_report).to_json(json_buffer)

    # # Save HTML report
    # html_buffer = StringIO()
    # report.save_html(html_buffer)

    # # Upload reports to S3
    # s3.put_object(Bucket=S3_BUCKET, Key=REPORT_JSON_KEY, Body=json_buffer.getvalue())
    # s3.put_object(Bucket=S3_BUCKET, Key=REPORT_HTML_KEY, Body=html_buffer.getvalue())

    # print(f"Data drift reports saved to S3: {REPORT_JSON_KEY}, {REPORT_HTML_KEY}")

    # Check if drift is detected
    drift_detected = any(
        dataset["result"]["dataset_drift"]
        for dataset in json_report["metrics"]
        if "dataset_drift" in dataset["result"]
        and dataset["result"]["dataset_drift"] is True
    )

    # If drift is detected, send an alert
    if drift_detected:
        print("Data drift detected! Send an alert.")
        # We consolidate the two csv files into one and save it to S3
        df_combined = pd.concat([df_old, df_new])
        dump_to_s3(df_combined, "data/iris_consolidated.csv", s3)
        # df_combined.to_csv("iris.csv", index=False)
        # s3.upload_file("iris.csv", S3_BUCKET, "data/iris.csv")
        print("Combined dataset saved to S3: iris__.csv")

    else:
        print("No data drift detected.")
        raise Exception("Data drift not detected")
