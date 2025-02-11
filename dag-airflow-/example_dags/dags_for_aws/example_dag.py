import sys
from datetime import datetime

import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator

# S3 bucket and script paths
S3_BUCKET = "glue-jobs-for-ml/"
SCRIPTS = ["dummy_step_3.py"]


# Download the scripts from S3 to /tmp in MWAA
def download_script_from_s3(script_name):
    s3 = boto3.client("s3")
    local_path = f"/tmp/{script_name}"
    s3.download_file(S3_BUCKET, f"jobs/{script_name}", local_path)
    sys.path.append("/tmp")
    return local_path


# Import and execute the script dynamically
def run_script(script_name):
    script_path = download_script_from_s3(script_name)
    with open(script_path, "r") as file:
        exec(file.read())


# Default arguments
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 2, 5),
    "catchup": False,
}

# Define DAG
dag = DAG(
    "hello_world_dag",
    default_args=default_args,
    schedule_interval=None,  # Manually trigger
    catchup=False,
)

# Create tasks
task1 = PythonOperator(
    task_id="hello_world_1",
    python_callable=run_script,
    op_kwargs={"script_name": "dummy_step_3.py"},
    dag=dag,
)

task2 = PythonOperator(
    task_id="hello_world_2",
    python_callable=run_script,
    op_kwargs={"script_name": "dummy_step_3.py"},
    dag=dag,
)

# Define task order
task1 >> task2
