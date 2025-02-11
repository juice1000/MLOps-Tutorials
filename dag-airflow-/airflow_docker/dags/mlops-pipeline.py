from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from scripts.data_validation import compare
from scripts.deploy import deploy
from scripts.dummy_step import test_boto3
from scripts.evaluate import test_eval
from scripts.train import train

# S3 bucket and script paths
# S3_BUCKET = "glue-jobs-for-ml/"
# SCRIPTS = ["dummy_step_3.py"]


# Download the scripts from S3 to /tmp in MWAA
# def download_script_from_s3(script_name):
#     s3 = boto3.client("s3")
#     local_path = f"/tmp/{script_name}"
#     s3.download_file(S3_BUCKET, f"jobs/{script_name}", local_path)
#     sys.path.append("/tmp")
#     return local_path


# Import and execute the script dynamically
# def run_script(script_name):
#     compare()


# Default arguments
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 2, 5),
    "catchup": False,
}

# Define DAG
dag = DAG(
    "mlops-pipeline",
    default_args=default_args,
    schedule_interval=None,  # Manually trigger
    catchup=False,
)

# Create tasks
# task1 = PythonOperator(
#     task_id="hello_world_1",
#     python_callable=test_boto3,
#     op_kwargs={"script_name": "dummy_step_3.py"},
#     dag=dag,
# )
task1 = PythonOperator(
    task_id="compare",
    python_callable=compare,
    op_kwargs={"script_name": "data_validation.py"},
    dag=dag,
)
task2 = PythonOperator(
    task_id="train",
    python_callable=train,
    provide_context=True,
    op_kwargs={"script_name": "train.py"},
    dag=dag,
)

task3 = PythonOperator(
    task_id="evaluate",
    python_callable=test_eval,
    op_kwargs={"script_name": "evaluate.py"},
    provide_context=True,
    dag=dag,
)

task4 = PythonOperator(
    task_id="deploy",
    python_callable=deploy,
    op_kwargs={"script_name": "deploy.py"},
    dag=dag,
)

# Define task order
task1 >> task2 >> task3 >> task4
