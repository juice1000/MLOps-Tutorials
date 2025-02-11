import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

# Define default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 2, 5),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    "batch_processing_dag",
    default_args=default_args,
    description="A DAG for batch processing using Apache Airflow",
    schedule_interval="@daily",
)


# Define batch processing function
def process_batch():
    """Simulates batch processing of large data using pandas"""
    num_rows = 10_000_000  # 10 million rows
    batch_size = 100_000  # Process in batches

    # Simulate large dataset
    df = pd.DataFrame(
        {"col1": np.random.randint(0, 100, num_rows), "col2": np.random.randn(num_rows)}
    )

    output_dir = "/tmp/processed_batches"
    os.makedirs(output_dir, exist_ok=True)

    # Process data in batches
    for i in range(0, num_rows, batch_size):
        batch = df.iloc[i : i + batch_size]
        batch["col3"] = batch["col1"] * batch["col2"]  # Example transformation
        batch.to_csv(f"{output_dir}/batch_{i}.csv", index=False)

    print(f"Processed {num_rows} rows in batches of {batch_size}")


# Define a task in the DAG
batch_processing_task = PythonOperator(
    task_id="process_large_data_batches",
    python_callable=process_batch,
    dag=dag,
)

# Set task dependencies
batch_processing_task
