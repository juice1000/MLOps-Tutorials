#!/bin/bash -xue
# Builds the Airflow Docker container

# Initialize the Airflow database
docker compose up airflow-webserver airflow-scheduler postgres mlflow-server -d
docker compose exec airflow-webserver airflow db init

# Restart Airflow services
docker compose restart

# Create an admin user
docker compose exec airflow-webserver airflow users create \
    --username admin \
    --password admin \
    --firstname Airflow \
    --lastname Admin \
    --role Admin \
    --email admin@example.com

