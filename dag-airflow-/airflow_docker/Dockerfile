FROM apache/airflow:2.6.3

# Copy and install requirements.txt
#Change ownership so 'airflow' user can install dependencies
# Switch back to airflow user
USER root
RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

USER airflow
COPY --chown=airflow:airflow requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Set working directory
WORKDIR /opt/airflow