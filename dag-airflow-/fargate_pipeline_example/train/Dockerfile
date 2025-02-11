FROM python:3.8-slim

# Install Python dependencies
RUN pip install mlflow scikit-learn boto3 pandas joblib

WORKDIR /app
COPY train.py .

CMD ["python", "train.py"]