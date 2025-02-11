FROM --platform=linux/amd64 python:3.8-slim-buster as build

# Install Python dependencies
RUN pip install boto3 pandas evidently

WORKDIR /app
COPY data-validate.py .

CMD ["python", "data-validate.py"]