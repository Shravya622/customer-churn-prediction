import os
import boto3
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, BUCKET_NAME

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

LOCAL_FOLDER = "registered_model"
S3_FOLDER = "registered_model"

for root, dirs, files in os.walk(LOCAL_FOLDER):
    for file in files:
        local_path = os.path.join(root, file)

        relative_path = os.path.relpath(local_path, LOCAL_FOLDER)
        s3_key = f"{S3_FOLDER}/{relative_path}".replace("\\", "/")

        print(f"Uploading {local_path} -> s3://{BUCKET_NAME}/{s3_key}")

        s3.upload_file(
            local_path,
            BUCKET_NAME,
            s3_key
        )

print("Registered model uploaded successfully to S3!")