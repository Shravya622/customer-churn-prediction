import os
from pathlib import Path

import boto3

from config import AWS_REGION, BUCKET_NAME


def download_model_from_s3(local_folder):
    # On Elastic Beanstalk, boto3 automatically uses
    # credentials from the EC2 instance profile/role.
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION
    )

    prefix = "registered_model/"

    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=prefix
    )

    for obj in response.get("Contents", []):
        s3_key = obj["Key"]

        if s3_key.endswith("/"):
            continue

        relative_path = s3_key[len(prefix):]
        local_path = Path(local_folder) / relative_path

        os.makedirs(local_path.parent, exist_ok=True)

        print(f"Downloading {s3_key} -> {local_path}")

        s3.download_file(
            BUCKET_NAME,
            s3_key,
            str(local_path)
        )

    print("Registered model downloaded successfully from S3!")