import os
import boto3
from config import (
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    AWS_REGION,
    BUCKET_NAME
)

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

PREFIX = "registered_model/"
LOCAL_DIR = "registered_model"


def download_registered_model():

    if os.path.exists(LOCAL_DIR):
        print("Model already exists locally.")
        return

    os.makedirs(LOCAL_DIR, exist_ok=True)

    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=PREFIX
    )

    if "Contents" not in response:
        raise Exception("registered_model folder not found in S3")

    for obj in response["Contents"]:

        key = obj["Key"]

        if key.endswith("/"):
            continue

        relative_path = os.path.relpath(key, PREFIX)

        local_file = os.path.join(
            LOCAL_DIR,
            relative_path
        )

        os.makedirs(
            os.path.dirname(local_file),
            exist_ok=True
        )

        print(f"Downloading {key}")

        s3.download_file(
            BUCKET_NAME,
            key,
            local_file
        )

    print("Model downloaded successfully.")