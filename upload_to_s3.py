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

local_folder = "registered_model"

for root, dirs, files in os.walk(local_folder):
    for file in files:
        local_path = os.path.join(root, file)

        s3_path = os.path.relpath(local_path, ".")

        print(f"Uploading {local_path} -> {s3_path}")

        s3.upload_file(
            local_path,
            BUCKET_NAME,
            s3_path
        )

print("All model artifacts uploaded successfully!")