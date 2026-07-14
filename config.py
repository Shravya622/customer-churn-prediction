import os
from dotenv import load_dotenv

# Loads .env locally during development
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")