
#first export varibles
# export GOOGLE_APPLICATION_CREDENTIALS="/home/krish/Documents/TF/training-data-analyst/dogcat-custom-dataset-23f38605dd95.json"

# Imports the Google Cloud client library
from google.cloud import storage

# Instantiates a client
storage_client = storage.Client()

# The name for the new bucket
bucket_name = "dogcat-custom-dataset"

# Creates the new bucket
bucket = storage_client.create_bucket(bucket_name)

print(f"Bucket {bucket.name} created.")