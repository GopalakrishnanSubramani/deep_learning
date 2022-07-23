from google.cloud import storage


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    bucket_name = "dogcat-custom-dataset"
    # The path to your file to upload
    source_file_name = "/home/krish/Documents/TF/dog_cat/check_obj/cat.0.jpg"
    # The ID of your GCS object
    destination_blob_name = "cat.0.jpg"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


# from google.cloud import storage


# def upload_blob_from_memory(bucket_name, contents, destination_blob_name):
#     """Uploads a file to the bucket."""

#     # The ID of your GCS bucket
#     bucket_name = "dogcat-custom-dataset"

#     # The contents to upload to the file
#     contents = "these are my contents"

#     # The ID of your GCS object
#     destination_blob_name = "storage-object-name"

#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)

#     blob.upload_from_string(contents)

#     print(
#         f"{destination_blob_name} with contents {contents} uploaded to {bucket_name}."
#     )