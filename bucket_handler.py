from google.cloud import storage
from google.cloud.exceptions import NotFound


def get_or_create_bucket(bucket_name):
    try:
        storage_client = storage.Client()
        the_bucket = storage_client.get_bucket(bucket_or_name=bucket_name)
        return the_bucket
    except NotFound:
        return create_bucket(bucket_name)


def create_bucket(bucket_name):

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = "STANDARD"
    new_bucket = storage_client.create_bucket(bucket, location="EU")

    print(
        "Created bucket {} in {} with storage class {}".format(
            new_bucket.name, new_bucket.location, new_bucket.storage_class
        )
    )

    return new_bucket


def upload_to_bucket(bucket, blob_path, data):
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data=data)


def get_bytes_from_blob(bucket, blob_path):
    blob = bucket.get_blob(blob_path)
    return blob.download_as_string()
