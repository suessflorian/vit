from google.cloud import storage

BUCKET_NAME = "florians_results"

def gcs(local_path: str, gcs_path: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Shipped to gcs://{BUCKET_NAME}/{gcs_path}")
