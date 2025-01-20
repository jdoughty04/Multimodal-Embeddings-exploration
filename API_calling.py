import os
import pandas as pd
from zipfile import ZipFile
from tqdm import tqdm
from pathlib import Path
import time
import requests
import logging
import json
import threading

from google.cloud import storage
from google.cloud import aiplatform
import vertexai

from vertexai.vision_models import MultiModalEmbeddingModel, Video, VideoSegmentConfig

from google.oauth2 import service_account

gcs_service_account_file = 'gcs-service-account-key.json'
gcs_credentials = service_account.Credentials.from_service_account_file(
    gcs_service_account_file
)

# Set the path to your service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(), 'jdo.json')

# (Optional) Force the use of REST transport
os.environ['GOOGLE_API_USE_REST'] = 'true'

# Set GRPC verbosity for debugging (optional)
os.environ['GRPC_VERBOSITY'] = 'DEBUG'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'openvid-441115'
own_project_id = 'openvid-441115'

# User identifier
user_id = 'Jonathan'  # Replace with your user identifier


def setup_logging(log_file_path):
    """Sets up logging to file and console."""
    logging.basicConfig(
        filename=log_file_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def download_file(url, dest_path, chunk_size=1024):
    """Downloads a file from a URL with a progress bar."""
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.name,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    bar.update(size)
        logging.info(f"Successfully downloaded {url} to {dest_path}")
        print(f"Successfully downloaded {url} to {dest_path}")
        return True
    except requests.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        print(f"Failed to download {url}: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extracts a ZIP file to a specified directory."""
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info(f"Successfully extracted {zip_path} to {extract_to}")
        print(f"Successfully extracted {zip_path} to {extract_to}")
        return True
    except Exception as e:
        logging.error(f"Failed to extract {zip_path}: {e}")
        print(f"Failed to extract {zip_path}: {e}")
        return False


def download_dataset_part(part_number, output_directory):
    """Downloads and extracts a specific part of the OpenVid dataset."""
    download_dir = Path(output_directory) / "download"
    video_dir = Path(output_directory) / "video"
    data_dir = Path(output_directory) / "data" / "train"

    download_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    log_file = download_dir / "download_log.txt"
    setup_logging(log_file)

    base_url = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/"

    # Download ZIP file
    zip_filename = f"OpenVid_part{part_number}.zip"
    zip_path = download_dir / zip_filename
    if zip_path.exists():
        logging.info(f"File {zip_path} already exists. Skipping download.")
        print(f"File {zip_path} already exists. Skipping download.")
    else:
        url = f"{base_url}{zip_filename}"
        success = download_file(url, zip_path)
        if not success:
            logging.error(f"Failed to download {zip_filename}")
            print(f"Failed to download {zip_filename}")
            return False, None  # Return False and None

    # Check if the extraction directory already exists
    # Since the folder name is random, we'll identify it after extraction
    # First, list existing directories before extraction
    existing_dirs = set([d.name for d in video_dir.iterdir() if d.is_dir()])

    # Extract the ZIP
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(video_dir)
        logging.info(f"Successfully extracted {zip_path} to {video_dir}")
        print(f"Successfully extracted {zip_path} to {video_dir}")
    except Exception as e:
        logging.error(f"Failed to extract {zip_path}: {e}")
        print(f"Failed to extract {zip_path}: {e}")
        return False, None

    # Identify new directories added after extraction
    new_dirs = set([d.name for d in video_dir.iterdir() if d.is_dir()]) - existing_dirs

    if not new_dirs:
        logging.error(f"No new directories found after extracting {zip_path}")
        print(f"No new directories found after extracting {zip_path}")
        return False, None

    # Assuming the ZIP contains a single top-level directory
    extracted_folder_name = new_dirs.pop()
    extracted_folder_path = video_dir / extracted_folder_name

    logging.info(f"Identified extracted folder: {extracted_folder_path}")
    print(f"Identified extracted folder: {extracted_folder_path}")

    # Download data CSV file
    data_filename = f"OpenVid-1M.csv"
    data_url = f"{base_url}data/train/{data_filename}"
    data_path = data_dir / data_filename
    if data_path.exists():
        logging.info(f"Data file {data_path} already exists. Skipping download.")
        print(f"Data file {data_path} already exists. Skipping download.")
    else:
        download_file(data_url, data_path)

    return True, extracted_folder_path  # Return True and the path


def download_dataset_part_async(part_number, output_directory, download_complete_event):
    """Downloads and extracts a specific part of the OpenVid dataset asynchronously."""
    try:
        success, _ = download_dataset_part(part_number, output_directory)
        if not success:
            logging.error(f"Failed to download dataset part {part_number}")
            print(f"Failed to download dataset part {part_number}")
    finally:
        download_complete_event.set()


def blob_exists(storage_client, bucket_name, blob_name):
    """Checks if a blob exists in GCS."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        logging.error(f"Failed to check if blob {blob_name} exists: {e}")
        print(f"Failed to check if blob {blob_name} exists: {e}")
        return False


def process_videos_in_part(part_number, extracted_folder_path, output_directory, bucket_name, user_id):
    """Processes videos in a dataset part: filtering, embedding, uploading, and cleaning."""
    logging.info(f"Starting processing for part {part_number}")
    print(f"Starting processing for part {part_number}")

    data_dir = Path(output_directory) / "data" / "train"

    # Read the CSV file
    data_filename = f"OpenVid-1M.csv"
    data_path = data_dir / data_filename
    if not data_path.exists():
        logging.error(f"Data file {data_path} does not exist.")
        print(f"Data file {data_path} does not exist.")
        return

    try:
        video_df = pd.read_csv(data_path)
    except Exception as e:
        logging.error(f"Failed to read CSV file {data_path}: {e}", exc_info=True)
        print(f"Failed to read CSV file {data_path}: {e}")
        return

    print("CSV Columns:", video_df.columns.tolist())

    # Create a mapping from video filenames to their full paths by recursively searching the extracted folder
    video_mapping = {mp4_path.name: str(mp4_path) for mp4_path in extracted_folder_path.rglob('*.mp4')}

    if not video_mapping:
        logging.error(f"No .mp4 files found in {extracted_folder_path}")
        print(f"No .mp4 files found in {extracted_folder_path}")
        return

    logging.info(f"Found {len(video_mapping)} .mp4 files in {extracted_folder_path}")
    print(f"Found {len(video_mapping)} .mp4 files in {extracted_folder_path}")

    # Map the 'video' column in the CSV to the actual file paths
    def map_video_path(filename):
        path = video_mapping.get(filename)
        return path

    video_df['video'] = video_df['video'].apply(map_video_path)

    # Remove rows where video paths could not be mapped
    initial_count = len(video_df)
    video_df = video_df[video_df['video'].notnull()]
    filtered_count = len(video_df)
    removed_count = initial_count - filtered_count

    if removed_count > 0:
        logging.info(f"Removed {removed_count} entries due to missing video files.")
        print(f"Removed {removed_count} entries due to missing video files.")

    print(f"Videos after mapping: {filtered_count}")

    # Filter videos by duration
    if 'seconds' not in video_df.columns:
        logging.error("CSV file does not contain 'seconds' column.")
        print("CSV file does not contain 'seconds' column.")
        return

    video_df = video_df[(video_df['seconds'] >= 4) & (video_df['seconds'] <= 15)]
    print(f"Videos after duration filter: {len(video_df)}")

    if video_df.empty:
        logging.info("No videos to process after filtering.")
        print("No videos to process after filtering.")
        return

    # Initialize the Vertex AI client
    try:
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        region = os.environ.get('GOOGLE_CLOUD_REGION', 'us-central1')
        aiplatform.init(project=own_project_id, location=region)
        vertexai.init(project=own_project_id, location=region)
        storage_client = storage.Client(project=project_id, credentials=gcs_credentials)
    except Exception as e:
        logging.error(f"Failed to initialize Google Cloud clients: {e}", exc_info=True)
        print(f"Failed to initialize Google Cloud clients: {e}")
        return

    # Get the model for generating embeddings
    try:
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        logging.info("Successfully loaded embedding model.")
        print("Successfully loaded embedding model.")
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}", exc_info=True)
        print(f"Failed to load embedding model: {e}")
        return

    # Process videos in batches of 60
    batch_size = 60
    total_videos = len(video_df)
    total_batches = (total_videos + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        batch_start_time = time.time()

        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_videos)
        batch_df = video_df.iloc[start_idx:end_idx]

        logging.info(f"Processing batch {batch_num + 1}/{total_batches}, videos {start_idx + 1} to {end_idx}")
        print(f"Processing batch {batch_num + 1}/{total_batches}, videos {start_idx + 1} to {end_idx}")

        for index, row in batch_df.iterrows():
            video_path = row['video']
            text = row['caption']

            # Construct the paths for embedding and video in GCS
            embedding_filename = f"embedding_part{part_number}_{index}.json"
            embedding_local_path = os.path.join(output_directory, 'embeddings', embedding_filename)
            embedding_blob_path = f"embeddings/part{part_number}/{embedding_filename}"
            video_blob_path = f"videos/part{part_number}/{os.path.basename(video_path)}"

            # Check if both the embedding and video files already exist in GCS
            embedding_exists = blob_exists(storage_client, bucket_name, embedding_blob_path)
            video_exists = blob_exists(storage_client, bucket_name, video_blob_path)

            if embedding_exists and video_exists:
                logging.info(f"Embeddings and video for {video_path} already exist in GCS. Skipping.")
                print(f"Embeddings and video for {video_path} already exist in GCS. Skipping.")
                continue

            # Get embeddings for the video and text
            try:
                # Upload the video file to cloud storage first to get the GCS URI
                video_gcs_uri = upload_to_bucket(
                    storage_client, bucket_name, video_path, video_blob_path
                )
                if not video_gcs_uri:
                    logging.error(f"Failed to upload video {video_path} to GCS. Skipping.")
                    print(f"Failed to upload video {video_path} to GCS. Skipping.")
                    continue

                # Load the video from the GCS URI
                video_input = Video.load_from_file(video_gcs_uri)

                # Get embeddings for the video and text
                embedding_result = model.get_embeddings(
                    video=video_input,
                    contextual_text=text,
                    dimension=1408,
                )

                # Extract video embeddings
                video_embeddings = [
                    {
                        'start_offset_sec': ve.start_offset_sec,
                        'end_offset_sec': ve.end_offset_sec,
                        'embedding': ve.embedding  # ve.embedding is a list of floats
                    } for ve in embedding_result.video_embeddings
                ]

                # Extract text embedding directly
                text_embedding = embedding_result.text_embedding  # Already a list of floats

                # Prepare data for upload
                embedding_data = {
                    'user_id': user_id,
                    'video_path': video_gcs_uri,  # Use GCS URI
                    'text': text,
                    'video_embeddings': video_embeddings,
                    'text_embedding': text_embedding  # Use text_embedding directly
                }

                # Save embeddings to a JSON file
                os.makedirs(os.path.dirname(embedding_local_path), exist_ok=True)
                with open(embedding_local_path, 'w') as f:
                    json.dump(embedding_data, f)

                # Upload the embedding JSON file to cloud storage
                upload_to_bucket(
                    storage_client, bucket_name, embedding_local_path, embedding_blob_path
                )
                logging.info(
                    f"Uploaded embeddings for video {video_path} to {embedding_blob_path}"
                )
                print(
                    f"Uploaded embeddings for video {video_path} to {embedding_blob_path}"
                )

                # Delete the local embedding file and video file
                os.remove(embedding_local_path)
                os.remove(video_path)
                logging.info(f"Deleted local files for video {video_path}")
                print(f"Deleted local files for video {video_path}")

            except Exception as e:
                logging.error(f"Error processing video {video_path}: {e}", exc_info=True)
                print(f"Error processing video {video_path}: {e}")

            # Handle rate limiting
            time_per_call = 0.5  # Since we can make 120 calls per minute (2 calls per second)
            time.sleep(time_per_call)

        # Sleep to respect the API rate limit
        batch_end_time = time.time()
        batch_elapsed_time = batch_end_time - batch_start_time
        if batch_elapsed_time < 60:
            time_to_sleep = 60 - batch_elapsed_time
            logging.info(f"Sleeping for {time_to_sleep} seconds to respect rate limit")
            print(f"Sleeping for {time_to_sleep} seconds to respect rate limit")
            time.sleep(time_to_sleep)


def delete_dataset_part_data(part_number, output_directory, extracted_folder_path=None):
    """Deletes the data associated with a dataset part."""
    video_dir = Path(output_directory) / "video"
    data_dir = Path(output_directory) / "data" / "train"
    download_dir = Path(output_directory) / "download"
    embeddings_dir = Path(output_directory) / 'embeddings'

    # Delete extracted video directory for this part
    if extracted_folder_path and extracted_folder_path.exists():
        for file in extracted_folder_path.rglob('*'):
            try:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    file.rmdir()
            except Exception as e:
                logging.warning(f"Failed to delete {file}: {e}")
                print(f"Failed to delete {file}: {e}")
        try:
            extracted_folder_path.rmdir()
            logging.info(f"Deleted extracted video directory {extracted_folder_path}")
            print(f"Deleted extracted video directory {extracted_folder_path}")
        except Exception as e:
            logging.warning(f"Failed to remove directory {extracted_folder_path}: {e}")
            print(f"Failed to remove directory {extracted_folder_path}: {e}")

    # Optionally delete the zip file for this part
    zip_filename = f"OpenVid_part{part_number}.zip"
    zip_path = download_dir / zip_filename
    if zip_path.exists():
        zip_path.unlink()
        logging.info(f"Deleted zip file {zip_path}")
        print(f"Deleted zip file {zip_path}")

    # Delete embeddings files for this part
    if embeddings_dir.exists():
        for file in embeddings_dir.rglob(f"embedding_part{part_number}_*.json"):
            try:
                file.unlink()
            except Exception as e:
                logging.warning(f"Failed to delete embedding file {file}: {e}")
                print(f"Failed to delete embedding file {file}: {e}")
        # Remove embeddings directory if empty
        try:
            if not any(embeddings_dir.iterdir()):
                embeddings_dir.rmdir()
        except Exception as e:
            logging.warning(f"Failed to remove embeddings directory {embeddings_dir}: {e}")
            print(f"Failed to remove embeddings directory {embeddings_dir}: {e}")
        logging.info(f"Deleted embeddings for part {part_number} in {embeddings_dir}")
        print(f"Deleted embeddings for part {part_number} in {embeddings_dir}")


def upload_to_bucket(storage_client, bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to a Google Cloud Storage bucket and returns the GCS URI."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        gcs_uri = f'gs://{bucket_name}/{destination_blob_name}'
        logging.info(f"File {source_file_name} uploaded to {destination_blob_name}.")
        print(f"File {source_file_name} uploaded to {destination_blob_name}.")
        return gcs_uri
    except Exception as e:
        logging.error(f"Failed to upload {source_file_name} to {destination_blob_name}: {e}")
        print(f"Failed to upload {source_file_name} to {destination_blob_name}: {e}")
        return None


if __name__ == '__main__':
    output_directory = "mydata"  # Replace with your desired output directory
    bucket_name = "openvideos"  # Replace with your GCS bucket name

    # Ensure the output directory exists
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # List of dataset parts to process
    parts_to_process = [109]  # Replace  with the parts you want to process

    total_parts = len(parts_to_process)
    current_part_index = 0

    while current_part_index < total_parts:
        current_part_number = parts_to_process[current_part_index]
        logging.info(f"Downloading dataset part {current_part_number}")
        print(f"Downloading dataset part {current_part_number}")
        success, extracted_folder_path = download_dataset_part(current_part_number, output_directory)
        if not success:
            logging.error(f"Failed to download or extract dataset part {current_part_number}")
            print(f"Failed to download or extract dataset part {current_part_number}")
            exit(1)  # Or handle the error appropriately

        # Process the current part
        process_videos_in_part(current_part_number, extracted_folder_path, output_directory, bucket_name, user_id)

        # Delete data of the current part
        delete_dataset_part_data(current_part_number, output_directory, extracted_folder_path)

        current_part_index += 1
