import argparse
import os
import re
import json
from google.cloud import storage
import pandas as pd
from io import BytesIO

def load_embeddings_from_gcs(bucket_name, parts, prefix='embeddings'):
    """
    Loads embeddings data from specified parts in a Google Cloud Storage bucket.

    Parameters:
    - bucket_name (str): Name of the GCS bucket.
    - parts (list of int): List of part numbers to load (e.g., [1, 2, 3]).
    - prefix (str): The top-level folder in the bucket ('embeddings' by default).

    Returns:
    - list of dict: List containing records extracted from JSON files.
    """
    # Initialize the GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Initialize a list to hold all records
    records = []
    
    for part in parts:
        # Define the folder path, e.g., 'embeddings/part1/'
        folder_path = f"{prefix}/part{part}/"
        print(f"Processing folder: {folder_path}")
        
        # List all blobs in the specified folder
        blobs = bucket.list_blobs(prefix=folder_path)
        
        # Iterate over each blob (file) in the folder
        for blob in blobs:
            # We're interested in JSON files only
            if blob.name.endswith('.json'):
                print(f"Reading file: {blob.name}")
                
                # Download the blob's content as bytes
                try:
                    content = blob.download_as_bytes()
                except Exception as e:
                    print(f"Error downloading file {blob.name}: {e}")
                    continue
                
                try:
                    # Parse the JSON content
                    data = json.loads(content)
                    
                    # Extract required fields
                    video_path = data.get('video_path', '')
                    text = data.get('text', '')
                    video_embeddings = data.get('video_embeddings', [])
                    text_embedding = data.get('text_embedding', [])
                    
                    # Optionally, extract part number and random number from filename
                    filename = os.path.basename(blob.name)
                    match = re.match(r'embedding_part(\d+)_(\d+)\.json', filename)
                    if match:
                        part_num = int(match.group(1))
                        random_num = int(match.group(2))
                    else:
                        part_num = None
                        random_num = None
                        
                    # Append the record to the list
                    records.append({
                        'part': part_num,
                        'random_number': random_num,
                        'video_path': video_path,
                        'text': text,
                        'video_embeddings': video_embeddings,
                        'text_embedding': text_embedding
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {blob.name}: {e}")
                except Exception as e:
                    print(f"Unexpected error processing file {blob.name}: {e}")
    
    return records

def update_csv(existing_csv_path, new_data, output_csv_path):
    """
    Updates the CSV file with new data. If the existing CSV is provided and exists,
    append to it. Otherwise, create a new CSV.

    Parameters:
    - existing_csv_path (str): Path to the existing CSV file (optional).
    - new_data (pd.DataFrame): New data to append.
    - output_csv_path (str): Path to the output CSV file.
    """
    if existing_csv_path:
        if os.path.exists(existing_csv_path):
            print(f"Loading existing CSV file from {existing_csv_path}")
            try:
                existing_df = pd.read_csv(existing_csv_path)
                print(f"Existing CSV has {len(existing_df)} records.")
            except Exception as e:
                print(f"Error reading existing CSV file: {e}")
                existing_df = pd.DataFrame()
        else:
            print(f"Provided CSV file {existing_csv_path} does not exist. A new CSV will be created.")
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()
    
    # Create a DataFrame from new data
    new_df = pd.DataFrame(new_data)
    print(f"New data contains {len(new_df)} records.")
    
    if not existing_df.empty:
        # Optionally, check for duplicates based on 'video_path' or other unique fields
        # For simplicity, we'll just concatenate
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"Combined DataFrame has {len(combined_df)} records.")
    else:
        combined_df = new_df
    
    # Save to the output CSV
    try:
        combined_df.to_csv(output_csv_path, index=False)
        print(f"Data successfully saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

def main():
    # Use predefined variables instead of command-line arguments
    bucket_name = 'openvideos'
    parts = [105, 106]
    existing_csv_path = 'embeddings_data2.csv'  # Assuming no existing CSV path is provided
    output_csv_path = 'embeddings_data3.csv'
    
    # Load new embeddings data from GCS
    new_records = load_embeddings_from_gcs(bucket_name=bucket_name, parts=parts)
    
    if not new_records:
        print("No new records to add. Exiting.")
        return
    
    # Update the CSV file
    update_csv(
        existing_csv_path=existing_csv_path,
        new_data=new_records,
        output_csv_path=output_csv_path
    )


if __name__ == "__main__":
    main()
