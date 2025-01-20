import pandas as pd
import numpy as np
import ast
import os
import umap
import json
import hdbscan
from sklearn.cluster import DBSCAN

# ---------------------------
# Configuration
# ---------------------------

# Path to  CSV file containing embeddings
CSV_PATH = 'embeddings_data3.csv'

# Output path for UMAP embeddings
VIDEO_UMAP_OUTPUT = 'video_umap_embeddings5.csv'

# GCS Public URL format
GCS_PUBLIC_URL_TEMPLATE = 'https://storage.googleapis.com/{bucket}/{path}'

# ---------------------------
# Data Loading and Preparation
# ---------------------------

def load_video_embeddings(csv_path):
    """
    Load and process video embeddings from the CSV file, including the 'text' column.

    Parameters:
    - csv_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame with video_path, normalized embeddings, video_url, and text.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()

    # Function to parse video embeddings
    def parse_video_embeddings(embedding_str):
        try:
            return ast.literal_eval(embedding_str)
        except Exception as e:
            print(f"Error parsing video embeddings: {e}")
            return []

    # Apply parsing
    df['video_embeddings'] = df['video_embeddings'].apply(parse_video_embeddings)

    # Expand video embeddings if multiple segments per video
    df_expanded = df.explode('video_embeddings').reset_index(drop=True)

    # Extract the embedding vector from video_embeddings
    df_expanded['video_embedding'] = df_expanded['video_embeddings'].apply(
        lambda x: x['embedding'] if isinstance(x, dict) and 'embedding' in x else []
    )

    # Handle the 'text' column
    if 'text' in df_expanded.columns:
        df_expanded['text'] = df_expanded['text'].fillna('')
    else:
        df_expanded['text'] = ''

    # Drop the intermediate column
    df_expanded = df_expanded.drop(columns=['video_embeddings'])

    # Prepare video DataFrame
    df_video = df_expanded[['video_embedding', 'video_path', 'text']].copy()
    df_video = df_video.dropna(subset=['video_embedding'])
    df_video['video_embedding'] = df_video['video_embedding'].apply(lambda x: np.array(x))

    # Aggregate video embeddings per video_path by computing the centroid
    df_video_grouped = df_video.groupby('video_path').agg({
        'video_embedding': lambda embeddings: np.mean(np.vstack(embeddings), axis=0),
        'text': lambda texts: ' '.join(texts)
    }).reset_index()

    # Normalize the embeddings
    df_video_grouped['video_embedding_normalized'] = df_video_grouped['video_embedding'].apply(
        lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x
    )

    # Generate Public URLs
    bucket_name = 'openvideos'  # Your GCS bucket name
    df_video_grouped['video_url'] = df_video_grouped['video_path'].apply(
        lambda path: GCS_PUBLIC_URL_TEMPLATE.format(
            bucket=bucket_name,
            path=path.replace(f'gs://{bucket_name}/', '')
        )
    )

    return df_video_grouped[['video_path', 'video_embedding_normalized', 'video_url', 'text']]

# ---------------------------
# UMAP Dimensionality Reduction
# ---------------------------

def compute_umap(embeddings, n_components=2, random_state=42):
    """
    Compute UMAP embeddings.

    Parameters:
    - embeddings (np.ndarray): Array of shape (n_samples, n_features).
    - n_components (int): Number of dimensions for UMAP.
    - random_state (int): Seed for reproducibility.

    Returns:
    - np.ndarray: UMAP reduced embeddings.
    """
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings

# ---------------------------
# Saving UMAP Embeddings
# ---------------------------

def save_umap_embeddings(df, umap_embeddings, id_column, output_path):
    """
    Save UMAP embeddings to a CSV file along with existing data and original embeddings.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the original data.
    - umap_embeddings (np.ndarray): UMAP reduced embeddings.
    - id_column (str): Column name for the identifier.
    - output_path (str): Path to save the CSV file.
    """
    df_umap = pd.DataFrame(umap_embeddings, columns=['umap_x', 'umap_y'])
    df_umap[id_column] = df[id_column].values

    if 'video_embedding_normalized' in df.columns:
        # Convert numpy arrays to lists for proper serialization
        df_umap['video_embedding_normalized'] = df['video_embedding_normalized'].apply(lambda x: x.tolist())
    else:
        df_umap['video_embedding_normalized'] = ''

    if 'video_url' in df.columns:
        df_umap['video_url'] = df['video_url'].values
    else:
        df_umap['video_url'] = ''

    if 'text' in df.columns:
        df_umap['text'] = df['text'].values
    else:
        df_umap['text'] = ''

    # Reorder columns to include all necessary data
    df_umap = df_umap[[id_column, 'umap_x', 'umap_y', 'video_embedding_normalized', 'video_url', 'text']]

    # Serialize the 'video_embedding_normalized' as JSON strings to ensure full storage
    df_umap['video_embedding_normalized'] = df_umap['video_embedding_normalized'].apply(json.dumps)

    df_umap.to_csv(output_path, index=False)
    print(f"UMAP embeddings saved to {output_path}")

# ---------------------------
# Main Execution
# ---------------------------

def main():
    # Load and process video embeddings
    print("Loading video embeddings...")
    df_video = load_video_embeddings(CSV_PATH)
    if df_video.empty:
        print("No video embeddings to process.")
    else:
        print(f"Video embeddings shape: {df_video.shape}")

        # Compute UMAP for video embeddings
        print("Computing UMAP for video embeddings...")
        video_embeddings = np.vstack(df_video['video_embedding_normalized'].values)
        video_umap = compute_umap(video_embeddings)
        save_umap_embeddings(df_video, video_umap, 'video_path', VIDEO_UMAP_OUTPUT)

if __name__ == "__main__":
    main()