import pandas as pd
import numpy as np
import hdbscan
import json

# ---------------------------
# Configuration
# ---------------------------

# Path to the UMAP-processed CSV file
UMAP_CSV_PATH = 'video_umap_embeddings5.csv'

# Output path for clustered data
VIDEO_CLUSTER_OUTPUT = 'video_clusters_hdbscan_minCS_7_minS_7.csv'

# HDBSCAN Configuration
MIN_CLUSTER_SIZE = 7  # The minimum size of clusters
MIN_SAMPLES = 7       # The number of samples in a neighborhood for a point to be considered as a core point (optional)

# ---------------------------
# Loading UMAP Data
# ---------------------------

def load_umap_data(csv_path):
    """
    Load UMAP-processed data from CSV.

    Parameters:
    - csv_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame with UMAP embeddings and other data.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading UMAP CSV file: {e}")
        return pd.DataFrame()
    
    return df

# ---------------------------
# HDBSCAN Clustering
# ---------------------------

def perform_hdbscan_clustering(df):
    """
    Perform HDBSCAN clustering on the normalized video embeddings.

    Parameters:
    - df (pd.DataFrame): DataFrame containing UMAP embeddings and other data.

    Returns:
    - pd.DataFrame: DataFrame with an added 'video_cluster' column.
    """
    if 'video_embedding_normalized' not in df.columns:
        print("Column 'video_embedding_normalized' not found in the DataFrame.")
        return df

    if df.empty:
        print("DataFrame is empty. No clustering performed.")
        df['cluster'] = np.nan
        return df

    print(f"Performing HDBSCAN clustering with min_cluster_size={MIN_CLUSTER_SIZE} and min_samples={MIN_SAMPLES} on video embeddings...")
    
    # Deserialize the JSON strings back to lists
    df['video_embedding_normalized'] = df['video_embedding_normalized'].apply(json.loads).apply(np.array)

    video_embeddings = np.vstack(df['video_embedding_normalized'].values)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES)
    cluster_labels = clusterer.fit_predict(video_embeddings)
    df['cluster'] = cluster_labels

    # Reorder clusters: most populous cluster -> 0, next -> 1, etc.
    # Exclude noise points labeled as -1
    cluster_counts = df['cluster'].value_counts().sort_values(ascending=False)
    clusters_sorted = cluster_counts.index.tolist()
    clusters_sorted = [c for c in clusters_sorted if c != -1]  # Exclude noise

    # Create a mapping from original cluster labels to new labels
    new_cluster_mapping = {original: new for new, original in enumerate(clusters_sorted)}

    # Apply the mapping
    def map_cluster(label):
        if label == -1:
            return -1  # Keep noise as -1
        return new_cluster_mapping.get(label, label)  # Default to original label if not found

    df['cluster'] = df['cluster'].apply(map_cluster)

    # Calculate number of clusters excluding noise (-1)
    unique_clusters = set(df['cluster'])
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    print(f"HDBSCAN clustering completed with {n_clusters} clusters.")

    return df

# ---------------------------
# Saving Clustered Data
# ---------------------------

def save_clustered_data(df, output_path):
    """
    Save clustered data to a CSV file.

    Parameters:
    - df (pd.DataFrame): DataFrame containing clustered data.
    - output_path (str): Path to save the CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Clustered data saved to {output_path}")

# ---------------------------
# Main Execution
# ---------------------------

def main():
    # Load UMAP-processed data
    print("Loading UMAP-processed data...")
    df_umap = load_umap_data(UMAP_CSV_PATH)
    if df_umap.empty:
        print("No data to process for clustering.")
    else:
        # Perform HDBSCAN clustering
        df_clustered = perform_hdbscan_clustering(df_umap)

        # Save clustered data
        save_clustered_data(df_clustered, VIDEO_CLUSTER_OUTPUT)

if __name__ == "__main__":
    main()


