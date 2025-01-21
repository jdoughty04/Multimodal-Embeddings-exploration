# OpenVid Latent Space Exploration

This project was orchestrated for my final presentation in the Machine Learning class at UML in 2024. It implements functionality to create and store multimodal embeddings of videos and text, perform clustering, semantic search, and dimensionality reduction for 2D visualization.

## Project Components

### API_calling.py
- Downloads data from the OpenVideo dataset.
- Calls Google's Vertex AI API to get the embeddings for text and video.
- Saves the videos and embeddings to Google Cloud Storage (GCS).

### extract_data.py
- Loads embeddings from GCS into a dataframe.

### cluster.py
- Clusters the embeddings using HDBSCAN.

### myUMAP.py
- Performs UMAP on embeddings for dimensionality reduction.

### main.py
- Displays embeddings in a 2D plot.
- Includes clusters, text summarization, and semantic search functionality.
- Retrieves data of selected points from google cloud

## Presentation
- [Presentation slides with video demo](https://docs.google.com/presentation/d/1tyiTiqCbMVo-Hzz4pQcFkseUlFpZ8C-AMq71Y24cPcg/edit?usp=sharing)
