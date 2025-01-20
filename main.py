import os
import pandas as pd
import subprocess
import uuid
import requests  # Ensure you have requests installed: pip install requests
import colorsys  # For generating custom colors
import openai  # For summarization
import vertexai  # For embedding
from vertexai.vision_models import MultiModalEmbeddingModel

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Div, Select, Button, TextInput
from bokeh.layouts import column, row
from bokeh.events import Tap
from bokeh.models import Spacer

import numpy as np
import ast  # For literal_eval
from functools import partial

# -----------------------------------
# Configuration and Constants
# -----------------------------------

UMAP_CSV_PATH = 'clustered_outputs/video_clusters_hdbscan_minCS_7_minS_7.csv'
EMBEDDINGS_CSV_PATH = 'embeddings_data3.csv'
SERVICE_ACCOUNT_KEY_PATH = 'jdo.json'  # Update this path
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment

# Initialize Vertex AI
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_KEY_PATH
vertexai.init(project="openvid-441115", location="us-central1")  # Update to your project and location

# Determine the absolute path to the 'static' directory within the 'interactive' app
app_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(app_dir, "static")
os.makedirs(static_dir, exist_ok=True)

# -----------------------------------
# Helper Functions
# -----------------------------------

def is_url_accessible(url):
    """Check if the video URL is accessible."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"URL accessibility check failed for {url}: {e}")
        return False

def summarize_texts(texts, max_tokens=150, temperature=0.5):
    """
    Summarize a list of texts using OpenAI's GPT-4 model.
    """
    if not texts:
        return "No texts available to summarize."

    if not OPENAI_API_KEY:
        return "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."

    openai.api_key = OPENAI_API_KEY

    combined_text = "\n\n".join(texts)
    prompt = (
        "Here are several summaries of several videos. Within 20 words, describe overall what the videos tend to be about:\n\n"
        + combined_text
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        summary = response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "An error occurred during summarization."

    return summary

def load_embeddings(csv_path):
    """
    Load video embeddings from the CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        return {}

    def parse_embedding(embedding_str, video_path='Unknown'):
        try:
            embedding_list = ast.literal_eval(embedding_str)
            if not isinstance(embedding_list, list):
                print(f"'video_embeddings' is not a list for video '{video_path}'.")
                return np.array([])

            embeddings = []
            for segment in embedding_list:
                if isinstance(segment, dict) and 'embedding' in segment:
                    embeddings.append(segment['embedding'])
                else:
                    print(f"Invalid segment format in video '{video_path}': {segment}")

            if not embeddings:
                print(f"No valid embeddings found for video '{video_path}'.")
                return np.array([])

            avg_embedding = np.mean(embeddings, axis=0)
            return avg_embedding.astype(np.float32)

        except Exception as e:
            print(f"Error parsing embedding for video '{video_path}': {e}")
            return np.array([])

    df['video_embedding_vector'] = df.apply(
        lambda row: parse_embedding(row['video_embeddings'], row.get('video_path', 'Unknown')),
        axis=1
    )

    initial_count = len(df)
    df = df[df['video_embedding_vector'].apply(lambda x: x.size > 0)]
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} videos due to failed embedding parsing.")

    embedding_dict = pd.Series(df.video_embedding_vector.values, index=df.video_path).to_dict()
    return embedding_dict

def get_embedding(video_path, embedding_dict):
    embedding = embedding_dict.get(video_path)
    if embedding is not None and embedding.size > 0:
        return embedding
    else:
        # Handle missing embeddings by returning a zero vector of the same dimension
        if len(embedding_dict) > 0:
            return np.zeros(len(next(iter(embedding_dict.values()))), dtype=np.float32)
        else:
            print("Embedding dictionary is empty. Returning empty array.")
            return np.array([])

def generate_distinct_colors(n):
    """Generate n distinct colors using HSL space."""
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.5
        saturation = 0.7
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#%02x%02x%02x' % tuple(int(255 * c) for c in rgb)
        colors.append(hex_color)
    return colors

# -----------------------------------
# Load Data
# -----------------------------------

# Load embeddings
embedding_dict = load_embeddings(EMBEDDINGS_CSV_PATH)
print(f"Loaded embeddings for {len(embedding_dict)} videos.")

# Load UMAP Data
try:
    df_umap = pd.read_csv(UMAP_CSV_PATH)
    print(f"Total rows in UMAP CSV: {len(df_umap)}")
except Exception as e:
    print(f"Error reading UMAP CSV file '{UMAP_CSV_PATH}': {e}")
    df_umap = pd.DataFrame()

if 'video_path' not in df_umap.columns:
    raise ValueError("'video_path' column not found in UMAP CSV.")

# Apply embeddings
embeddings = df_umap['video_path'].apply(lambda vp: get_embedding(vp, embedding_dict))
if embeddings.isnull().any():
    print("Some embeddings are missing. They will be set to zero vectors.")
    embeddings = embeddings.fillna(np.array([]))

try:
    embeddings_matrix = np.vstack(embeddings.values)
except ValueError as e:
    print(f"Error stacking embeddings: {e}")
    embeddings_matrix = np.array([])

if embeddings_matrix.size == 0:
    print("Embeddings matrix is empty. Exiting.")
    exit(1)

# Normalize embeddings for cosine similarity
try:
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings_matrix / norms
except Exception as e:
    print(f"Error normalizing embeddings: {e}")
    embeddings_norm = np.array([])

# Assign colors
num_clusters = df_umap['cluster'].nunique()
palette = generate_distinct_colors(num_clusters)

def assign_color(cluster_id):
    if 0 <= cluster_id < num_clusters:
        return palette[cluster_id]
    else:
        return "#1f78b4"  # Default color

df_umap['color'] = df_umap['cluster'].apply(assign_color)
df_umap['original_color'] = df_umap['color']
df_umap['fill_alpha'] = 0.7

source = ColumnDataSource(data=dict(
    x=df_umap['umap_x'].tolist(),
    y=df_umap['umap_y'].tolist(),
    video_url=df_umap['video_url'].tolist(),
    video_path=df_umap['video_path'].tolist(),
    text=df_umap['text'].tolist(),
    color=df_umap['color'].tolist(),
    original_color=df_umap['original_color'].tolist(),
    cluster=df_umap['cluster'].tolist(),
    fill_alpha=df_umap['fill_alpha'].tolist()
))

print(f"ColumnDataSource contains {len(source.data['video_url'])} entries.")
print("Sample video URLs:")
for i in range(min(5, len(source.data['video_url']))):
    print(f"Index {i}: {source.data['video_url'][i]}")

# Initialize embedding model
embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
print("Embedding model loaded successfully.")

# -----------------------------------
# Bokeh Plot and Widgets
# -----------------------------------

p = figure(tools="pan,wheel_zoom,box_zoom,reset,tap",
           title="2D Video Embeddings by Cluster",
           width=800, height=600)

scatter_renderer = p.scatter('x', 'y', source=source, size=8, fill_alpha='fill_alpha', color='color', alpha=1.0)

video_div = Div(
    text="<b>Select a point to see the video, its path, and cluster here</b>",
    width=600
)
summary_div = Div(text="<b>Cluster Summary:</b> Select a cluster to see its summary.", width=600, height=200)

search_input = TextInput(title="Search Videos:", placeholder="Enter your search query here...")
search_button = Button(label="Search", button_type="primary")
search_results_div = column(Div(text="<b>Search Results:</b> Enter a query above to find similar videos.", width=800))

unique_clusters = sorted(df_umap['cluster'].unique())
cluster_options = [str(cluster) for cluster in unique_clusters]
cluster_select = Select(title="Select Cluster:", value="All", options=["All"] + cluster_options)
reset_button = Button(label="Reset Filter", button_type="success")

# -----------------------------------
# FFmpeg and Video Display Logic
# -----------------------------------

def download_and_prepare_video(video_url):
    """
    Download and re-encode the video using FFmpeg, and return the local file path.
    """
    if not is_url_accessible(video_url):
        print(f"Invalid or inaccessible video URL: {video_url}")
        return None

    filename = f"output_{uuid.uuid4().hex}.mp4"
    local_path = os.path.join(static_dir, filename)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_url,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "faststart",
        local_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"/interactive/static/{filename}"
    except subprocess.CalledProcessError as e:
        ffmpeg_error = e.stderr.decode('utf-8', 'replace')
        print(f"FFmpeg error: {ffmpeg_error}")
        return None

def display_video(index):
    """
    Given an index in df_umap/source, display the corresponding video in the video_div.
    """
    if index < 0 or index >= len(df_umap):
        video_div.text = "<b>Selected index is out of range.</b>"
        return

    video_url = df_umap.iloc[index]['video_url']
    video_path = df_umap.iloc[index]['video_path']
    text_caption = df_umap.iloc[index]['text']
    cluster_id = df_umap.iloc[index]['cluster']

    # Download and prepare the video
    local_video_src = download_and_prepare_video(video_url)
    if local_video_src is None:
        video_div.text = f"<b>Error processing video from:</b> {video_url}"
        return

    video_div.text = f"""
    <b>Cluster ID:</b> {cluster_id}<br>
    <b>Video Path:</b> {video_path}<br>
    <b>Caption:</b> {text_caption}<br>
    <video width="600" height="400" controls>
    <source src="{local_video_src}" type="video/mp4">
    Your browser does not support the video tag.
    </video>
    <br><br>
    """

# -----------------------------------
# Callbacks
# -----------------------------------

def tap_callback(event):
    selected_indices = source.selected.indices
    if not selected_indices:
        return
    index = selected_indices[0]
    display_video(index)

def filter_callback():
    selected_cluster = cluster_select.value
    data = source.data
    if selected_cluster == "All":
        new_fill_alpha = [0.7] * len(data['cluster'])
        new_color = data['original_color'][:]
        summary_div.text = "<b>Cluster Summary:</b> All clusters are selected."
    else:
        try:
            selected_cluster_int = int(selected_cluster)
        except ValueError:
            print(f"Invalid cluster selection: {selected_cluster}")
            return

        new_fill_alpha = [
            1.0 if cluster == selected_cluster_int else 0.1
            for cluster in data['cluster']
        ]
        neutral_color = "#d3d3d3"
        new_color = [
            original_color if cluster == selected_cluster_int else neutral_color
            for cluster, original_color in zip(data['cluster'], data['original_color'])
        ]

        selected_texts = df_umap[df_umap['cluster'] == selected_cluster_int]['text'].dropna().tolist()
        summary = summarize_texts(selected_texts)
        summary_div.text = f"<b>Cluster {selected_cluster_int} Summary:</b><br>{summary}"
        print(f"Filter applied: Cluster {selected_cluster_int}")

    source.data['fill_alpha'] = new_fill_alpha
    source.data['color'] = new_color
    video_div.text = "<b>Select a point to see the video, its path, and cluster here</b>"
    print(f"Filter applied: {'All' if selected_cluster == 'All' else selected_cluster}")

def reset_filter():
    cluster_select.value = "All"
    filter_callback()

def generate_query_embedding(query):
    """
    Generate an embedding vector for the given query using the embedding model.
    """
    try:
        embedding_result = embedding_model.get_embeddings(
            contextual_text=query,
            dimension=1408
        )
        text_embedding = embedding_result.text_embedding
        query_embedding = np.array(text_embedding, dtype=np.float32)
        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            return query_embedding
        return query_embedding / norm
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return np.array([])

def select_video(video_idx):
    """
    Select the video point on the plot and display its video.
    """
    source.selected.indices = [video_idx]
    display_video(video_idx)

def search_callback():
    query = search_input.value.strip()
    if not query:
        search_results_div.children = [Div(text="<b>Search Results:</b> Please enter a query.")]
        return

    query_embedding = generate_query_embedding(query)
    if query_embedding.size == 0:
        search_results_div.children = [Div(text="<b>Search Results:</b> Invalid or empty query embedding.")]
        return

    try:
        similarities = np.dot(embeddings_norm, query_embedding)
    except Exception as e:
        print(f"Error computing similarities: {e}")
        search_results_div.children = [Div(text="<b>Search Results:</b> Error computing similarities.")]
        return

    try:
        top5_indices = np.argsort(similarities)[-5:][::-1]
        top5_similarities = similarities[top5_indices]
    except Exception as e:
        print(f"Error retrieving top 5 similar videos: {e}")
        search_results_div.children = [Div(text="<b>Search Results:</b> Error retrieving similar videos.")]
        return

    new_children = [Div(text="<b>Search Results:</b>")]

    for rank, (video_idx, score) in enumerate(zip(top5_indices, top5_similarities), start=1):
        if video_idx >= len(df_umap):
            print(f"Index {video_idx} is out of bounds for df_umap with size {len(df_umap)}.")
            continue
        video_path = df_umap.iloc[video_idx]['video_path']
        text_caption = df_umap.iloc[video_idx]['text']
        cluster_id = df_umap.iloc[video_idx]['cluster']

        button_label = f"Rank {rank}: Cluster {cluster_id}\nSimilarity: {score:.4f}\nPath: {video_path}"
        button = Button(label=button_label, width=800, height=50)
        button.on_click(partial(select_video, video_idx=video_idx))

        new_children.append(button)

    search_results_div.children = new_children

# -----------------------------------
# Assign Callbacks
# -----------------------------------

p.on_event(Tap, tap_callback)
cluster_select.on_change('value', lambda attr, old, new: filter_callback())
reset_button.on_click(reset_filter)
search_button.on_click(search_callback)

# -----------------------------------
# Layout
# -----------------------------------

# Create a layout for cluster filters and summary
filter_layout = row(cluster_select, reset_button)
video_section = column(summary_div, video_div)
search_section = column(search_input, search_button, search_results_div)

# Place cluster selection and summary/video display side by side with search
side_layout = column(filter_layout, video_section, sizing_mode='fixed')
full_layout = row(p, side_layout, Spacer(width=50), search_section)

curdoc().add_root(full_layout)
curdoc().title = "Interactive Video Embeddings with Semantic Search and Cluster Selection"
