import os
import sys
import numpy as np
import tiktoken
import argparse
from openai import OpenAI
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
from tqdm import tqdm

client = OpenAI()
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

def read_file(file_path):
    """Reads the content of a file. Skips files that are not UTF-8 encoded."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        tqdm.write(f"Skipping non-UTF-8 file: {file_path}")
        return None

def split_text_into_chunks(text, max_tokens=8000):
    """Splits the text into chunks, each with a maximum of max_tokens."""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def get_embedding(text, model="text-embedding-3-small"):
    """Gets the embedding for a given text using OpenAI's embedding API."""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def get_file_embedding(text):
    """Splits the text into chunks, computes embeddings for each chunk, and averages them."""
    chunks = split_text_into_chunks(text)
    chunk_embeddings = []

    # Compute embeddings for each chunk
    for chunk in chunks:
        chunk_embedding = get_embedding(chunk)
        chunk_embeddings.append(np.array(chunk_embedding))

    # Average the embeddings of the chunks
    combined_embedding = np.mean(chunk_embeddings, axis=0)
    return combined_embedding

def get_files_in_directory(directory, extensions=None, exclude_dirs=None):
    """Gets all files in a directory that match the given extensions (if any),
       excluding any directories specified in exclude_dirs."""
    exclude_dirs = set(exclude_dirs or [])
    files = []
    for root, dirs, filenames in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for filename in filenames:
            if extensions:
                if any(filename.endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, filename))
            else:
                files.append(os.path.join(root, filename))
    return files

def cosine_similarity(embedding1, embedding2):
    """Computes the cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def cluster_files(files, threshold=0.1):
    """Clusters files based on their embeddings and returns clusters."""
    embeddings = []
    tqdm.write("Computing embeddings for files...")

    # Create a progress bar with a postfix to show dynamic information
    with tqdm(files, desc="Processing files", unit="file") as pbar:
        for file_path in files:
            # Read the file (skip if non-UTF-8)
            text = read_file(file_path)
            if text is None:
                pbar.update(1)
                continue

            # Compute the combined embedding for the file
            combined_embedding = get_file_embedding(text)

            # Update progress bar postfix with the current file
            pbar.set_postfix({
                "file": os.path.basename(file_path)
            })

            embeddings.append(combined_embedding)
            pbar.update(1)

    # Compute the pairwise cosine distance matrix and convert it to similarity
    pairwise_distances = pdist(np.array(embeddings), metric='cosine')
    pairwise_similarity = 1 - squareform(pairwise_distances)  # Convert distance to similarity

    # Perform single linkage clustering
    Z = linkage(pairwise_distances, method='single')

    # Cluster files based on the specified threshold
    clusters = fcluster(Z, t=threshold, criterion='distance')

    # Group files by cluster
    clustered_files = defaultdict(list)
    for i, cluster_id in enumerate(clusters):
        clustered_files[cluster_id].append((files[i], embeddings[i]))

    # Filter out clusters with only one file
    return {cluster_id: file_list for cluster_id, file_list in clustered_files.items() if len(file_list) > 1}, pairwise_similarity


# Argument parsing and program execution starts directly here
parser = argparse.ArgumentParser(description="Cluster files based on text embeddings.")
parser.add_argument("directory", type=str, help="Directory to scan for files.")
parser.add_argument("extensions", nargs="*", help="File extensions to include (e.g. .txt .md).")
parser.add_argument("--exclude-dirs", nargs="*", help="Directories to exclude by name.", default=None)
parser.add_argument("--threshold", type=float, help="Clustering threshold for cosine distance.", default=0.1)

args = parser.parse_args()

# Get all matching files
files = get_files_in_directory(args.directory, args.extensions, args.exclude_dirs)

if not files:
    print("No files found.")
    sys.exit(1)

# Cluster the files based on their embeddings and get similarity matrix
clusters, similarities = cluster_files(files, args.threshold)

# Output clusters with more than one file and the lowest similarity in each cluster
if clusters:
    tqdm.write("Clusters found:")
    for cluster_id, cluster_files in clusters.items():
        tqdm.write(f"\nCluster {cluster_id}:")
        for file, _ in cluster_files:
            tqdm.write(f"  {file}")

        # Compute the lowest similarity within the cluster
        min_similarity = float('inf')
        for i, (_, embedding1) in enumerate(cluster_files):
            for j, (_, embedding2) in enumerate(cluster_files):
                if i < j:  # Only compare each pair once
                    similarity = cosine_similarity(embedding1, embedding2)
                    if similarity < min_similarity:
                        min_similarity = similarity

        tqdm.write(f"  Lowest similarity in Cluster {cluster_id}: {min_similarity:.4f}")
else:
    tqdm.write("No clusters with more than one file found.")
