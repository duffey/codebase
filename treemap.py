import os
import argparse
import plotly.express as px
from collections import defaultdict
from tqdm import tqdm  # Import tqdm for the progress bar

# Set up argument parser
parser = argparse.ArgumentParser(description='Visualize a directory of files as a tree map based on file structure.')
parser.add_argument('directory', type=str, help='Directory to visualize')
parser.add_argument('extensions', type=str, nargs='*', help='File extensions to consider (e.g., .py .txt .md). If not provided, all files will be included.')
parser.add_argument('--exclude-dirs', type=str, nargs='*', default=[], help='Directories to exclude from the visualization')

# Parse the arguments
args = parser.parse_args()
directory = os.path.abspath(args.directory)  # Convert to absolute path to handle relative paths
extensions = args.extensions if args.extensions else None  # Set extensions to None if no extensions are provided
exclude_dirs = [os.path.abspath(os.path.join(directory, d)) for d in args.exclude_dirs]  # Convert exclude directories to absolute paths

# Gather the data for the given directory
def gather_file_data(dir_path):
    file_data = []

    # Calculate total number of files to be processed for the progress bar
    file_count = sum(len(files) for _, _, files in os.walk(dir_path))  # Get the number of files

    with tqdm(total=file_count, desc="Processing files") as pbar:  # Create tqdm progress bar
        for root, dirs, files in os.walk(dir_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) not in exclude_dirs]

            for file in files:
                pbar.update(1)  # Increment the progress bar
                filepath = os.path.join(root, file)

                # If extensions are provided, check if the file extension matches the provided extensions
                if extensions and not any(filepath.endswith(ext) for ext in extensions):
                    continue

                # Count lines in the file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        num_lines = sum(1 for _ in f)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue

                # Skip files with zero lines
                if num_lines == 0:
                    continue

                # Add file data
                relative_path = os.path.relpath(filepath, dir_path)
                dir_structure = relative_path.split(os.sep)
                file_data.append((dir_structure, num_lines))

    return file_data

# Gather the data for the given directory
file_data = gather_file_data(directory)

# Prepare the data for Plotly's treemap
ids = ['.']  # Start with root node as '.'
names = ['.']  # Root node name
parents = ['']  # Root node has no parent
values = [0]  # Root node starts with 0 lines
hover_lines = [0]  # Root node hover starts with 0 lines

# Dictionary to accumulate line counts for directories
dir_line_counts = defaultdict(int)

for f in file_data:
    path_parts = f[0]  # Directory structure with file name at the end
    num_lines = f[1]
    
    # Build unique IDs and labels
    for i in range(len(path_parts)):
        current_path = os.path.join('.', *path_parts[:i+1])  # Unique ID (rooted at '.')
        parent_path = os.path.join('.', *path_parts[:i]) if i > 0 else '.'  # Parent ID, root points to itself

        # Extract the name to display
        name = path_parts[i]

        # For files, we always display just the filename
        display_name = name

        if current_path not in ids:
            ids.append(current_path)
            parents.append(parent_path if parent_path else '.')  # Root has no parent
            names.append(display_name)

            if i == len(path_parts) - 1:  # It's a file
                values.append(num_lines)
                hover_lines.append(num_lines)
            else:
                # For directories, initialize hover_lines with 0 (we'll calculate total later)
                values.append(0)
                hover_lines.append(0)

        # Accumulate the line count for directories
        if i < len(path_parts) - 1:  # Only for directories, not the file itself
            dir_path = os.path.join('.', *path_parts[:i+1])
            dir_line_counts[dir_path] += num_lines

# After processing all files, update hover information for directories
for i, id in enumerate(ids):
    if id in dir_line_counts:
        hover_lines[i] = dir_line_counts[id]

# Accumulate the total lines for the root node
total_lines = sum(values)
hover_lines[0] = total_lines  # Update hover for root (index 0)

# Create the treemap with default coloring
fig = px.treemap(
    ids=ids,
    names=names,
    parents=parents,
    values=values,
    custom_data=[hover_lines]  # Pass total lines for hover
)

# Customize text and hover information
fig.data[0].textinfo = 'label'
fig.data[0].hovertemplate = (
    'Lines: %{customdata[0]:,}<extra></extra>'  # Add :, to format numbers with commas
)

fig.update_traces(root_color="lightgrey")
fig.update_layout(
    margin=dict(t=50, l=5, r=5, b=5),
    coloraxis_showscale=False  # Hide the color bar
)

# Show the interactive treemap
fig.show()
