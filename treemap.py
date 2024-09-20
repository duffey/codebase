import os
import argparse
import plotly.express as px
from collections import defaultdict
from tqdm import tqdm  # Import tqdm for the progress bar

# Define a tree node to represent files and directories
class TreeNode:
    def __init__(self, name, is_dir=True):
        self.name = name
        self.is_dir = is_dir
        self.children = []
        self.num_lines = 0  # Number of lines for files, 0 for directories

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_empty(self):
        # A directory is empty if it contains no files and no non-empty directories
        return all(child.is_empty() for child in self.children) if self.is_dir else False

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

# Function to build the tree of files and directories
def build_file_tree(dir_path):
    root = TreeNode(name=".", is_dir=True)  # Set root directory name to "."

    # Calculate total number of files to be processed for the progress bar
    file_count = sum(len(files) for _, _, files in os.walk(dir_path))  # Get the number of files

    with tqdm(total=file_count, desc="Processing files") as pbar:  # Create tqdm progress bar
        for root_dir, dirs, files in os.walk(dir_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root_dir, d)) not in exclude_dirs]

            # Construct the relative path from the root directory
            relative_root_dir = os.path.relpath(root_dir, dir_path)
            if relative_root_dir == ".":
                relative_root_dir = ""  # Treat root directory as empty

            current_node = get_or_create_node(root, relative_root_dir.split(os.sep))

            # Add files to the current node
            for file in files:
                pbar.update(1)  # Increment the progress bar
                filepath = os.path.join(root_dir, file)

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

                # Add file node to the current directory node
                file_node = TreeNode(name=file, is_dir=False)
                file_node.num_lines = num_lines
                current_node.add_child(file_node)

    return root

# Helper function to get or create a node in the tree based on the path parts
def get_or_create_node(root, path_parts):
    current_node = root
    for part in path_parts:
        if not part:  # Skip empty part (root directory)
            continue
        # Find if part exists as a child
        matching_child = next((child for child in current_node.children if child.name == part and child.is_dir), None)
        if not matching_child:
            matching_child = TreeNode(name=part, is_dir=True)
            current_node.add_child(matching_child)
        current_node = matching_child
    return current_node

# Function to prune empty directories from the tree
def prune_empty_dirs(node):
    if node.is_dir:
        # Recursively prune children
        node.children = [child for child in node.children if not child.is_empty()]
        for child in node.children:
            prune_empty_dirs(child)

# Function to collapse directories with only one child directory
def collapse_single_child_dirs(node):
    if node.is_dir:
        while len(node.children) == 1 and node.children[0].is_dir:
            # Collapse this directory with its only child
            child = node.children[0]
            node.name = os.path.join(node.name, child.name)  # Concatenate directory names
            node.children = child.children  # Inherit the child's children

        # Recursively collapse for all children
        for child in node.children:
            collapse_single_child_dirs(child)

# Function to traverse the tree and build the treemap data
def traverse_tree(node, parent_id, ids, names, parents, values, hover_lines, file_names):
    current_id = os.path.join(parent_id, node.name).replace('\\', '/')

    ids.append(current_id)
    names.append(node.name)
    parents.append(parent_id)

    if node.is_dir:
        values.append(0)  # Directory gets a value of 0, we'll calculate line count later
        hover_lines.append(0)  # Directory hover lines start with 0
        file_names.append(node.name)  # Just the directory name
        # Traverse children
        for child in node.children:
            traverse_tree(child, current_id, ids, names, parents, values, hover_lines, file_names)
    else:
        values.append(node.num_lines)
        hover_lines.append(node.num_lines)
        file_names.append(node.name)  # Just the file name

# Gather the file tree for the given directory
file_tree = build_file_tree(directory)

# Prune empty directories from the tree
prune_empty_dirs(file_tree)

# Collapse directories with only one child
collapse_single_child_dirs(file_tree)

# Prepare the data for Plotly's treemap
ids = []
names = []
parents = []
values = []
hover_lines = []
file_names = []  # Store just the file names for hover information

# Traverse the tree and fill in the treemap data
traverse_tree(file_tree, '', ids, names, parents, values, hover_lines, file_names)

# Update hover lines for directories by summing child line counts
for i, value in enumerate(values):
    if value == 0:  # Only for directories
        dir_id = ids[i]
        total_lines = sum(hover_lines[j] for j in range(len(ids)) if ids[j].startswith(dir_id + '/'))
        hover_lines[i] = total_lines

# Create the treemap with default coloring
fig = px.treemap(
    ids=ids,
    names=names,
    parents=parents,
    values=values,
    custom_data=[hover_lines, file_names]  # Pass total lines and file names for hover
)

# Customize text and hover information
fig.data[0].textinfo = 'label'
fig.data[0].hovertemplate = (
    'Name: %{customdata[1]}<br>'  # Add file or directory name to hover
    'Lines: %{customdata[0]:,}<extra></extra>'  # Add :, to format numbers with commas
)

fig.update_traces(root_color="lightgrey")
fig.update_layout(
    margin=dict(t=50, l=5, r=5, b=5),
    coloraxis_showscale=False  # Hide the color bar
)

# Show the interactive treemap
fig.show()
