import os
import git
import argparse
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import re
import sys

def tokenize_line(line):
    """
    Tokenize the given line into words, operators, and other meaningful elements.
    """
    return re.findall(r"\w+|\S", line)

def unique_token_density(tokens):
    """
    Calculate the unique token density as the ratio of unique tokens to total tokens.
    """
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0
    unique_tokens = len(set(tokens))
    return unique_tokens / total_tokens

def token_character_ratio(tokens, line_length):
    """
    Calculate the ratio of token count to the length of the line (characters).
    """
    if line_length == 0:
        return 0
    return len(tokens) / line_length

def classify_line(line, threshold_density=0.5, threshold_ratio=0.2):
    """
    Classify the line as 'non-code', 'boilerplate', or 'non-boilerplate'.
    """
    stripped = line.strip()

    # Common non-code patterns across languages
    if stripped == "":
        return "non-code"  # Empty lines
    if stripped in {";", "{", "}"}:  # Structural braces or semicolons
        return "non-code"
    if stripped.startswith(("//", "#")):  # Single-line comments (C-style and Python-style)
        return "non-code"
    if stripped.startswith(("/*", "*/")) or stripped.endswith(("*/", "/*")):  # Multi-line comment blocks
        return "non-code"

    # Token-based analysis
    tokens = tokenize_line(line)
    line_length = len(stripped)

    if tokens:
        density = unique_token_density(tokens)
        ratio = token_character_ratio(tokens, line_length)

        # If token density or token/character ratio is too low, consider it boilerplate
        if density <= threshold_density or ratio <= threshold_ratio:
            return "boilerplate"

    return "non-boilerplate"

# Set up argument parser
parser = argparse.ArgumentParser(description='Analyze lines of code in a Git repository over time.')
parser.add_argument('repo_path', type=str, help='Path to the Git repository')
parser.add_argument('branch', type=str, help='Branch to analyze')
parser.add_argument('extensions', type=str, nargs='+', help='File extensions to consider (e.g., .py .txt .md). At least one extension is required.')
parser.add_argument('-n', '--include-non-code', action='store_true', help='Include non-code lines in the graph')
parser.add_argument('-m', '--monthly', action='store_true', help='Show monthly data instead of yearly')
parser.add_argument('--exclude-dirs', nargs='*', default=[], help='Directory names to exclude from analysis')

# Parse the arguments
args = parser.parse_args()
repo_path = os.path.abspath(args.repo_path)  # Convert to absolute path to handle relative paths
branch = args.branch
extensions = args.extensions  # At least one extension is required due to nargs='+' 
include_non_code = args.include_non_code  # Boolean flag to include non-code lines
monthly = args.monthly  # Boolean flag to show monthly data
exclude_dirs = args.exclude_dirs  # List of directory names to exclude

# Set the time format and interval based on the monthly flag
if monthly:
    time_format = "%Y-%m"
    min_time_difference = relativedelta(months=1)
else:
    time_format = "%Y"
    min_time_difference = relativedelta(years=1)

# Try to open the git repository
try:
    repo = git.Repo(repo_path)
except git.exc.InvalidGitRepositoryError:
    print(f"{repo_path} is not a valid git repository.")
    sys.exit(1)

# Checkout the specified branch
try:
    repo.git.checkout(branch)
except git.exc.GitCommandError:
    print(f"Branch '{branch}' does not exist in this repository.")
    sys.exit(1)

dates = []
non_code_counts = defaultdict(list)
boilerplate_counts = defaultdict(list)
non_boilerplate_counts = defaultdict(list)

current_time = datetime.now()  # Get the current time for comparison
commits = list(repo.iter_commits(branch, reverse=True))  # Get all commits from the branch, ordered by commit date (descending)

# Initialize the progress bar with tqdm
with tqdm(total=len(commits), desc=f"Processing commits on branch {branch}") as pbar:
    last_checked_date = None

    for commit in commits:
        commit_date = datetime.fromtimestamp(commit.committed_date)

        if commit_date > current_time:
            pbar.set_postfix_str(f"Skipping future commit: {commit_date.date()}")
            pbar.update(1)
            continue

        if last_checked_date is not None and commit_date < last_checked_date + min_time_difference:
            pbar.update(1)
            continue

        last_checked_date = commit_date
        pbar.set_postfix_str(f"Processing commit {commit.hexsha[:7]} on {commit_date.date()}")
        repo.git.checkout(commit)

        non_code_by_extension = defaultdict(int)
        boilerplate_by_extension = defaultdict(int)
        non_boilerplate_by_extension = defaultdict(int)

        for root, dirs, files in os.walk(repo.working_tree_dir):
            # Exclude specified directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                ext = os.path.splitext(file)[1]  # Get the file extension
                if ext not in extensions:
                    continue  # Skip files that don't match any of the extensions

                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            classification = classify_line(line)
                            if classification == "non-code":
                                non_code_by_extension[ext] += 1
                            elif classification == "boilerplate":
                                boilerplate_by_extension[ext] += 1
                            else:
                                non_boilerplate_by_extension[ext] += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue  # Skip files that cause an error

        # Append to dates and counts
        dates.append(commit_date.strftime(time_format))

        for ext in extensions:
            non_code_counts[ext].append(non_code_by_extension.get(ext, 0))
            boilerplate_counts[ext].append(boilerplate_by_extension.get(ext, 0))
            non_boilerplate_counts[ext].append(non_boilerplate_by_extension.get(ext, 0))

        pbar.update(1)

# Prepare data for plotting
df = pd.DataFrame({'Date': dates})

# Add boilerplate and non-boilerplate counts for each extension
for ext in extensions:
    df[f'Boilerplate LOC for {ext}'] = boilerplate_counts[ext]
    df[f'Non-Boilerplate LOC for {ext}'] = non_boilerplate_counts[ext]

# Optionally add non-code lines if the flag is set
if include_non_code:
    for ext in extensions:
        df[f'Non-Code LOC for {ext}'] = non_code_counts[ext]

# Aggregate data by year or month
df = df.groupby('Date').sum().reset_index()

# Create the value_vars list based on whether non-code should be included
value_vars = [f'Boilerplate LOC for {ext}' for ext in extensions] + [f'Non-Boilerplate LOC for {ext}' for ext in extensions]

if include_non_code:
    value_vars += [f'Non-Code LOC for {ext}' for ext in extensions]

# Melt the DataFrame for easier plotting with Plotly Express
df_melted = pd.melt(df, id_vars=['Date'], value_vars=value_vars,
                    var_name='File Type and LOC Type', value_name='Lines of Code')

# Plot the stacked area chart
fig = px.area(df_melted, x='Date', y='Lines of Code', color='File Type and LOC Type',
              title=f'LOC Breakdown on Branch {branch}')

# Adjust tick format based on monthly/yearly
fig.update_xaxes(tickformat=time_format)
fig.show()

# Checkout back to the specified branch
repo.git.checkout(branch)
