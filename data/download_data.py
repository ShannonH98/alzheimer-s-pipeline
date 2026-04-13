import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("uraninjo/augmented-alzheimer-mri-dataset")

print("Downloaded to:", path)

# Define your project data path
target_path = "data/raw"

# Create folder if it doesn't exist
os.makedirs(target_path, exist_ok=True)

# Copy dataset into your project
shutil.copytree(path, target_path, dirs_exist_ok=True)

print("Dataset copied to:", target_path)