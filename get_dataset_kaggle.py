import os
import kaggle
import zipfile
import pandas as pd

# Kaggle dataset name
DATASET_NAME = "balaka18/email-spam-classification-dataset-csv"

# Define paths
DATA_DIR = "datasets/spamassassin"
ZIP_PATH = os.path.join(DATA_DIR, "spam_dataset.zip")
EXTRACTED_FILE = os.path.join(DATA_DIR, "emails.csv")

# Ensure dataset directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Download dataset from Kaggle
print("Downloading dataset from Kaggle...")
kaggle.api.dataset_download_files(DATASET_NAME, path=DATA_DIR, unzip=True)

# Check if ZIP file exists (in case it's not automatically unzipped)
zip_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".zip")]
if zip_files:
    print(f"Extracting {zip_files[0]}...")
    with zipfile.ZipFile(os.path.join(DATA_DIR, zip_files[0]), "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

# List files in the directory
print("Files in dataset directory:", os.listdir(DATA_DIR))

# Ensure the expected file exists
if not os.path.exists(EXTRACTED_FILE):
    raise FileNotFoundError(f"Expected file '{EXTRACTED_FILE}' not found in {DATA_DIR}")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(EXTRACTED_FILE)

# Display column names
print("Dataset columns:", df.columns)

# Show first few rows to confirm correct data
print(df.head())
