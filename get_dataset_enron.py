import os
import pandas as pd
import datasets
import nltk
import re
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")
stopwords_list = set(stopwords.words("english"))

# Define storage paths
DATA_DIR = "datasets/enron_spam"
SPAM_FILE = os.path.join(DATA_DIR, "spam.csv")
HAM_FILE = os.path.join(DATA_DIR, "ham.csv")

# Ensure dataset directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load dataset from Hugging Face
print("Downloading dataset from Hugging Face...")
dataset = datasets.load_dataset("SetFit/enron_spam", split="train")

# Check dataset columns
print("Dataset columns:", dataset.column_names)

# Function to clean email text
def clean_email(email):
    """Removes stopwords, newlines, and non-alphanumeric characters, and lowercases the text."""
    email = re.sub("\\n", " ", email).lower()
    email = re.sub(r"[^a-zA-Z0-9 ]", "", email).lower()
    words = email.split()
    filtered_email = " ".join(word for word in words if word not in stopwords_list)
    return filtered_email

# Process dataset
spam_emails = []
ham_emails = []

for row in dataset:
    cleaned_text = clean_email(row["text"])
    if row["label"] == 1:
        spam_emails.append(cleaned_text)
    else:
        ham_emails.append(cleaned_text)

# Save cleaned emails
with open(SPAM_FILE, "w", encoding="utf-8") as f:
    f.writelines("\n".join(spam_emails))

with open(HAM_FILE, "w", encoding="utf-8") as f:
    f.writelines("\n".join(ham_emails))

print(f"Processing complete! Cleaned spam emails saved in '{SPAM_FILE}', ham emails in '{HAM_FILE}'.")
