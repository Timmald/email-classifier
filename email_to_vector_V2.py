from sentence_transformers import SentenceTransformer
import csv
from tqdm import tqdm
import pandas as pd

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def email_to_vector_v2(sentences):
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    similarities = model.similarity(embeddings, embeddings)
    return(similarities)


relative_path = "newsgroups.csv"
column_data = []
column_data_2 = []
vec_storage= []

def first_ten_newsgroups(file_path):
    with open(file_path, newline="") as file:
        reader = csv.reader(file, delimiter="\n")
        column_data = [row[0] for row in reader if row]

    column_data_2 = list(tqdm(column_data))

    vec_storage = email_to_vector_v2(column_data_2)


    df = pd.DataFrame(vec_storage)

    # Save to CSV
    csv_file_path = "newsgroupsV.csv"
    df.to_csv(csv_file_path, index=False)

first_ten_newsgroups(relative_path)