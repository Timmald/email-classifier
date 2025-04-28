from sentence_transformers import SentenceTransformer
import csv
from tqdm import tqdm
import pandas as pd

#Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Paths that can be changed
relative_path = "newsgroups_minimal.csv"
csv_file_path = "newsgroups_minimalV.csv"
csv_file_path_indexed = "newsgroups_minimalIndexed.csv"

#storages
column_data = []
vec_storage= []

#converts a sentence to a vector
def email_to_vector_v2(sentence):
    embedding = model.encode(sentence)
    return(embedding)

#splits the file by sentence into a list
def sentences_newsgroups(file_path):
    global column_data
    with open(file_path, newline="") as file:
        reader = csv.reader(file, delimiter="\n")
        column_data = [row[0] for row in reader if row]

#turns whole list into vectors
def convert_toVector():
    vec_storage = email_to_vector_v2(column_data)
    df = pd.DataFrame(vec_storage)

    # Save to CSV
    df.to_csv(csv_file_path, index=False)



def indexed_file():
    df = pd.DataFrame(column_data)
    df.to_csv(csv_file_path_indexed, index=False)




sentences_newsgroups(relative_path)
convert_toVector()
indexed_file()
