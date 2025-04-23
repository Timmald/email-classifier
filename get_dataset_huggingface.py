import pandas as pd
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
import re

#stop words
nltk.download("stopwords")
stopwords_list = set(stopwords.words("english"))

#loading newsgroups dataset
ds = load_dataset("SetFit/20_newsgroups")
vals = ds["train"]["text"]

def clean_email(email):
    email = email.replace("\\n"," ")
    email = email.replace("\n"," ")
    email = re.sub(r"[^\w\s@.,']", "", email).lower()
    email = re.sub(r"\s{2,}", "", email).lower()
    email = re.sub(r"\n{2,}", "", email).lower()
    words = email.split()
    filtered_email = " ".join(word for word in words if word not in stopwords_list)
    return filtered_email

df = pd.DataFrame(vals)
csv_file_path = 'newsgroups.csv'
with open(csv_file_path, mode='w') as file:
    for row in vals:
        cleaned_row = clean_email(row)
        file.write(cleaned_row + '\n')
    


