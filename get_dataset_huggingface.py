import pandas as pd
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
import re

#sets stop words
nltk.download("stopwords")
stopwords_list = set(stopwords.words("english"))

#loading newsgroups dataset, takes the text from the TRAIN section of the dataset (did not use the test section) 
ds = load_dataset("SetFit/20_newsgroups")
vals = ds["train"]["text"]

#cleans the email(single line) for new lines, random symbols, stop words,
def clean_email(email):
    email = email.replace("\\n"," ")
    email = email.replace("\n"," ")
    email = re.sub(r"[^\w\s@.,']", "", email).lower()
    email = re.sub(r"\s{2,}", "", email).lower()
    email = re.sub(r"\n{2,}", "", email).lower()
    words = email.split()
    filtered_email = " ".join(word for word in words if word not in stopwords_list)
    return filtered_email

#cleans email but no stop words removed this time
def clean_email_noStopWords_removed(email):
    email = email.replace("\\n"," ")
    email = email.replace("\n"," ")
    email = re.sub(r"[^\w\s@.,']", "", email).lower()
    email = re.sub(r"\s{2,}", "", email).lower()
    email = re.sub(r"\n{2,}", "", email).lower()
    return email

#Cleans the emails as minimally as possible
def clean_email_minimal(email):
    email = email.replace("\n"," ")
    email = email.replace("\\n"," ")
    email = re.sub(r"\s{2,}", "", email).lower()
    email = re.sub(r"\n{2,}", "", email).lower()
    return(email)


#main, creates a df of vals, creates an empty csv to store the sentences after cleaning.
#cleans each sentence, and adds it to the csv file
def main():
    df = pd.DataFrame(vals)

    #uses clean_email, which cleans as much as possible
    csv_file_path = 'newsgroups.csv'
    with open(csv_file_path, mode='w') as file:
        for row in vals:
            cleaned_row = clean_email(row)
            file.write(cleaned_row + '\n')

    #stop words kept
    csv_file_path = 'newsgroups_stopwords_kept.csv'
    with open(csv_file_path, mode='w') as file:
        for row in vals:
            cleaned_row = clean_email_noStopWords_removed(row)
            file.write(cleaned_row + '\n')
    
    #minimal cleaning
    csv_file_path = 'newsgroups_minimal.csv'
    with open(csv_file_path, mode='w') as file:
        for row in vals:
            cleaned_row = clean_email_minimal(row)
            file.write(cleaned_row + '\n')

main()


