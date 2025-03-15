from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stopwords = stopwords.words('english')

def clean_email(email):#restricts to alphanumeric, removes the many \\n (newlines with an extra escape \)in this dataset, does lowercase
    #PROBLEM: This seems to ignore some character that many spamassasin emails are using as word separators
    #so a lot of wordslooklikethis which will be a problem for BERT
    #also, it's turning some \\t into t, and keeping any genuine \t
    email = email.replace("\\n"," ")
    return re.sub(f'({" | ".join(stopwords)})|[^a-zA-Z0-9 ]',"",email).lower()

def get_dataset_example():
    ds = load_dataset("talby/spamassassin", "text")
    #separate ham from spam
    #different datasets will be structured differently
    hamText = [i[0] for i in zip(ds["train"]["text"],ds["train"]["label"]) if i[1] == 1]
    spamText = [i[0] for i in zip(ds["train"]["text"],ds["train"]["label"]) if i[1] == 0]
    #remove stopwords and nonalphanumeric characters
    hamText = [clean_email(i) for i in hamText]
    spamText = [clean_email(i) for i in spamText]

    with open("datasets/spamassasin/spam.csv","w") as writer:
        writer.writelines("\n".join(spamText))
    with open("datasets/spamassasin/ham.csv","w") as writer:
        writer.writelines("\n".join(hamText))
get_dataset_example()