import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import csv

with open(r"C:\Users\minhy\OneDrive\Documents\spam-vectors.csv", 'r') as spam:
    print("open csv file")
    csv_reader = csv.reader(spam)
    print("read csv file")
    SpamVectors = []    
    for item in csv_reader: 
        SpamVectors.append(item)

def center(vectors):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(vectors)
    return(kmeans.cluster_centers_)


def label(vectors):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(vectors)
    labels = kmeans.fit_predict(vectors)
    return labels
#test:
# print(center(Spamlist))

labeledVectors = label(SpamVectors)

#fit_predict, category label

with open("datasets\enron_spam\spam.csv", 'r') as spam:
    print("open csv file")
    csv_reader = csv.reader(spam)
    print("read csv file")
    SpamEnron = []
    for item in csv_reader: 
        SpamEnron.append(item)

print(SpamEnron[1])

assert len(labeledVectors) == len(SpamEnron), "Length mismatch"


matched = []
for label, email in zip(labeledVectors, SpamEnron):
    matched.append((label, email[0]))

for label, text in matched[:5]:
    print(f"Label {label}: {text[:100]}...")



print(center(Spamlist))




#scikit train_test_split()
#splits dataset in half, so for training
#two lists of vectors, label everything, tuple method
