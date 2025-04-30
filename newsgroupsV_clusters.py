import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import csv

with open(r"C:\Users\minhy\Downloads\newsgroupsV2.csv", 'r') as ham:
    print("open csv file")
    csv_reader = csv.reader(ham)
    print("read csv file")
    HamVectors = []    
    for item in csv_reader: 
        HamVectors.append(item)

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

labeledVectors = label(HamVectors)

#fit_predict, category label

with open(r"C:\Users\minhy\Downloads\newsgroupsIndexed2.csv", 'r') as ham:
    print("open csv file")
    csv_reader = csv.reader(ham)
    print("read csv file")
    HamEnron = []
    for item in csv_reader: 
        HamEnron.append(item)

print(len(HamEnron))
print(len(HamVectors))
print(len(labeledVectors))

assert len(labeledVectors) == len(HamEnron), "Length mismatch"


matched = []
for label, email in zip(labeledVectors, HamEnron):
    matched.append((label, email[0]))

for label, text in matched[:5]:
    print(f"Label {label}: {text[:100]}...")




