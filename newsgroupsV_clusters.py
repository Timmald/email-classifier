import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import csv

vector_file_path = "newsgroupsV2.csv"
indexed_file_path = "newsgroupsIndexed2.csv"

#(UNUSED) find cluster centers using KMeans with 2 clusters
def center(vectors):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(vectors)
    return(kmeans.cluster_centers_)

# label data points using KMeans with 3 clusters
def label(vectors):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(vectors)
    labels = kmeans.fit_predict(vectors)
    return labels

def main():
    # Load vector data from a CSV file
    with open(vector_file_path, 'r') as vector:
        print("[vectors] open csv file")
        csv_reader = csv.reader(vector)
        print("read csv file")
        vectorsArr = []    
        for item in csv_reader: 
            vectorsArr.append(item)

    #Apply the clustering to the vector data
    global label
    labeledVectors = label(vectorsArr)

    #fit_predict, category label
    # Load email from CSV file
    with open(indexed_file_path, 'r') as email:
        print("[emails] open csv file")
        csv_reader = csv.reader(email)
        print("read csv file")
        emailArr = []
        for item in csv_reader: 
            emailArr.append(item)

    # Ensure the number of labels matches the number of email texts
    assert len(labeledVectors) == len(emailArr), "Length mismatch"

    # Match each email with its corresponding cluster label
    matched = []
    for label, email in zip(labeledVectors, emailArr):
        matched.append((label, email[0]))

    for label, text in matched:
        print(f"Label {label}: {text[:100]}...")



main()
