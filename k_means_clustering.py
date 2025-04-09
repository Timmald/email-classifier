import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import csv

with open('spamV.csv', 'r') as spam:
    csv_reader = csv.reader(spam)
    Spamlist = []    
    for item in csv_reader:
        Spamlist.append(item)

def center(vectors):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(vectors)
    return(kmeans.cluster_centers_)

#test:
print(center(Spamlist))