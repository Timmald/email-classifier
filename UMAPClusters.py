import numpy
import math 
import csv

from tqdm import tqdm
import newsgroupsV_clusters
import pandas as pd
import umap
import umap.plot
import csv

# Some plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import bokeh 
from bokeh.plotting import show, save, output_notebook, output_file
from bokeh.resources import INLINE
from sklearn.cluster import KMeans
import datashader 
import holoviews 
import skimage
import colorcet
output_notebook(resources=INLINE)

#assume you have clusters of emails: 

#given file:
def popular_words(cluster, all, cluster_number):

    cluster = cluster.split()
    all = all.split()


    #tf[t,c] finding counts in cluster 
    unique_cluster, count_cluster = numpy.unique(cluster, return_counts=True)

    #A finding avergae number of words per cluster 
    unique_all, count_all = numpy.unique(all, return_counts=True)
    #make these into dictonary: 
    allDict = dict(zip(unique_all, count_all))

    # make unique_all into array indv words
    print(len(unique_all))
    print(len(unique_cluster))
    print(count_all)

    

    A = 0
    for i in range(len(count_all)):
        A += count_all[i]
    A = A/cluster_number

    W_list = []
    
    for word_c_index in tqdm(range(len(unique_cluster))):
        # tf[t] now find how many times word appears in unique_all
        word_in_all = 0
        if unique_cluster[word_c_index] in allDict:
            word_in_all+= allDict[unique_cluster[word_c_index]]


        # for word_all_index in (range(len(unique_all))):
        #     # print(len(unique_all))
        #     if unique_all[word_all_index] == unique_cluster[word_c_index]:
        #         word_in_all += count_all[word_all_index]

        #now apply formula: 
        W = count_cluster[word_c_index] * math.log(1+(A/word_in_all))

        #if word occurs in all less than 5 times, add 0 to the W score
        if count_cluster[word_c_index]<5:
            W = 0
        W_list.append(W)


    # get top 3 popular words:

    max_1 = numpy.max(W_list)
    max1_index = W_list.index(max_1)
    W_list.remove(max_1)

    max_2 = numpy.max(W_list)
    max2_index = W_list.index(max_2)
    W_list.remove(max_2)


    max_3 = numpy.max(W_list)
    max3_index = W_list.index(max_3)
    W_list.remove(max_3)

    return [unique_cluster[max1_index],unique_cluster[max2_index], unique_cluster[max3_index]]


labeled_clusters = newsgroupsV_clusters.main()

all_text = " ".join(labeled_clusters.values())

cluster_number = len(labeled_clusters)

for label, cluster_text in labeled_clusters.items():
    print(f"\nCluster {label}:")
    top_words = popular_words(cluster_text, all_text, cluster_number)
    print(top_words)











vector_file_path = "newsgroupsV2.csv"
vectorsArr = []

def get_vectors():
    global vectorsArr
    with open(vector_file_path, 'r') as vector:
            csv_reader = csv.reader(vector)
            for item in csv_reader: 
                vectorsArr.append(item)
    print("Read csv file")   

get_vectors()
reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(vectorsArr)

# Save reduced embedding to CSV
df = pd.DataFrame(embedding, columns=['x', 'y'])
df.to_csv('newsgroupsV2REDUCED.csv', index=False)

kmeans = KMeans(7)
cluster_results = kmeans.fit_predict(embedding)
umap.plot.points(reducer, labels=cluster_results)

#must add break point on below line for it work and then run it using python debugger!!
print("hi")