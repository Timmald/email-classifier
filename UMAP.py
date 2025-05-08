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