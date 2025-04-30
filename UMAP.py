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
embedding = umap.UMAP(n_components=2, metric='euclidean').fit(vectorsArr)
f = umap.plot.points(embedding)