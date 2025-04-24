import numpy
import math 

#assume you have clusters of emails: 

#given file:
def popular_words(cluster, all, cluster_number):
    #tf[t,c] finding counts in cluster 
    unique_cluster, count_cluster = numpy.unique(cluster, return_counts=True)

    #A finding avergae number of words per cluster 
    unique_all, count_all = numpy.unique(all, return_counts=True)
    A = 0
    for i in len(count_all):
        A += count_all[i]
    A = A/cluster_number

    W_list = []
    
    for word_c_index in len(unique_cluster):
        # tf[t] now find how many times word appears in unique_all
        word_in_all = 0
        for word_all_index in len(unique_all):
            if unique_all[word_all_index] == unique_cluster[word_c_index]:
                word_in_all = count_all[word_all_index]

        #now apply formula: 
        W = count_cluster[word_c_index] * math.log(1+(A/word_in_all))
        W_list.append(W)
