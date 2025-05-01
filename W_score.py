import numpy
import math 
import csv
from tqdm import tqdm

indexed_file_path = "newsgroupsIndexed2.csv"

#given file:
def popular_words(cluster, all, cluster_number):

    cluster = cluster.split()
    all = all.split()


    #tf[t,c] finding counts in cluster 
    unique_cluster, count_cluster = numpy.unique(cluster, return_counts=True)

    #A finding avergae number of words per cluster 
    unique_all, count_all = numpy.unique(all, return_counts=True)
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
        for word_all_index in (range(len(unique_all))):
            # print(len(unique_all))
            if unique_all[word_all_index] == unique_cluster[word_c_index]:
                word_in_all += count_all[word_all_index]

        #now apply formula: 
        W = count_cluster[word_c_index] * math.log(1+(A/word_in_all))
        W_list.append(W)


    # get top 3 popular words:
    # W_list = numpy.array(W_list)

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

    # need to add in return most popular score


#Testing: 
# Load email from CSV file
with open(indexed_file_path, 'r') as email_file:
    csv_reader = csv.reader(email_file)
    emailArr: list[str] = []

    for item in csv_reader:
        emailArr.append(item[0])  
    
    # create 2 clusters
    quarter = len(emailArr) // 100
    cluster_1 = ' '.join(emailArr[:quarter])
    cluster_2 = ' '.join(emailArr[quarter:2*quarter])  

    all = cluster_1 + cluster_2 
    W_list = popular_words(cluster_1, all, 2)
    print(W_list)


