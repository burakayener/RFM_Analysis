import pickle
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_prep import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from eda import *
import seaborn as sns


def pickle_load(dataframe):
    rfm_data  = pickle.load(open(dataframe, "rb"))
    print(rfm_data.head(3))
    return rfm_data


def check_all_columns(dataframe):
    for col in dataframe.columns:
        if col == "CUSTOMERID":
            continue
        print("Is {} have outliers ? {}".format(col, check_outlier(dataframe, col,q1=.05, q3=.95)))    


def print_about_columns(dataframe):
    for col in dataframe.columns:
        if col == "CUSTOMERID":
            continue
        print("Max value of {} is {}".format(col,dataframe[col].max()))
        print("Min value of {} is {}".format(col,dataframe[col].min()))
        print("Mean value of {} is {}".format(col,dataframe[col].mean()))
        if col == "Frequency" or col == "Monetary":
            low_limit, up_limit = outlier_thresholds(dataframe, col, q1=.05, q3=.95)
            print("Low limit is: {}, Up limit is {}".format(low_limit, up_limit))
        print("#####################################")

        
def  multiple_replacement_with_thresholds(dataframe):
    for col in dataframe.columns:
        if check_outlier(dataframe, col, q1=0.05, q3=0.95) == True:
            replace_with_thresholds(dataframe, col, q1=0.05, q3=0.95)


def drop_index(dataframe, col):
    drop_col = dataframe[col]
    dataframe.drop(col,axis=1, inplace=True)
    return drop_col


def call_scalers(standard=False, min_max=False):
    if standard==True:
        standard_scaler = StandardScaler()
        return standard_scaler
    elif min_max==True:
        min_max_scaler = MinMaxScaler((0, 1))
        return min_max_scaler
    

def scaled_data(dataframe, scaler):
    scaled_data = scaler.fit_transform(dataframe)
    new_name_df = pd.DataFrame(scaled_data)
    new_name_df.columns = ["Recency","Frequency","Monetary"]
    return new_name_df


def best_k(dataframe, show=False):
    km = KMeans()
    best_value = KElbowVisualizer(km, k=(2,20))
    best_value.fit(dataframe)
    best_value.show()
    print("Optimal number of cluster is {}".format(best_value.elbow_value_))
    print("--------------------------------------------------------------------------------")
    return best_value.elbow_value_


def kmean_model(dataframe ,n_number, verbose=0):
    model = KMeans(n_clusters=n_number, n_init=10,
    max_iter=500, tol=0.0001,verbose=verbose).fit(X=dataframe)
    dataframe["cluster_no"] = model.labels_ + 1
    return dataframe

def report_data(compare_df_1, compare_df_2, opt=False):
    if opt==True:
        return compare_df_1.groupby("cluster_no").agg(['min','max','mean','count'])
        print("############################################################################")
    else:
        compare_df_2.columns = ["REAL_"+ col.upper() for col in compare_df_2.columns]        
        compare_df = pd.concat([compare_df_1, compare_df_2], axis=1)
        print("Variables starts with 'REAL' is from original rfm data values.")
        print("############################################################################")
        return compare_df.groupby("cluster_no").agg(['min','max','mean','count'])


def comments():
    print("For unscaled data we have 5 clusters.")
    print("Cluster1: Second worst cluster. %21.04 of the customers in this cluster.Low monetary, frequency and moderately high recency. ")
    print("Cluster2: Second best cluster after Cluster1. Contains %2.86 customers ")
    print("Cluster3: Third best cluster after  Cluster2. Contains %8.11")
    print("Cluster4: Worst cluster. Lowest monetary, frequency and highest recency. %65.83 customers in this cluster ")
    print("Cluster5: Best cluster. Highest monetary and frequency and lowest recency. Contains %2.15 of the customers")
    print("KMeans works with distance, so for unscaled data KMeans clustring is not optimal")
    print("###################################################################################")
    print("For standard scaled data we have 6  clusters")
    print("Cluster1: Second best monetary. And have the lowest recency and highest frequency. Values of frequency is much higher than the others and the recency is the way to low than the other clusters. Contains %2.55 of the customers")
    print("Cluster2: Moderate monetary, quite low recency and high frequency. %15.07 customers in this group.")
    print("Cluster3: Worst cluster. Lowest monetary, frequency and highest frequency. This cluster may contains the customers that only 1 or 2 time bought items in this company %13.87 ")
    print("Cluster4: Low monetary, frequency and recency. This cluster may contain most of the new customers. This is why all variables are low. Half of the customers in this group %49.44")
    print("Cluster5: Has the highest monetary and have second bests of recency and frequency. %2.18 of the customers in this cluster. ")
    print("Cluster6: This group is likely the cluster 3. Only diffrence is they spend more money. They may have bought more valuable items. %16.85 of the customers in this cluster.")
    print("###################################################################################")
    print("For min_max scaled data we have 6  clusters")
    print("Cluster1: Worst cluster. Lowest monetary, frequency and highest frequency. This cluster may contains the customers that only 1 or 2 time bought items in this company %13.92")
    print("Cluster2: Low monetary, frequency and recency. This cluster may contain most of the new customers. This is why all variables are low. Half of the customers in this group %46.21")
    print("Cluster3: Best monetary value, second best frequency and recency. This cluster contains most of the most-valuable customers %2.65")
    print("Cluster4: This group is likely the cluster 1. Only diffrence is they spend more money. They may have bought more valuable items. %17.05 of the customers in this cluster")
    print("Cluster5: Have the lowest recency, highest frequency and the second best monetary. This cluster contains most-valuable customers too %3.66")
    print("Cluster6: Moderately high monetary and frequency, low recency. Contains the %16.48 customers.")
    print("###################################################################################")






































