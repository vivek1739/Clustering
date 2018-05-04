# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("ProcessedData.csv")
X = df.values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter= 300,n_init=10)
    # n init = num of times k means will be run with diff random Initialization
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    # interia attribute will compute within cluster sum of squares

plt.plot(range(1,11),wcss)
plt.title("The Elbow method")
plt.xlabel('Number of clusters ')
plt.ylabel('wcss')
plt.show()

# We got right num of clusters = 5
# applying k=5
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter= 300,n_init=10)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1], s = 100,label ="C1",c='cyan')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1], s = 100,label ="C2",c='blue')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1], s = 100,label ="C3",c='red')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1], s = 100,label ="C4",c='green')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1], s = 100,label ="C5",c='yellow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='black', label="Clusters")
plt.title("Cluster of clients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()