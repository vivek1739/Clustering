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