# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:16:34 2025

@author: sarah
"""
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load the Data
column_names = [
    "Pelvic Incidence", "Pelvic Tilt", "Lumbar Lordosis Angle", 
    "Sacral Slope", "Pelvic Radius", "Grade of Spondylolisthesis", "Class"
]

df = pd.read_csv(r'C:\Users\sarah\OneDrive\Documents\Msc\stats and ML 2\coursework\data.txt', 
                 delim_whitespace=True, 
                 header=None, 
                 names=column_names)

# Standardise the Data
feature_columns = [
    "Pelvic Incidence", "Pelvic Tilt", "Lumbar Lordosis Angle", 
    "Sacral Slope", "Pelvic Radius", "Grade of Spondylolisthesis"
]

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_columns]), columns=feature_columns)

# Determine Optimal K using Elbow Method for K-Means
kMin = 1
kMax = 8
repsPerK = 10

kValues = np.arange(kMin, kMax + 1)
SumOfSquares = np.zeros(len(kValues))

for i, k in enumerate(kValues):
    kmeans = KMeans(n_clusters=k, n_init=repsPerK, random_state=42)
    kmeans.fit(df_scaled)
    SumOfSquares[i] = kmeans.inertia_

# Plotting the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(kValues, SumOfSquares, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared distances (Inertia)')
plt.title('Elbow Method for Optimal k (K-Means)')
plt.grid(True)
plt.show()

# Elbow plot suggests that otimal K is between 2-3
# Applying Naive K-Means with Optimal K
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, n_init=repsPerK, random_state=42)
df_scaled['KMeans Cluster'] = kmeans.fit_predict(df_scaled[feature_columns])

# Visualising K-Means Clustering
sns.pairplot(df_scaled, hue='KMeans Cluster', diag_kind='kde')
plt.suptitle('Naive K-Means Clustering Results', y=1.02)
plt.show()

# Hierarchical Clustering (Dendrogram)
linked = linkage(df_scaled[feature_columns], method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index or cluster size')
plt.ylabel('Distance')
plt.show()

# Applying Agglomerative Clustering
hierarchical_model = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
df_scaled['Hierarchical Cluster'] = hierarchical_model.fit_predict(df_scaled[feature_columns])

# Visualising Hierarchical Clusters
sns.pairplot(df_scaled, hue='Hierarchical Cluster', diag_kind='kde')
plt.suptitle('Hierarchical Clustering Results', y=1.02)
plt.show()

# Model-Based Clustering (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
df_scaled['GMM Cluster'] = gmm.fit_predict(df_scaled[feature_columns])

# Visualising GMM Clustering
sns.pairplot(df_scaled, hue='GMM Cluster', diag_kind='kde')
plt.suptitle('Model-Based Clustering (GMM) Results', y=1.02)
plt.show()

# Silhouette Scores to Evaluate Cluster Quality
for method in ['KMeans Cluster', 'Hierarchical Cluster', 'GMM Cluster']:
    score = silhouette_score(df_scaled[feature_columns], df_scaled[method])
    print(f'Silhouette Score for {method}: {score:.4f}')

#Experimenting with k values
for k in range(2, 6):
    gmm = GaussianMixture(n_components=k, random_state=42)
    cluster_labels = gmm.fit_predict(df_scaled[feature_columns])
    score = silhouette_score(df_scaled[feature_columns], cluster_labels)
    print(f'Silhouette Score for GMM with {k} clusters: {score:.4f}')

