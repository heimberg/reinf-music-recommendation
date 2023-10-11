import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

"""
elbow-method.py
---------------
Module for determining the optimal number of clusters for k-means clustering using the elbow method.
Visualizes the results using a plot, helping in identifying the point where adding more clusters doesn't significantly reduce the sum of squared distances.
"""

# Load the embeddings
df = pd.read_csv('../genre_embeddings.csv')
embeddings = np.array([eval(e) for e in df['Embedding'].values])

# Apply PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Determine the optimal number of clusters
sum_of_squared_distances = []
K = range(1, 30)  # Check for up to 30 clusters, adjust as needed
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(embeddings_2d)
    sum_of_squared_distances.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 7))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k (Anzahl Cluster)')
plt.ylabel('Summe  der quadrierten Abstände')
plt.title('Elbow Methode für die Bestimmung der optimalen Anzahl an Clustern')
plt.savefig('elbow-method.png')
plt.show()

# save the plot
