import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# load embeddings
df = pd.read_csv('genre_embeddings.csv')
genres = df['Genre'].values
embeddings = np.array([eval(e) for e in df['Embedding'].values])

# PCA to reduce the embeddings to 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# K-means clustering to group the embeddings into clusters, use value from elbow method
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(embeddings_2d)

# plot the clusters
plt.figure(figsize=(15, 15))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='rainbow')
#for i, genre in enumerate(genres):
#    plt.annotate(genre, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.7)

plt.title('Genres gruppiert nach Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# save the plot
plt.savefig('genre-clusters.png')
plt.show()
