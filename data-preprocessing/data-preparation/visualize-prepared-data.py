import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



# Load the data
df = pd.read_csv('spotify_data_processed.csv')

# Determine the number of embedding columns
embedding_columns = [col for col in df.columns if col.startswith('embed_')]
num_embeddings = len(embedding_columns)

# Extract features for PCA
features_for_pca = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence', 'instrumentalness'] + [f'embed_{i}' for i in range(num_embeddings)]

data_for_pca = df[features_for_pca]

# Apply PCA and get the first two principal components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_for_pca)

# Create a DataFrame with the principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plot the 2D data
plt.figure(figsize=(10,10))
plt.scatter(pc_df['PC1'], pc_df['PC2'], alpha=0.5)
plt.title('2D PCA of Songs Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
# save the plot
plt.savefig('pca.png')
plt.show()

# cluster the data
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(principal_components)

# Plot the 2D PCA data with clusters color-coded
plt.figure(figsize=(10,10))
for i in range(n_clusters):
    plt.scatter(principal_components[clusters == i, 0], 
                principal_components[clusters == i, 1], 
                label=f'Cluster {i+1}', alpha=0.6)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Cluster Centers')
plt.title('2D PCA of Songs Data with KMeans Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# analyze the clusters
# Add cluster labels to the original DataFrame
df['cluster'] = clusters

# Display the number of songs in each cluster
print(df['cluster'].value_counts())

# Sample and display a few songs from each cluster
for i in range(n_clusters):
    print(f"\nCluster {i+1} Sample:")
    print(df[df['cluster'] == i].sample(5)[['album_name', 'track_name', 'artist_name', 'genre']])

# Define the list of features
features = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence', 'instrumentalness']
# Create cluster profiles
cluster_profiles = df.groupby('cluster')[features].mean()

print("\nCluster Profiles:")
print(cluster_profiles)

# plot datapoints from one specific artist
# Create a boolean mask for songs by the artist "Archive"
archive_mask = df['artist_name'] == 'Archive'

# Plot the 2D PCA data with clusters color-coded
plt.figure(figsize=(10,10))

# Plot the songs not by "Archive"
for i in range(n_clusters):
    plt.scatter(principal_components[(clusters == i) & ~archive_mask, 0], 
                principal_components[(clusters == i) & ~archive_mask, 1], 
                label=f'Cluster {i+1}', alpha=0.6)

# Plot the songs by "Archive" with a star marker
plt.scatter(principal_components[archive_mask, 0], 
            principal_components[archive_mask, 1], 
            color='black', s=100, label='Archive Songs', marker='*')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Cluster Centers')
plt.title('2D PCA of Songs Data with KMeans Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
