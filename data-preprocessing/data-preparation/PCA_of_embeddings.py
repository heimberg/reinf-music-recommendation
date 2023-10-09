import pandas as pd
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv('all_songs_processed.csv')

# Extract embedding columns
embedding_columns = [f'embed_{i}' for i in range(1536)]
embedding_data = df[embedding_columns]

# Apply PCA to reduce the embeddings to 2 dimensions
pca = PCA(n_components=2)
compressed_embeddings = pca.fit_transform(embedding_data)

# Add the PCA results back to the dataframe
df['PCA_1'] = compressed_embeddings[:, 0]
df['PCA_2'] = compressed_embeddings[:, 1]

# Optionally drop the original embedding columns to save space
df = df.drop(columns=embedding_columns)

# Save the resulting dataframe to a new CSV
df.to_csv('all_songs_compressed_embeddings.csv', index=False)
