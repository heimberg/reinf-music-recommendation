import pandas as pd
from sklearn.preprocessing import StandardScaler
import ast

genre_embeddings = pd.read_csv('../genre_embeddings.csv').set_index('Genre')
genre_embeddings['Embedding'] = genre_embeddings['Embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
genre_embeddings = genre_embeddings.to_dict()['Embedding']

# Load the data
df = pd.read_csv('../spotify_data.csv')

# Extract spotify features
features = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence', 'instrumentalness']
data_continuous = df[features]

# Scale features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_continuous)


def get_embedding_for_genre(genre_str):
    if not isinstance(genre_str, str):
        return [0] * len(genre_embeddings[list(genre_embeddings.keys())[0]])
    
    genres = [g.strip() for g in genre_str.split(',')]
    embeddings = [genre_embeddings.get(g, [0]*len(genre_embeddings[list(genre_embeddings.keys())[0]])) for g in genres]
    
    # Average the embeddings if there's more than one
    avg_embedding = [sum(col)/len(col) for col in zip(*embeddings)]
    
    return avg_embedding


df['genre_embedding'] = df['genre'].apply(get_embedding_for_genre)

# Convert the embedding list to individual columns
embedding_df = pd.DataFrame(df['genre_embedding'].tolist(), columns=[f'embed_{i}' for i in range(len(df['genre_embedding'].iloc[0]))])

# Combine the scaled continuous data with the embeddings
data_combined = pd.concat([df, embedding_df], axis=1)

# save the data to a csv
data_combined.to_csv('spotify_data_processed.csv', index=False)
