"""
data-preparation.py
-------------------
Module handling the data preparation process for the song data.
Includes tasks like loading genre embeddings, extracting features, scaling features, and genre embedding.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import ast

genre_embeddings = pd.read_csv('../genre_embeddings.csv').set_index('Genre')
genre_embeddings['Embedding'] = genre_embeddings['Embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
genre_embeddings = genre_embeddings.to_dict()['Embedding']

# Load the data
df = pd.read_csv('../combined_songs.csv')

# Extract spotify features
features = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence', 'instrumentalness']

# Make a copy of the dataframe to avoid SettingWithCopyWarning
data_continuous = df[features].copy()

# Scale features
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_continuous), columns=data_continuous.columns)

# Embed genres
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

# Drop the original columns and combine the scaled continuous data with the embeddings
df = df.drop(features + ['genre_embedding'], axis=1)
data_combined = pd.concat([df, data_scaled, embedding_df], axis=1)

# Save the data to a csv
data_combined.to_csv('all_songs_processed.csv', index=False)
