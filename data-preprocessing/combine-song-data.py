import pandas as pd

# Read the CSVs
liked_songs_df = pd.read_csv('liked_songs.csv')
random_songs_df = pd.read_csv('spotify_playlist_features.csv')

# Remove duplicates
liked_songs_df.drop_duplicates(inplace=True)
random_songs_df.drop_duplicates(inplace=True)

# Remove rows with missing information
liked_songs_df.dropna(inplace=True)
random_songs_df.dropna(inplace=True)

# Add 'liked_songs' column
liked_songs_df['liked_songs'] = 1
random_songs_df['liked_songs'] = 0

# Combine both DataFrames
combined_df = pd.concat([liked_songs_df, random_songs_df], ignore_index=True)

# Save the combined DataFrame to a new CSV
combined_df.to_csv('combined_songs.csv', index=False)
