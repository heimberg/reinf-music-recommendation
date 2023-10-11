"""
check_data.py
-------------
Module providing utility functions to inspect and manipulate song data.
Includes functionality to check the distribution of liked songs and to discard a fraction of them.
"""

import pandas as pd

def check_liked_songs_distribution(filename):
    """Returns the distribution of values in the 'liked_songs' column."""
    # Load the CSV file
    data = pd.read_csv(filename)
    # Check the distribution of values in the 'liked_songs' column
    distribution = data['liked_songs'].value_counts(normalize=True)
    return distribution

def discard_liked_songs(filename, frac=0.20):
    """Discards a fraction of the liked songs and saves the result in a new CSV."""
    # Load the data
    data = pd.read_csv(filename)
    # Separate the liked songs and other songs
    liked_songs = data[data['liked_songs'] == 1]
    other_songs = data[data['liked_songs'] == 0]
    # Randomly sample a fraction of the liked songs
    sampled_liked_songs = liked_songs.sample(frac=frac)
    # Concatenate the sampled liked songs with the other songs to get the final dataset
    final_data = pd.concat([sampled_liked_songs, other_songs], axis=0).reset_index(drop=True)
    # Save the final dataset to a new CSV file
    final_data.to_csv('modified_' + filename, index=False)

if __name__ == '__main__':
    filename = 'all_songs_compressed_embeddings.csv'
    distribution = check_liked_songs_distribution(filename)
    print(distribution)
    discard_liked_songs(filename)
    distribution = check_liked_songs_distribution('modified_' + filename)
    print(distribution)
