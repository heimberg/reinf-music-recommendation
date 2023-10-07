import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# Authenticate with Spotify API
client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

playlists = ['37i9dQZEVXbLiRSasKsNU9', '37i9dQZF1DXc3KygMa1OE7', '37i9dQZF1DWYtKpmml7moA', '37i9dQZF1DZ06evO1XahPz']

# Loop through each playlist
for playlist_id in playlists:
    results = sp.playlist_tracks(playlist_id)  # Using the playlist_tracks method

    # Extract IDs, Titles, and Artists
    playlist_tracks_data = results['items']
    playlist_tracks_id = [track['track']['id'] for track in playlist_tracks_data]
    playlist_tracks_titles = [track['track']['name'] for track in playlist_tracks_data]
    playlist_tracks_first_artists = [track['track']['artists'][0]['name'] for track in playlist_tracks_data]

    # Extract Audio Features
    features = sp.audio_features(playlist_tracks_id)

    # Create a dataframe with audio features
    features_df = pd.DataFrame(data=features, columns=features[0].keys())
    features_df['album_name'] = ''  # Placeholder
    features_df['track_name'] = playlist_tracks_titles
    features_df['artist_name'] = playlist_tracks_first_artists

    # Extract genre for the primary artist for each track
    genres = []
    for artist in playlist_tracks_first_artists:
        artist_data = sp.search(q='artist:' + artist, type='artist')
        artist_genre = artist_data['artists']['items'][0]['genres']
        genres.append(', '.join(artist_genre))
    features_df['genre'] = genres

    # Rearrange columns and append to CSV
    features_df = features_df[['album_name', 'track_name', 'artist_name', 'genre',
                               'danceability', 'energy', 'speechiness', 
                               'acousticness', 'valence', 'instrumentalness']]
    features_df.to_csv('spotify_playlist_features.csv', mode='a', header=False, index=False)
