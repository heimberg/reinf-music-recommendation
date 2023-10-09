import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os


client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# app credentials from spotify dashboard
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id, client_secret, redirect_uri="http://localhost:8086", scope="user-library-read"))

# function to fetch liked songs from spotify
def fetch_liked_songs_data():
    """Fetch specified data for tracks in the user's Liked Songs."""
    results = sp.current_user_saved_tracks()
    tracks = results['items']
    
    song_data = []
    while results:
        for item in tracks:
            track_obj = item['track']
            track_name = track_obj['name']
            track_id = track_obj['id']
            album_name = track_obj['album']['name']
            artist_name = track_obj['artists'][0]['name']
            artist_id = track_obj['artists'][0]['id']  # Get the first artist's ID
            
            # Fetch artist to get the genre
            artist = sp.artist(artist_id)
            genres = artist['genres']
            
            # Fetch audio features
            audio_features = sp.audio_features([track_id])[0]
            
            song_data.append({
                'album_name': album_name,
                'track_name': track_name,
                'artist_name': artist_name,
                'genre': ', '.join(genres),
                'danceability': audio_features['danceability'],
                'energy': audio_features['energy'],
                'speechiness': audio_features['speechiness'],
                'acousticness': audio_features['acousticness'],
                'valence': audio_features['valence'],
                'instrumentalness': audio_features['instrumentalness']
            })

        # Check if there are more pages of results
        if results['next']:
            results = sp.next(results)
            tracks = results['items']
        else:
            results = None

    return pd.DataFrame(song_data)

# put the data into a dataframe
df = fetch_liked_songs_data()
print(df.head())
df.to_csv('liked_songs.csv', index=False)