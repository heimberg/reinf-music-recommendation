import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from dotenv import load_dotenv
import os
import time

load_dotenv()

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# Authenticate with Spotify API using spotipy
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

playlists = ['37i9dQZF1DX0XUsuxWHRQd', '37i9dQZF1DXdPec7aLTmlC', '37i9dQZF1DZ06evO2fOjra', '37i9dQZF1DXcZDD7cfEKhW', '37i9dQZF1DWT2jS7NwYPVI', '37i9dQZF1DX9WU5Losjsy8', '37i9dQZF1DXdcGcdKNYAG0', '37i9dQZF1DWVGjWxwGtpup', '37i9dQZF1DWYmmr74INQlb']

# Check if CSV file exists to decide about the header
write_header = not os.path.exists('spotify_playlist_features.csv')

CHUNK_SIZE = 100
WAIT_TIME = 5

def get_all_tracks(spotify, playlist_id, wait_time=WAIT_TIME):
    all_tracks = []
    offset = 0
    limit = 100
    
    while True:
        try:
            items = spotify.playlist_items(playlist_id, limit=limit, offset=offset)
            all_tracks.extend(items['items'])
            if len(items['items']) < limit:
                break
            offset += limit
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error fetching tracks for playlist {playlist_id} at offset {offset}: {e}")
            break

    return all_tracks

for playlist_id in playlists:
    all_tracks = get_all_tracks(sp, playlist_id)

    # Extract IDs, Titles, and Artists from all_tracks with None check
    playlist_tracks_id = [track['track']['id'] for track in all_tracks if track['track'] is not None]
    playlist_tracks_titles = [track['track']['name'] for track in all_tracks if track['track'] is not None]
    playlist_tracks_first_artists = [track['track']['artists'][0]['name'] for track in all_tracks if track['track'] is not None and track['track']['artists']]
    playlist_tracks_album_names = [track['track']['album']['name'] for track in all_tracks if track['track'] is not None]
    
    # Make chunks of tracks to get audio features
    playlist_tracks_id_chunks = [playlist_tracks_id[x:x+CHUNK_SIZE] for x in range(0, len(playlist_tracks_id), CHUNK_SIZE)]
    playlist_tracks_titles_chunks = [playlist_tracks_titles[x:x+CHUNK_SIZE] for x in range(0, len(playlist_tracks_titles), CHUNK_SIZE)]
    playlist_tracks_artists_chunks = [playlist_tracks_first_artists[x:x+CHUNK_SIZE] for x in range(0, len(playlist_tracks_first_artists), CHUNK_SIZE)]
    playlist_tracks_album_names_chunks = [playlist_tracks_album_names[x:x+CHUNK_SIZE] for x in range(0, len(playlist_tracks_album_names), CHUNK_SIZE)]

    # Loop through each chunk and get audio features
    for idx, chunk in enumerate(playlist_tracks_id_chunks):
        try:
            print(f'Getting features for chunk {idx+1} of {len(playlist_tracks_id_chunks)}')
            features = sp.audio_features(chunk)
            time.sleep(WAIT_TIME)

            # Create a dataframe with audio features
            features_df = pd.DataFrame(data=features)
            features_df['album_name'] = playlist_tracks_album_names_chunks[idx]
            features_df['track_name'] = playlist_tracks_titles_chunks[idx]
            features_df['artist_name'] = playlist_tracks_artists_chunks[idx]

            # Extract genre for the primary artist for each track
            genres = []
            for artist in playlist_tracks_artists_chunks[idx]:
                try:
                    artist_data = sp.search(q=f'artist:{artist}', type='artist', limit=1)
                    artist_genre = artist_data['artists']['items'][0]['genres']
                    genres.append(', '.join(artist_genre))
                    print(f'Genre for {artist}: {artist_genre}')
                    print('---')
                except Exception as e:
                    print(f"Error fetching genre for artist {artist}: {e}")
                    genres.append('')
                time.sleep(WAIT_TIME)

            features_df['genre'] = genres
            print(features_df.head())

            # Rearrange columns and append to CSV
            columns_order = ['album_name', 'track_name', 'artist_name', 'genre',
                             'danceability', 'energy', 'speechiness', 
                             'acousticness', 'valence', 'instrumentalness']
            features_df = features_df[columns_order]
            features_df.to_csv('spotify_playlist_features.csv', mode='a', header=write_header, index=False)

            # Set write_header to False after the first write
            write_header = False
        except Exception as e:
            print(f"Error processing chunk {idx+1}: {e}")
