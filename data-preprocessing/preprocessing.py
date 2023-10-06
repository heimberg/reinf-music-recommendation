import h5py
import pandas as pd
import os

def read_h5_file(filepath):
    """Same as before, but now it returns a pandas Series."""
    with h5py.File(filepath, 'r') as f:
        song_id = f['metadata']['songs']['song_id'][0].decode('utf-8')
        artist_name = f['metadata']['songs']['artist_name'][0].decode('utf-8')
        title = f['metadata']['songs']['title'][0].decode('utf-8')
        
        # Convert to pandas Series
        return pd.Series({'song_id': song_id, 'artist_name': artist_name, 'title': title})

# Recursively get all h5 files in directory and subdirectories
data_dir = "C:\\Users\\matth\\Downloads\\millionsongsubset.tar\\MillionSongSubset"
all_files = [os.path.join(root, file)
             for root, dirs, files in os.walk(data_dir)
             for file in files if file.endswith('.h5')]

df_list = [read_h5_file(filepath) for filepath in all_files]
songs_df = pd.concat(df_list, axis=1).transpose()

print(songs_df.head())
songs_df.to_csv('songs_data.csv', index=False)
