import pandas as pd
import matplotlib.pyplot as plt

# Daten aus CSV-Datei einlesen
data = pd.read_csv("modified_all_songs_compressed_embeddings.csv")

# Daten filtern
liked_songs = data[data["liked_songs"] == 1]
not_liked_songs = data[data["liked_songs"] != 1]

# 2D-Karte erstellen
plt.figure(figsize=(10, 6))
plt.scatter(liked_songs["PCA_1"], liked_songs["PCA_2"], color='red', label='Liked Songs', alpha=0.6)
plt.scatter(not_liked_songs["PCA_1"], not_liked_songs["PCA_2"], color='blue', label='Not Liked Songs', alpha=0.6)
plt.title("2D Map of Songs based on PCA components")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.savefig("2d_map_of_all_songs2.png")
