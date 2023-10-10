import matplotlib.pyplot as plt

def plot_songs_on_pca_map(pca_values, liked_flags):
    # Konvertieren Sie die Liste der Tupel in separate Listen für x und y
    x_values = [x[0] for x in pca_values]
    y_values = [x[1] for x in pca_values]
    
    # Erstellen Sie eine Farbliste basierend auf "liked_songs"
    colors = ['red' if liked else 'blue' for liked in liked_flags]
    
    # Erstellen Sie einen Scatter-Plot
    plt.scatter(x_values, y_values, c=colors, alpha=0.5)
    
    # Beschriftungen und Titel hinzufügen
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Songs on PCA Map")
    
    # Legende hinzufügen
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Liked Songs', markersize=10, markerfacecolor='red'),
                        plt.Line2D([0], [0], marker='o', color='w', label='Other Songs', markersize=10, markerfacecolor='blue')],
               loc='upper right')
    
    # Anzeigen des Plots
    plt.show()



