"""
visualizations.py
-----------------
Module providing visualization utilities for the music recommendation system.
Includes functions for plotting songs on PCA maps and plotting the learning curve.
"""

import matplotlib.pyplot as plt
import datetime
import config

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

def plot_learning_curve(all_epoch_rewards):
    plt.figure(figsize=(12, 6))
    timesteps = range(0, config.TRAINING_TIMESTEPS, config.EVALUATION_INTERVAL)
    for epoch, rewards in enumerate(all_epoch_rewards):
        plt.plot(timesteps, rewards, label=f'Epoch {epoch+1}')
    
    plt.title('Learning Curve')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)

    # Get the hyperparameters from the config file
    hyperparameters = {k: getattr(config, k) for k in dir(config) if not k.startswith('_')}
    hyperparam_str = "\n".join([f"{key}: {value}" for key, value in hyperparameters.items()])
    # add the hyperparameters as text to the plot
    plt.figtext(1.02, 0.5, "Hyperparameters:\n" + hyperparam_str, horizontalalignment='left', verticalalignment='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
    
    # Saving the figure with current date and time
    current_time = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    filename = config.PLOT_SAVE_PATH + f"learning_curve_{current_time}.png"
    plt.tight_layout()
    # bbox_inches='tight' to cut off the white space around the plot
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

    return filename 

    

