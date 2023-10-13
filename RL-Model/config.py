"""
config.py
---------
Module containing configurations for data paths, RL-agent parameters, and reward definitions.
"""

# configurations
DATA_PATH = '../data-preprocessing/data-preparation/all_songs_compressed_embeddings.csv'
STATE_FEATURES = ['danceability','energy','speechiness','acousticness','valence','instrumentalness','PCA_1','PCA_2']

# RL-Agent configurations
LEARNING_RATE = 0.00001
GAMMA = 0.9 # discount factor
BUFFER_SIZE = 10_000 # replay buffer size (experience replay)
BL3_POLICY = "MlpPolicy"
CONTEXT_WINDOW_SIZE = 2000 # number of songs that are remembered for the state
TRAINING_TIMESTEPS = 60_000 # number of training timesteps
EPISODE_LENGTH = 500 # maximum number of recommendations until the episode is done (same for eval)
LEARNING_STARTS = 1000 # number of steps before the first update of the model (warmup)

# epsilon-greedy exploration
EXPLORATION_EPSILON_INITIAL = 0.4 # initial value of epsilon
EXPLORATION_EPSILON_FINAL = 0.15 # final value of epsilon
EXPLORATION_FRACTION = 0.2 # fraction of training timesteps during which the epsilon factor is decreased to epsilon_final

# rewards
REWARD_FOR_LIKED_SONG = 1
PENALTY_FOR_UNLIKED_SONG = -0.5
PENALTY_FOR_SAME_SONG = -1
GENRE_DISTANCE_WEIGHT = 1
REWARD_THRESHOLD = 200 # stop training if the average reward exceeds this value
SONG_HISTORY_SIZE = 20 # number of songs that are remembered for the same song penalty

# load/save configurations
MODEL_SAVE_PATH = 'models/'
PLOT_SAVE_PATH = 'visualizations/'
TENSORBOARD_LOG_DIR = 'logs/'
LOG_EVAL_FREQUENCY = 2500