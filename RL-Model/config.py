# configurations
DATA_PATH = '../data-preparation/all_songs_compresed_embeddings.csv'
STATE_FEATURES = ['danceability','energy','speechiness','acousticness','valence','instrumentalness','PCA_1','PCA_2']

# RL-Agent configurations
LEARNING_RATE = 0.001
GAMMA = 0.99 # discount factor
BUFFER_SIZE = 100_000 # replay buffer size
# epsilon-greedy exploration
EXPLORATION_EPSILON_INITIAL = 1.0 # initial value of epsilon
EXPLORATION_EPSILON_FINAL = 0.02 # final value of epsilon
EXPLORATION_FRACTION = 0.1 # fraction of training timesteps during which the epsilon factor is decreased to epsilon_final
TRAINING_TIMESTEPS = 10_000 # number of training timesteps

# load/save configurations
MODEL_SAVE_PATH = 'models/rl_model.pkl'