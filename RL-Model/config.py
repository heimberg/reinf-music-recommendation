# configurations
DATA_PATH = '../data-preprocessing/data-preparation/all_songs_compressed_embeddings.csv'
STATE_FEATURES = ['danceability','energy','speechiness','acousticness','valence','instrumentalness','PCA_1','PCA_2']

# RL-Agent configurations
LEARNING_RATE = 0.0001
GAMMA = 0.99 # discount factor
BUFFER_SIZE = 100_000 # replay buffer size
BL3_POLICY = "MlpPolicy"
CONTEXT_WINDOW_SIZE = 2000 # number of songs that are remembered

# epsilon-greedy exploration
EXPLORATION_EPSILON_INITIAL = 1.0 # initial value of epsilon
EXPLORATION_EPSILON_FINAL = 0.3 # final value of epsilon
EXPLORATION_FRACTION = 0.5 # fraction of training timesteps during which the epsilon factor is decreased to epsilon_final
TRAINING_TIMESTEPS = 2000 # number of training timesteps
EVALUATION_INTERVAL = 200  # test the agent every n timesteps
NUM_EPOCHS = 200 # number of epochs to train the agent

# rewards
REWARD_FOR_LIKED_SONG = 2
REWARD_FOR_UNLIKED_SONG = -1
REWARD_FOR_SAME_SONG = -5
GENRE_DISTANCE_WEIGHT = 2

# load/save configurations
MODEL_SAVE_PATH = 'models/rl_model.pkl'
PLOT_SAVE_PATH = 'visualizations/rl_training_rewards_plot.png'