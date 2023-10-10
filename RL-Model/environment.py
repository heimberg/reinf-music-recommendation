import gym
from gym import spaces
import numpy as np
import config
from collections import deque

# music recommendation environment class (inheriting gym.Env)
# defines the action and observation space as well as the step function including the reward
class MusicRecommendationEnv(gym.Env):
    # constructor
    # data_path: path to the dataset
    # state_features: list of features to be used as state (danceability,energy,speechiness,acousticness,valence,instrumentalness,PCA_1,PCA_2)
    def __init__(self, data, state_features):
        super(MusicRecommendationEnv, self).__init__()
        
        # load the dataset
        self.data = data
        self.state_features = state_features
        self.current_state = None
        self.max_recommendations = 20  # needed for the done condition
        self.current_step = 0
        self.played_genres = [] # list of genres that were already played
        self.genre_memory = 10 # number of genres that are remembered
        self.action_history = []
        self.pca_history = []
        self.liked_history = []
        self.context_window = deque(maxlen=config.CONTEXT_WINDOW_SIZE) # list of the last x songs that were played
                
        # define the action space
        # action: choose the next song, for every song in the dataset
        self.action_space = spaces.Discrete(len(self.data))
        
        # set low and high limits for the observation space depending on the state features (0-1/-1-1)
        # adjust for the context window size
        low_limits = np.tile(np.array(self.data[state_features].min()), config.CONTEXT_WINDOW_SIZE)
        high_limits = np.tile(np.array(self.data[state_features].max()), config.CONTEXT_WINDOW_SIZE)

        
        # define the observation space
        # continuous state space -> Box Space (https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box)
        # box space choosed because nuances in the state features could be possibly important
        # alternative: Discrete Space (https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete)
        self.observation_space = spaces.Box(low=low_limits, high=high_limits, dtype=np.float32)
    
    # execute the given action and return new state and reward
    def step(self, action):
        next_state = self.data.iloc[action][self.state_features].values.astype('float32')
        # genre_distance_reward = self.calculate_pca_distance_reward(next_state[-2:])  # Last two values are PCA components (genre)
        
        current_song = self.data.iloc[action]
        self.pca_history.append(current_song[['PCA_1', 'PCA_2']].values)
        self.liked_history.append(current_song['liked_songs'])
        
        # rewards
        # reward if a liked song was chosen, else negative reward
        reward = config.REWARD_FOR_LIKED_SONG if self.data.iloc[action]['liked_songs'] == 1 else config.REWARD_FOR_UNLIKED_SONG
        # negative reward if the same song was chosen already in the last x steps
        if action in self.action_history[-100:]:
            reward += config.REWARD_FOR_SAME_SONG
        # reward playing songs from different genres
        # reward += genre_distance_reward  # Add the pca distance reward
        
        # update action history
        self.action_history.append(action)
        
        # Update the played genre history
        self.played_genres.append(next_state[-2:])
        if len(self.played_genres) > self.genre_memory:
            self.played_genres.pop(0)  # Remove the oldest PCA value if history is too long

        # set the new state
        self.current_state = self._get_current_state()
        # done by max_recommendations
        self.current_step += 1
        if self.current_step >= self.max_recommendations:
            done = True
        else:
            done = False
        
        # Log details:
        #print(f"Action taken: {action}")
        # print(f"Reward received: {reward}")
        # print(f"Current state: {self.current_state}")

        # Update the context window with the next state
        self.context_window.append(next_state)
        return self.current_state, reward, done, {}
    
    # reset the environment to the initial state (random state)
    def reset(self):
        # Initialize/reset histories and counters
        self.action_history = []
        self.current_step = 0

        # Initialize the context window
        self.context_window = deque(maxlen=config.CONTEXT_WINDOW_SIZE)

        # Initialize the context window with default states (zero vectors)
        default_state = np.zeros(len(self.state_features))  
        for _ in range(config.CONTEXT_WINDOW_SIZE):
            self.context_window.append(default_state)

        # Sample an initial state and add it to the context window
        initial_state = self.data.sample()[self.state_features].values.astype('float32')[0]
        self.context_window.append(initial_state)

        # Form the full state including the context and set it as the current state
        self.current_state = self._get_current_state()

        return self.current_state

    def _get_current_state(self):
        # Flatten the context window and concatenate to form the current state
        return np.concatenate([state for state in self.context_window])

    
    def calculate_pca_distance_reward(self, current_genre):
        # If no songs have been played yet, return a reward of 0
        if not self.played_genres:
            return 0
        
        # Calculate the average of the played genres
        average_genre = np.mean(self.played_genres, axis=0)
        
        # Calculate the squared distance between the current genre and the average genre
        distance = np.sum((current_genre - average_genre) ** 2)
        
        # Convert distance to  reward. The larger the distance, the higher the reward
        distance_reward = config.GENRE_DISTANCE_WEIGHT * distance
        return distance_reward
        