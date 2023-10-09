import gym
from gym import spaces
import numpy as np
import config

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
        self.max_recommendations = 100  # needed for the done condition
        self.current_step = 0
        self.played_genres = [] # list of genres that were already played
        self.genre_memory = 10 # number of genres that are remembered
                
        # define the action space
        # action: choose the next song, for every song in the dataset
        self.action_space = spaces.Discrete(len(self.data))
        
        # set low and high limits for the observation space depending on the state features (0-1/-1-1)
        low_limits = np.array(self.data[state_features].min())
        high_limits = np.array(self.data[state_features].max())
        
        # define the observation space
        # continuous state space -> Box Space (https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box)
        # box space choosed because nuances in the state features could be possibly important
        # alternative: Discrete Space (https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete)
        self.observation_space = spaces.Box(low=low_limits, high=high_limits, dtype=np.float32)
    
    # execute the given action and return new state and reward
    def step(self, action):
        next_state = self.data.iloc[action][self.state_features].values.astype('float32')
        genre_distance_reward = self.calculate_pca_distance_reward(next_state[-2:])  # Last two values are PCA components (genre)
        
        
        # reward if a liked song was chosen, else negative reward
        reward = 1 if self.data.iloc[action]['liked_songs'] == 1 else -1
        # reward playing songs from different genres
        reward += genre_distance_reward  # Add the pca distance reward
        
        # Update the played genre history
        self.played_genres.append(next_state[-2:])
        if len(self.played_genres) > self.genre_memory:
            self.played_genres.pop(0)  # Remove the oldest PCA value if history is too long

        # set the new state
        self.current_state = next_state
        # done by max_recommendations
        self.current_step += 1
        if self.current_step >= self.max_recommendations:
            done = True
        else:
            done = False
        return next_state, reward, done, {}
    
    # reset the environment to the initial state (random state)
    def reset(self):
        self.current_state = self.data.sample(1)[self.state_features].values[0].astype('float32')
        self.current_step = 0
        return self.current_state
    
    def calculate_pca_distance_reward(self, current_genre):
        # If no songs have been played yet, return a reward of 0
        if not self.played_genres:
            return 0
        
        # Calculate the average of the played genres
        average_genre = np.mean(self.played_genres, axis=0)
        
        # Calculate the euclidean distance between the current and the average genre
        distance = np.linalg.norm(current_genre - average_genre)
        
        # Convert distance to  reward. The larger the distance, the higher the reward
        distance_reward = config.GENRE_DISTANCE_WEIGHT * distance
        return distance_reward
        