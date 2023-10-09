import gym
from gym import spaces
import pandas as pd
import numpy as np

# music recommendation environment class (inheriting gym.Env)
class MusicRecommendationEnv(gym.Env):
    # constructor
    # data_path: path to the dataset
    # state_features: list of features to be used as state (danceability,energy,speechiness,acousticness,valence,instrumentalness,PCA_1,PCA_2)
    def __init__(self, data_path, state_features):
        super(MusicRecommendationEnv, self).__init__()
        
        # load the dataset
        self.data = pd.read_csv(data_path)
        self.state_features = state_features
        self.current_state = None
        
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
    
    def step(self, action):
        return NotImplemented
    
    def reset(self):
        return NotImplemented
    
    def render(self, mode='human'):
        return NotImplemented
