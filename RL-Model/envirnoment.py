import gym
from gym import spaces
import pandas as pd
import numpy as np

# music recommendation environment class (inheriting gym.Env)
class MusicRecommendationEnv(gym.Env):
    # constructor
    # data_path: path to the dataset
    # state_features: list of features to be used as state
    def __init__(self, data_path, state_features):
        super(MusicRecommendationEnv, self).__init__()
        
        # load the dataset
        self.data = pd.read_csv(data_path)
        self.state_features = state_features
        self.current_state = None
        
        # define the action space
        # only one action: choose the next song
        self.action_space = spaces.Discrete(1)
        return NotImplemented
    
    def step(self, action):
        return NotImplemented
    
    def reset(self):
        return NotImplemented
    
    def render(self, mode='human'):
        return NotImplemented
