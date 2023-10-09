import gym
from gym import spaces
import numpy as np

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
        # reward if a liked song was chosen, else negative reward
        reward = 1 if self.data.iloc[action]['liked_songs'] == 1 else -1
        # set the new state
        self.current_state = next_state
        # TODO: check if done
        done = False
        return next_state, reward, done, {}
    
    # reset the environment to the initial state (random state)
    def reset(self):
        self.current_state = self.data.sample(1)[self.state_features].values[0].astype('float32')
        return self.current_state