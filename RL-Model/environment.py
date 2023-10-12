"""
environment.py
--------------
Module defining the music recommendation environment for the RL agent.
Provides the MusicRecommendationEnv class, inheriting from gym.Env, and specifies action and observation spaces.
"""

import gym
from gym import spaces
import numpy as np
import config
from collections import deque


class MusicRecommendationEnv(gym.Env):

    def __init__(self, data, state_features, mode='train'):
        super(MusicRecommendationEnv, self).__init__()
        self.mode = mode
        
        # load the dataset
        self.data = data
        self.state_features = state_features
        self.current_state = None
        self.max_recommendations = config.EPISODE_LENGTH  # needed for the done condition
        
        self.current_step = 0
        self.played_songs_set = set()
        self.action_history = []
        self.context_window = deque(maxlen=config.CONTEXT_WINDOW_SIZE)
        
        # define the action space
        # action: choose the next song, for every song in the dataset
        self.action_space = spaces.Discrete(len(self.data))
        
        # set low and high limits for the observation space depending on the state features (0-1 / -1-1)
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
        # print(f'Running in {self.mode} mode.')
        # print(f'Step {self.current_step} of {self.max_recommendations}')
        # print(f'Action taken: {action}')
        next_state = self._update_state(action)
        reward = self._calculate_reward(action)
        done = self._is_done()
        # Add the selected action (song) to the set of played songs
        self.played_songs_set.add(int(action))
        
        return next_state, reward, done, {}
    
    # reset the environment to the initial state (random state)
    def reset(self):
        # Initialize/reset histories and counters
        self.played_songs_set.clear()
        self.action_history = []
        self.current_step = 0

        # Initialize the context window with default states (zero vectors)
        default_state = np.zeros(len(self.state_features))  
        for _ in range(config.CONTEXT_WINDOW_SIZE):
            self.context_window.append(default_state)

        # Sample a random initial state
        initial_state = self.data.sample()[self.state_features].values.astype('float32')[0]
        self.context_window.append(initial_state)
        
        return self._get_current_state()

    def _update_state(self, action):
        self.current_step += 1
        self.action_history.append(action)
        next_state = self.data.iloc[action][self.state_features].values.astype('float32')
        self.context_window.append(next_state)
        
        return self._get_current_state()
    
    def _calculate_reward(self, action):
        # print("Size of played_songs_set:", len(self.played_songs_set))
        # TODO: Reimplement reward for genre distance

        if action in self.played_songs_set:
            return config.PENALTY_FOR_SAME_SONG
        elif self.data.iloc[action]['liked_songs'] == 1:
            return config.REWARD_FOR_LIKED_SONG
        else:
            return config.PENALTY_FOR_UNLIKED_SONG
    def _is_done(self):
        return self.current_step >= self.max_recommendations
    
    def _get_current_state(self):
        # Flatten the context window and concatenate to form the current state
        return np.concatenate([state for state in self.context_window])


        