from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv

class MusicRecommendationAgent:
    # initalize the agent, load the model if a path is given
    def __init__(self, env, model_path=None):
        self.env = DummyVecEnv([lambda: env])
        if model_path:
            self.model = DQN.load(model_path, env=self.env)
        else:
            self.model = DQN("MlpPolicy", self.env, verbose=1)

    # train the agent for a given number of timesteps
    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)

    # save the trained model to a given path
    def save(self, path):
        self.model.save(path)

    # load a trained model from a given path
    def load(self, path):
        self.model = DQN.load(path, env=self.env)

    # predict the action for a given state
    def predict(self, state):
        return self.model.predict(state)
