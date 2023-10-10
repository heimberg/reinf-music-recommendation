from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import config

class MusicRecommendationAgent:
    # initalize the agent, load the model if a path is given
    def __init__(self, env, model_path=None):
        self.env = DummyVecEnv([lambda: env])
        if model_path:
            self.model = DQN.load(model_path, env=self.env)
        else:
            self.model = DQN(
                policy=config.BL3_POLICY, # DQN Architecture, default MLP Policy (Multi-Layer Perceptron)
                env=self.env,
                learning_rate=config.LEARNING_RATE,
                buffer_size=config.BUFFER_SIZE, # max size of the replay buffer (experience replay)
                exploration_final_eps=config.EXPLORATION_EPSILON_FINAL,
                exploration_fraction=config.EXPLORATION_FRACTION,
                exploration_initial_eps=config.EXPLORATION_EPSILON_INITIAL,
            )
                

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
