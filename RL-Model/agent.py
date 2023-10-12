"""
agent.py
--------
Module for music recommendation using the DQN algorithm from stable_baselines3.
Defines the MusicRecommendationAgent class for initializing, training, and making predictions.
"""

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import config

  
class MusicRecommendationAgent:
    # initalize the agent (new DQN-Model), load existing model if a path is given
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
                tensorboard_log=config.TENSORBOARD_LOG_DIR
            )
                

    # train the agent for a given number of timesteps
    def train(self, eval_env, timesteps=config.TRAINING_TIMESTEPS):
        stop_on_max_reward = StopTrainingOnMaxRewardCallback(reward_threshold=config.REWARD_THRESHOLD)
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model',
                                     log_path='./logs/results', eval_freq=config.LOG_EVAL_FREQUENCY,
                                     deterministic=True, render=False)
        self.model.learn(total_timesteps=timesteps, callback=[eval_callback, stop_on_max_reward])
        
    # predict the action for a given state
    def predict(self, state, deterministic=False):
        # deterministic: if True, the action with the highest probability is chosen
        predict = self.model.predict(state, deterministic=deterministic)
        return predict

    # save the trained model to a given path
    def save(self, path):
        self.model.save(path)

    # load a trained model from a given path
    def load(self, path):
        self.model = DQN.load(path, env=self.env)

class StopTrainingOnMaxRewardCallback(BaseCallback):
    def __init__(self, reward_threshold):
        super(StopTrainingOnMaxRewardCallback, self).__init__()
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        is_training_continue = True
        if self.model.ep_info_buffer:
            mean_reward = sum([ep_info["r"] for ep_info in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
            if mean_reward >= self.reward_threshold:
                print(f"Stopping training because mean reward {mean_reward} reached the set threshold {self.reward_threshold}")
                is_training_continue = False
        return is_training_continue