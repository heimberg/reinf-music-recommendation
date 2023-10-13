"""
agent.py
--------
Module for music recommendation using the DQN algorithm from stable_baselines3.
Defines the MusicRecommendationAgent class for initializing, training, and making predictions.
"""

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import HParam
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
                learning_starts=config.LEARNING_STARTS, # number of steps before the first update
                buffer_size=config.BUFFER_SIZE, # max size of the replay buffer (experience replay)
                exploration_final_eps=config.EXPLORATION_EPSILON_FINAL,
                exploration_fraction=config.EXPLORATION_FRACTION,
                exploration_initial_eps=config.EXPLORATION_EPSILON_INITIAL,
                tensorboard_log=config.TENSORBOARD_LOG_DIR,
                gamma=config.GAMMA # discount factor
            )
                

    # train the agent for a given number of timesteps
    def train(self, eval_env, timesteps=config.TRAINING_TIMESTEPS):
        stop_on_max_reward = StopTrainingOnMaxRewardCallback(reward_threshold=config.REWARD_THRESHOLD)
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model',
                                     log_path='./logs/results', eval_freq=config.LOG_EVAL_FREQUENCY,
                                     deterministic=True, render=False)
        self.model.learn(total_timesteps=timesteps, progress_bar = True, callback=[eval_callback, stop_on_max_reward, HParamCallback()])
        
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
    
# log hyperparameters and metrics to Tensorboard
# from https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "buffer size": self.model.buffer_size,
            "exploration fraction": self.model.exploration_fraction,
            "exploration initial eps": self.model.exploration_initial_eps,
            "exploration final eps": self.model.exploration_final_eps,
            "policy": config.BL3_POLICY,
            "context window size": config.CONTEXT_WINDOW_SIZE,
            "training timesteps": config.TRAINING_TIMESTEPS,
            "episode length": config.EPISODE_LENGTH,
            "reward for liked song": config.REWARD_FOR_LIKED_SONG,
            "penalty for unliked song": config.PENALTY_FOR_UNLIKED_SONG,
            "penalty for same song": config.PENALTY_FOR_SAME_SONG,
            "genre distance weight": config.GENRE_DISTANCE_WEIGHT,
            "reward threshold": config.REWARD_THRESHOLD,
            "song history size": config.SONG_HISTORY_SIZE,
            "learning starts": config.LEARNING_STARTS
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True