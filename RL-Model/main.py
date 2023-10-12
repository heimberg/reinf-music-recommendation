"""
main.py
-------
Main driver module for the music recommendation system.
Manages dataset loading, environment and agent setup, training, and evaluation.
"""

import config
import pandas as pd
from environment import MusicRecommendationEnv
from agent import MusicRecommendationAgent
from stable_baselines3.common.monitor import Monitor


def main():
    try:
        print('Loading dataset...')
        data = pd.read_csv(config.DATA_PATH)
        data[config.STATE_FEATURES] = data[config.STATE_FEATURES].astype('float32')
        print('Dataset loaded successfully.')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Shuffle the dataset
    data_shuffled = data.sample(frac=1).reset_index(drop=True)
    train_env = MusicRecommendationEnv(data_shuffled, config.STATE_FEATURES, mode='train')
    eval_env = Monitor(MusicRecommendationEnv(data_shuffled, config.STATE_FEATURES, mode='eval'))
    
    # Agent initialisieren
    agent = MusicRecommendationAgent(train_env)

    # Hier beginnt der Trainingsprozess, der Evaluierungsprozess wird über Callbacks gesteuert
    agent.train(eval_env, timesteps=config.TRAINING_TIMESTEPS)


if __name__ == '__main__':
    main()
