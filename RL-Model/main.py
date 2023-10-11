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
from test import evaluate_agent
from visualizations import plot_learning_curve
import matplotlib.pyplot as plt


def main():
    try:
        print('Loading dataset...')
        data = pd.read_csv(config.DATA_PATH)
        data[config.STATE_FEATURES] = data[config.STATE_FEATURES].astype('float32')
        print('Dataset loaded successfully.')
        print(data[config.STATE_FEATURES].dtypes)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    all_epoch_rewards = []

    # Create environment and agent
    env = MusicRecommendationEnv(data, config.STATE_FEATURES)
    agent = MusicRecommendationAgent(env)

    # Print hyperparameters (assuming they are in config)
    print("Training with hyperparameters:", vars(config))

    best_avg_reward = float('-inf')  # For model checkpointing

    # Train and evaluate for a set number of epochs
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n==== Epoch {epoch + 1}/{config.NUM_EPOCHS} ====")

        env.reset()
        
        rewards = []

        # Train for one epoch
        for i in range(0, config.TRAINING_TIMESTEPS, config.EVALUATION_INTERVAL):
            print(f'Training for timesteps {i}-{i+config.EVALUATION_INTERVAL}...')
            # Train agent on interval and get average reward
            agent.train(timesteps=config.EVALUATION_INTERVAL)
            average_reward, actions_taken = evaluate_agent(agent, env, config.EVALUATION_INTERVAL, evaluate=True)
            rewards.append(average_reward)
            print(f'Average reward after {i+config.EVALUATION_INTERVAL} timesteps: {average_reward:.2f}')
            # Analyze the actions taken by the agent
            unique_actions = set(action.item() for episode in actions_taken for action in episode)
            print(f"Unique actions taken during this interval: {unique_actions}")
            # Checkpoint model if its the best so far
            if average_reward > best_avg_reward:
                best_avg_reward = average_reward
                agent.save(config.MODEL_SAVE_PATH + f"best_model_epoch{epoch + 1}.pkl")

        all_epoch_rewards.append(rewards)

    filename = plot_learning_curve(all_epoch_rewards)
    print(f"Learning curve saved as {filename}")
    
    # save the final model
    try:
        print('Saving final model...')
        agent.save(config.MODEL_SAVE_PATH + "final_model.pkl")
        print('Model saved successfully.')
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == '__main__':
    main()
