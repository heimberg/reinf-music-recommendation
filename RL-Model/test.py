"""
test.py
-------
Module containing utility functions for evaluating the performance of the music recommendation agent.
Offers insights into the agent's average reward over multiple episodes and the actions taken.
"""
import numpy as np


def evaluate_agent(agent, env, num_episodes=100, evaluate=False):
    total_rewards = 0
    all_actions = []  # List to store actions taken by the agent
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_actions = []  # Store actions for each episode
        done = False
        
        while not done:
            action_array, _ = agent.predict(state, deterministic=evaluate)
            # Extract action value with ugly workaround for type errors
            action_value = action_array.item() if isinstance(action_array, np.ndarray) and action_array.ndim == 0 else action_array[0]
            
            action = int(action_value)
            episode_actions.append(action)
            state, reward, done, _ = env.step(action)
            total_rewards += reward
            
        all_actions.append(episode_actions)
        
    average_reward = total_rewards / num_episodes
    return average_reward, all_actions
