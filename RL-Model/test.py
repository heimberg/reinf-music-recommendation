# added evaluate flag to set exploration on/off during evaluation
def evaluate_agent(agent, env, num_episodes=100, evaluate=False):
    total_rewards = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = agent.predict(state, deterministic=evaluate)  # Use the deterministic flag
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
        total_rewards += episode_reward
        
    average_reward = total_rewards / num_episodes
    return average_reward
