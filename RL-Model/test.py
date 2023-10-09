def evaluate_agent(agent, env, num_episodes=100):
    total_rewards = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = agent.predict(state)
            state, reward, done, _ = env.step(action)
            total_rewards += reward
            
        total_rewards += episode_reward
    
    average_reward = total_rewards / num_episodes
    return average_reward