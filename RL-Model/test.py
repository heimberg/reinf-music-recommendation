def evaluate_agent(agent, env, num_episodes=100, evaluate=False):
    total_rewards = 0
    all_actions = []  # List to store actions taken by the agent
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_actions = []  # Store actions for each episode
        done = False
        
        while not done:
            action, _ = agent.predict(state, deterministic=evaluate)
            episode_actions.append(action)
            state, reward, done, _ = env.step(action)
            total_rewards += reward
            
        all_actions.append(episode_actions)
        
    average_reward = total_rewards / num_episodes
    return average_reward, all_actions
