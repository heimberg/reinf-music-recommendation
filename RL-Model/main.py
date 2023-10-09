import config
from utils import load_data
from environment import MusicRecommendationEnv
from agent import MusicRecommendationAgent
from test import evaluate_agent

def main():
    # load the dataset and convert the state features to float32
    print('Loading dataset...')
    data = load_data(config.DATA_PATH)
    data[config.STATE_FEATURES] = data[config.STATE_FEATURES].astype('float32')
    print(data[config.STATE_FEATURES].dtypes)

    
    # init RL-environment
    print('Initializing environment...')
    env = MusicRecommendationEnv(data, config.STATE_FEATURES)
    
    # init RL-agent
    print('Initializing agent...')
    agent = MusicRecommendationAgent(env)
    
    # train & test the agent
    rewards = []
    print('Training agent...')
    for i in range(0, config.TRAINING_TIMESTEPS, 1000):
        agent.train(timesteps=1000)
        average_reward = evaluate_agent(agent, env)
        print(f'Average reward after {i+1000} timesteps: {average_reward}')
        rewards.append(average_reward)
    
    # save the trained model
    print('Saving model...')
    agent.save(config.MODEL_SAVE_PATH)
    

if __name__ == '__main__':
    main()