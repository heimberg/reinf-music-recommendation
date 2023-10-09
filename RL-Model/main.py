import config
from utils import load_data
from environment import MusicRecommendationEnv
from agent import MusicRecommendationAgent
from test import evaluate_agent
from visualizations import plot_pca_heatmap, plot_action_distribution, plot_rewards
import matplotlib.pyplot as plt

def main():
    # load the dataset and convert the state features to float32
    print('Loading dataset...')
    data = load_data(config.DATA_PATH)
    data[config.STATE_FEATURES] = data[config.STATE_FEATURES].astype('float32')
    print(data[config.STATE_FEATURES].dtypes)

    all_epoch_rewards = []

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nStarting Epoch {epoch + 1}...")
        
        # init RL-environment
        print('Initializing environment...')
        env = MusicRecommendationEnv(data, config.STATE_FEATURES)
        
        # init RL-agent
        print('Initializing agent...')
        agent = MusicRecommendationAgent(env)
        
        # train & test the agent
        rewards = []
        print('Training agent...')
        for i in range(0, config.TRAINING_TIMESTEPS, config.EVALUATION_INTERVAL):
            agent.train(timesteps=config.EVALUATION_INTERVAL)
            average_reward = evaluate_agent(agent, env, config.EVALUATION_INTERVAL)
            print(f'Average reward after {i+config.EVALUATION_INTERVAL} timesteps: {average_reward}')
            rewards.append(average_reward)
        
        all_epoch_rewards.append(rewards)
        
        # At the end of each epoch, visualize the data
        action_filename = f"epoch_{epoch + 1}_actions"
        pca_filename = f"epoch_{epoch + 1}_genre_heatmap"
        plot_action_distribution(env.action_history, action_filename)
        plot_pca_heatmap(env.pca_history, pca_filename)

    # Plot training rewards
    plt.figure(figsize=(10, 6))
    for epoch, rewards in enumerate(all_epoch_rewards):
        plt.plot(range(0, config.TRAINING_TIMESTEPS, config.EVALUATION_INTERVAL), rewards, label=f'Epoch {epoch + 1}')
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Reward")
    plt.title("Training Progress of Music Recommendation Agent over Multiple Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(config.PLOT_SAVE_PATH)
    plt.show()
    
    # save the trained model (last model)
    print('Saving model...')
    agent.save(config.MODEL_SAVE_PATH)

if __name__ == '__main__':
    main()
