import config
from utils import load_data
from environment import MusicRecommendationEnv
from agent import MusicRecommendationAgent
from test import evaluate_agent
from visualizations import plot_songs_on_pca_map
import matplotlib.pyplot as plt

def main():
    try:
        print('Loading dataset...')
        data = load_data(config.DATA_PATH)
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

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n==== Epoch {epoch + 1}/{config.NUM_EPOCHS} ====")

        env.reset()
        
        rewards = []

        for i in range(0, config.TRAINING_TIMESTEPS, config.EVALUATION_INTERVAL):
            print(f'Training for timesteps {i}-{i+config.EVALUATION_INTERVAL}...')
            agent.train(timesteps=config.EVALUATION_INTERVAL)
            average_reward = evaluate_agent(agent, env, config.EVALUATION_INTERVAL, evaluate=True)
            rewards.append(average_reward)
            print(f'Average reward after {i+config.EVALUATION_INTERVAL} timesteps: {average_reward:.2f}')
            
            # Checkpoint model if it's the best so far
            if average_reward > best_avg_reward:
                best_avg_reward = average_reward
                agent.save(f"best_model_epoch{epoch + 1}.pth")

        all_epoch_rewards.append(rewards)

    plt.figure(figsize=(12, 6))
    timesteps = range(0, config.TRAINING_TIMESTEPS, config.EVALUATION_INTERVAL)
    for epoch, rewards in enumerate(all_epoch_rewards):
        plt.plot(timesteps, rewards, label=f'Epoch {epoch+1}')
    
    plt.title('Learning Curve')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve.png")
    plt.show()

    try:
        print('Saving final model...')
        agent.save(config.MODEL_SAVE_PATH)
        print('Model saved successfully.')
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == '__main__':
    main()
