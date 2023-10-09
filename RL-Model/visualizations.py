import matplotlib.pyplot as plt

def plot_action_distribution(actions, filename):
    scalar_actions = [a[0] for a in actions]
    plt.hist(scalar_actions, bins=len(set(scalar_actions)), density=True)
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.savefig('./visualizations/' + filename + '.png')

def plot_rewards(rewards, filename):
    plt.plot(rewards)
    plt.title("Reward over Time")
    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.savefig('./visualizations/' + filename + '.png')

def plot_pca_heatmap(pca_values, filename):
    pca_1_values = [x[0] for x in pca_values]
    pca_2_values = [x[1] for x in pca_values]
    
    plt.hist2d(pca_1_values, pca_2_values, bins=(50, 50), cmap=plt.cm.jet)
    plt.title("PCA Genre Heatmap")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar()
    plt.savefig('./visualizations/' + filename + '.png')


