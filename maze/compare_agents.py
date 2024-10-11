import numpy as np
import matplotlib.pyplot as plt
from q_learning import QLearningAgent
from deep_q_learning import DeepQLearningAgent

def compare_agents():
    """Compare Q-Learning and Deep Q-Learning performance."""
    param_sets_q = [
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.1},
        {'alpha': 0.5, 'gamma': 0.9, 'epsilon': 0.1},
        {'alpha': 0.1, 'gamma': 0.5, 'epsilon': 0.3},
        {'alpha': 0.5, 'gamma': 0.5, 'epsilon': 0.3}
    ]

    param_sets_dqn = [
        {'alpha': 0.001, 'gamma': 0.9, 'epsilon': 0.1},
        {'alpha': 0.01, 'gamma': 0.9, 'epsilon': 0.1},
        {'alpha': 0.001, 'gamma': 0.5, 'epsilon': 0.3},
        {'alpha': 0.01, 'gamma': 0.5, 'epsilon': 0.3}
    ]

    all_rewards_q = []
    all_steps_q = []
    all_rewards_dqn = []
    all_steps_dqn = []

    for params in param_sets_q:
        agent = QLearningAgent(**params)
        rewards, steps = agent.train(episodes=1000)
        all_rewards_q.append(np.mean(rewards))
        all_steps_q.append(np.mean(steps))

    for params in param_sets_dqn:
        agent = DeepQLearningAgent(**params)
        rewards, steps = agent.train(episodes=1000)
        all_rewards_dqn.append(np.mean(rewards))
        all_steps_dqn.append(np.mean(steps))

    # Plotting comparison for average rewards
    x_labels = [f"Q-Learning {i + 1}" for i in range(len(param_sets_q))] + \
               [f"DQN {i + 1}" for i in range(len(param_sets_dqn))]

    plt.figure(figsize=(12, 6))
    plt.bar(x_labels[:len(param_sets_q)], all_rewards_q, label='Q-Learning Avg Reward', alpha=0.6)
    plt.bar(x_labels[len(param_sets_q):], all_rewards_dqn, label='DQN Avg Reward', alpha=0.6)
    plt.xlabel('Agent Type and Parameter Set')
    plt.ylabel('Average Reward')
    plt.title('Q-Learning vs Deep Q-Learning Average Rewards')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_agents()  # Execute the comparison


