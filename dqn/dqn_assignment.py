import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import patches
from collections import deque

# Define the Maze Environment
class MazeEnv:
    def __init__(self):
        self.grid_size = 5
        self.state = (0, 0)  # Start at top-left corner
        self.reward_room = (4, 4)  # Reward room at bottom-right corner
        self.action_space = 4  # Up, down, left, right

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state

        # Action 0 = up, 1 = down, 2 = left, 3 = right
        if action == 0 and x > 0: x -= 1
        elif action == 1 and x < self.grid_size - 1: x += 1
        elif action == 2 and y > 0: y -= 1
        elif action == 3 and y < self.grid_size - 1: y += 1
        
        new_state = (x, y)
        self.state = new_state

        # Check if reached the reward room
        if new_state == self.reward_room:
            return new_state, 10, True  # Reward of 10, done = True
        else:
            return new_state, -1, False  # Penalty of -1 for each step

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to convert state to a one-hot vector
def state_to_vector(state, grid_size):
    vec = np.zeros(grid_size * grid_size)
    vec[state[0] * grid_size + state[1]] = 1
    return vec

# Function to visualize the maze and agent's path
def visualize_maze(env, agent_path, episode):
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Create grid lines
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            ax.add_patch(patches.Rectangle((y, x), 1, 1, fill=False, edgecolor='black'))
    
    # Mark reward room
    reward_x, reward_y = env.reward_room
    ax.add_patch(patches.Rectangle((reward_y, reward_x), 1, 1, fill=True, color='green', alpha=0.5))
    plt.text(reward_y + 0.4, reward_x + 0.5, 'G', fontsize=12, ha='center', va='center')

    # Plot agent's path
    for (x, y) in agent_path:
        plt.plot(y + 0.5, x + 0.5, 'ro', markersize=10)  # Plot agent as red circle

    # Set limits and labels
    plt.xlim(-0.5, env.grid_size - 0.5)
    plt.ylim(-0.5, env.grid_size - 0.5)
    plt.gca().invert_yaxis()  # Invert y-axis so (0,0) is at the top-left
    plt.title(f"Episode {episode + 1} - Agent's Path")
    plt.show()

# Function to run a single episode
def run_episode(env, q_network, epsilon, gamma, memory, optimizer):

    return total_reward, agent_path  # Return total reward and the agent's path

# Deep Q-Learning training function
def dqn(env, episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=200):
    state_size = env.grid_size * env.grid_size
    action_size = env.action_space
    q_network = QNetwork(state_size, action_size)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    memory = deque(maxlen=2000)

    for episode in range(episodes):
        # Calculate epsilon for this episode
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  np.exp(-1. * episode / epsilon_decay)

        # Run a single episode
        total_reward, agent_path = run_episode(env, q_network, epsilon, gamma, memory, optimizer)

        if episode % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            visualize_maze(env, agent_path, episode)

# Initialize environment and run DQN
env = MazeEnv()
dqn(env)
