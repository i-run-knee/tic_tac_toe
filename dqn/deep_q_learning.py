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
        self.grid_size = 4
        self.state = (0, 0)  # Start at top-left corner
        self.reward_room = (3, 3)  # Reward room at bottom-right corner
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

# Deep Q-Learning training function
def dqn(env, episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=200):
    state_size = env.grid_size * env.grid_size
    action_size = env.action_space
    q_network = QNetwork(state_size, action_size)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    memory = deque(maxlen=2000)

    for episode in range(episodes):
        state = env.reset()
        state_vec = state_to_vector(state, env.grid_size)
        total_reward = 0
        agent_path = [state]  # Track agent's path for visualization

        for t in range(50):  # Limit steps per episode
            # Epsilon-greedy action selection
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      np.exp(-1. * episode / epsilon_decay)
            if random.random() < epsilon:
                action = np.random.choice(action_size)  # Explore
            else:
                with torch.no_grad():
                    action = torch.argmax(q_network(torch.FloatTensor(state_vec))).item()  # Exploit

            next_state, reward, done = env.step(action)
            next_state_vec = state_to_vector(next_state, env.grid_size)
            memory.append((state_vec, action, reward, next_state_vec, done))
            state_vec = next_state_vec
            total_reward += reward
            agent_path.append(next_state)  # Track the path

            if done:
                break

            # Experience replay
            if len(memory) > 32:
                batch = random.sample(memory, 32)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
                max_next_q_values = q_network(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

                loss = nn.MSELoss()(q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            visualize_maze(env, agent_path, episode)

# Initialize environment and run DQN
env = MazeEnv()
dqn(env)
