import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import patches

# Maze environment setup (5x5 grid with a reward room)
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

# Define the policy network (actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# Function to select an action based on the policy
def select_action(state, policy):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    action = np.random.choice(len(probs[0]), p=probs.detach().numpy()[0])
    return action, probs[0, action]

# Convert state (grid position) to a vector for the network
def state_to_vector(state, grid_size):
    vec = np.zeros(grid_size * grid_size)
    vec[state[0] * grid_size + state[1]] = 1
    return vec

# Visualization function
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

# REINFORCE training function with visualization
def reinforce(env, policy, optimizer, episodes=1000, gamma=0.99):
    for episode in range(episodes):
        state = env.reset()
        rewards, log_probs = [], []
        agent_path = [state]  # Track agent's path for visualization

        # Generate an episode
        for t in range(50):  # Limit the number of steps per episode
            state_vec = state_to_vector(state, env.grid_size)
            action, log_prob = select_action(state_vec, policy)
            next_state, reward, done = env.step(action)

            log_probs.append(log_prob)  # Store the log probability directly
            rewards.append(reward)
            agent_path.append(next_state)
            state = next_state

            if done:
                break

        # Compute discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Policy gradient update
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            # Ensure that log_prob is a tensor
            policy_loss.append(-log_prob * reward)

        optimizer.zero_grad()

        # Check if policy_loss is non-empty before concatenation
        if policy_loss:  # Only concatenate if there's at least one element
            policy_loss = torch.stack(policy_loss).sum()  # Use torch.stack instead of torch.cat
            policy_loss.backward()
            optimizer.step()

        # Visualize agent's path every 100 episodes
        if episode % 100 == 0:
            visualize_maze(env, agent_path, episode)

        if episode % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward: {sum(rewards)}")

# Initialize environment, policy network, and optimizer
env = MazeEnv()
state_size = env.grid_size * env.grid_size  # One-hot encoded state for grid positions
action_size = env.action_space
policy = PolicyNetwork(state_size, action_size)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Train the agent using REINFORCE with visualization
reinforce(env, policy, optimizer)
