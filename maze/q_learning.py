import numpy as np
import random
import matplotlib.pyplot as plt

# Define the rooms and actions
rooms = ['Room 1', 'Room 2', 'Room 3', 'Room 4']
actions = ['move_right', 'move_left', 'move_up', 'move_down']

# Define rewards for rooms
rewards = {
    'Room 1': -1,   # Penalty for being in Room 1
    'Room 2': 10,   # Treasure found in Room 2
    'Room 3': -5,   # Trap in Room 3
    'Room 4': 5     # Small reward in Room 4
}

# Define room connections
room_connections = {
    'Room 1': ['Room 2', 'Room 3'],
    'Room 2': ['Room 1', 'Room 4'],
    'Room 3': ['Room 1'],
    'Room 4': ['Room 2']
}

class QLearningAgent:
    """Q-Learning agent for navigating rooms."""
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.q_table = np.zeros((len(rooms), len(actions)))  # Q-table initialized to zero

    def get_action(self, state):
        """Select action using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:  # Explore
            return np.random.choice(len(actions))
        return np.argmax(self.q_table[state])  # Exploit

    def train(self, episodes=1000):
        """Train the Q-Learning agent over a specified number of episodes."""
        rewards_per_episode = []
        steps_per_episode = []

        for episode in range(episodes):
            state_index = random.randint(0, len(rooms) - 1)  # Start from a random room
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action = self.get_action(state_index)  # Choose action
                next_room = room_connections[rooms[state_index]][action % len(room_connections[rooms[state_index]])]
                next_state_index = rooms.index(next_room)  # Get next state index
                reward = rewards[next_room]  # Get reward for next room

                # Update Q-value using the Bellman equation
                self.q_table[state_index][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state_index]) - self.q_table[state_index][action])

                state_index = next_state_index  # Move to next state
                total_reward += reward  # Accumulate total reward
                steps += 1

                if next_room == 'Room 2':  # Check if treasure is found
                    done = True

            rewards_per_episode.append(total_reward)  # Store total reward
            steps_per_episode.append(steps)  # Store steps taken

        return rewards_per_episode, steps_per_episode

def evaluate_agent(agent, episodes=100):
    """Evaluate the agent's success rate and average steps over a number of episodes."""
    successes = 0
    total_steps = 0

    for _ in range(episodes):
        state_index = random.randint(0, len(rooms) - 1)  # Start from a random room
        done = False
        steps = 0

        while not done:
            action = agent.get_action(state_index)  # Choose action
            next_room = room_connections[rooms[state_index]][action % len(room_connections[rooms[state_index]])]
            next_state_index = rooms.index(next_room)  # Get next state index

            state_index = next_state_index  # Move to next state
            steps += 1

            if next_room == 'Room 2':  # Check if treasure is found
                successes += 1
                done = True

        total_steps += steps

    success_rate = (successes / episodes) * 100  # Success rate in percentage
    avg_steps = total_steps / episodes  # Average steps taken
    return success_rate, avg_steps

def plot_rewards_and_steps(rewards, steps, title='Q-Learning: Rewards and Steps per Episode'):
    """Plot the total rewards and steps for each episode."""
    plt.figure(figsize=(12, 6))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Total Reward')
    plt.title(title + ' - Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()

    # Plot steps
    plt.subplot(1, 2, 2)
    plt.plot(steps, label='Steps', color='orange')
    plt.title(title + ' - Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_parameter_comparison(test_results):
    """Plot average rewards for different parameter sets using a histogram."""
    plt.figure(figsize=(10, 6))

    # Extract data for plotting
    avg_rewards = [result['avg_reward'] for result in test_results]
    labels = [f"α={result['alpha']}, γ={result['gamma']}, ε={result['epsilon']}" for result in test_results]

    # Create histogram of average rewards
    plt.bar(labels, avg_rewards, color='skyblue')
    plt.title('Average Rewards for Different Q-Learning Parameter Sets')
    plt.xlabel('Parameter Sets')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def run_test_cases():
    """Run test cases with varying hyperparameters."""
    test_results = []

    for alpha in [0.1, 0.5]:
        for gamma in [0.5, 0.9]:
            for epsilon in [0.1, 0.3]:
                # Train Q-Learning Agent
                agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
                rewards, steps = agent.train(episodes=1000)

                # Evaluate the agent
                success_rate, avg_steps = evaluate_agent(agent, episodes=100)
                avg_reward = np.mean(rewards)

                # Store results
                test_results.append({
                    'alpha': alpha,
                    'gamma': gamma,
                    'epsilon': epsilon,
                    'success_rate': success_rate,
                    'avg_steps': avg_steps,
                    'avg_reward': avg_reward,
                    'rewards': rewards,
                    'steps': steps
                })

                # Plot the training performance for the current test case
                plot_rewards_and_steps(rewards, steps, f'Q-Learning (α={alpha}, γ={gamma}, ε={epsilon})')

                # Print results
                print(f"Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon} => Success Rate: {success_rate:.2f}%, Avg Steps: {avg_steps:.2f}, Avg Reward: {avg_reward:.2f}")

    # Plot comparison of average rewards across all parameter sets
    plot_parameter_comparison(test_results)

if __name__ == "__main__":
    run_test_cases()  # Execute the test cases


