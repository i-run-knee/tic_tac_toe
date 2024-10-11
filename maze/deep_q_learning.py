import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

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

class ReplayBuffer:
    """Replay buffer for storing experiences."""
    def __init__(self, max_size=2000):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        """Add experience to the buffer."""
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)  # Remove oldest experience
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)

    @property
    def size(self):
        """Get current size of the buffer."""
        return len(self.buffer)

class DeepQLearningAgent:
    """Deep Q-Learning agent for navigating rooms."""
    def __init__(self, alpha=0.001, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.replay_buffer = ReplayBuffer(max_size=2000)
        self.model = self._build_model()  # Neural network model

    def _build_model(self):
        """Build the neural network model."""
        model = keras.Sequential()
        model.add(layers.Dense(24, input_dim=len(rooms), activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(len(actions), activation='linear'))  # Output layer for Q-values
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.alpha))
        return model

    def get_action(self, state):
        """Select action using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:  # Explore
            return np.random.choice(len(actions))
        q_values = self.model.predict(state)  # Predict Q-values
        return np.argmax(q_values[0])  # Exploit

    def train(self, episodes=1000):
        """Train the Deep Q-Learning agent over a specified number of episodes."""
        rewards_per_episode = []
        steps_per_episode = []

        for episode in range(episodes):
            state_index = random.randint(0, len(rooms) - 1)  # Start from a random room
            total_reward = 0
            steps = 0
            done = False

            while not done:
                state = np.zeros(len(rooms))
                state[state_index] = 1  # One-hot encoding for state representation
                action = self.get_action(state.reshape(1, -1))  # Choose action

                next_room = room_connections[rooms[state_index]][action % len(room_connections[rooms[state_index]])]
                next_state_index = rooms.index(next_room)  # Get next state index
                reward = rewards[next_room]  # Get reward for next room

                # Store experience in replay buffer
                self.replay_buffer.add((state, action, reward, next_state_index, done))

                # Experience replay
                if self.replay_buffer.size >= 32:  # Minimum batch size for training
                    batch = self.replay_buffer.sample(32)
                    for state, action, reward, next_state_index, done in batch:
                        target = reward
                        if not done:
                            target += self.gamma * np.max(self.model.predict(np.eye(len(rooms))[next_state_index].reshape(1, -1)))
                        target_f = self.model.predict(state.reshape(1, -1))
                        target_f[0][action] = target  # Update target for Q-value
                        self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)  # Train the model

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
            state = np.zeros(len(rooms))
            state[state_index] = 1  # One-hot encoding for state representation
            action = agent.get_action(state.reshape(1, -1))  # Choose action

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

def plot_rewards_and_steps(rewards, steps, title='Deep Q-Learning: Rewards and Steps per Episode'):
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
    """Plot average rewards for different parameter sets."""
    plt.figure(figsize=(10, 6))

    # Extract data for plotting
    alphas = [result['alpha'] for result in test_results]
    gammas = [result['gamma'] for result in test_results]
    epsilons = [result['epsilon'] for result in test_results]
    avg_rewards = [result['avg_reward'] for result in test_results]

    # Plotting average rewards for each parameter combination
    for idx in range(len(test_results)):
        plt.scatter(f"α={alphas[idx]}, γ={gammas[idx]}, ε={epsilons[idx]}", avg_rewards[idx], label=f'α={alphas[idx]}, γ={gammas[idx]}, ε={epsilons[idx]}')

    plt.title('Average Rewards for Different Deep Q-Learning Parameter Sets')
    plt.xlabel('Parameter Sets')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_test_cases():
    """Run test cases with varying hyperparameters."""
    test_results = []

    for alpha in [0.001, 0.01]:
        for gamma in [0.5, 0.9]:
            for epsilon in [0.1, 0.3]:
                # Train Deep Q-Learning Agent
                agent = DeepQLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
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
                plot_rewards_and_steps(rewards, steps, f'Deep Q-Learning (α={alpha}, γ={gamma}, ε={epsilon})')

                # Print results
                print(f"Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon} => Success Rate: {success_rate:.2f}%, Avg Steps: {avg_steps:.2f}, Avg Reward: {avg_reward:.2f}")

    # Plot comparison of average rewards across all parameter sets
    plot_parameter_comparison(test_results)

if __name__ == "__main__":
    run_test_cases()  # Execute the test cases


