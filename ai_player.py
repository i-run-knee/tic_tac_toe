import random
import numpy as np
import csv

def check_win(board, player):
    """Helper function to check if a player can win."""
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # Horizontal wins
                      (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Vertical wins
                      (0, 4, 8), (2, 4, 6)]             # Diagonal wins
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
            return True
    return False

def check_tie(board):
    """Helper function to check if the game is a tie."""
    return ' ' not in board and not any(check_win(board, p) for p in ['X', 'O'])

def random_ai_move(board):
    """AI makes a random move."""
    available_moves = [i for i, spot in enumerate(board) if spot == ' ']
    return random.choice(available_moves)

def greedy_ai_move(board):
    """Greedy AI makes the best possible move to win or block."""
    available_positions = [i for i in range(9) if board[i] == ' ']
    for position in available_positions:
        board[position] = 'O'  # AI is 'O'
        if check_win(board, 'O'):
            return position
        board[position] = ' '  # Reset if not a winning move
    return blocking_ai_move(board)

def blocking_ai_move(board):
    """Blocking AI blocks opponent's winning move or uses greedy AI if no block is possible."""
    available_positions = [i for i in range(9) if board[i] == ' ']
    for position in available_positions:
        board[position] = 'X'  # Opponent is 'X'
        if check_win(board, 'X'):
            return position  # Block the opponent's winning move
        board[position] = ' '  # Reset if not a blocking move
    return random_ai_move(board)

def minimax(board, depth, is_maximizing):
    """Minimax algorithm to choose the best move for AI."""
    # Terminal states
    if check_win(board, 'O'):
        return 10 - depth
    if check_win(board, 'X'):
        return depth - 10
    if ' ' not in board:
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for move in [i for i, spot in enumerate(board) if spot == ' ']:
            board[move] = 'O'
            score = minimax(board, depth + 1, False)
            board[move] = ' '
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for move in [i for i, spot in enumerate(board) if spot == ' ']:
            board[move] = 'X'
            score = minimax(board, depth + 1, True)
            board[move] = ' '
            best_score = min(score, best_score)
        return best_score

def minimax_ai_move(board):
    """Minimax AI chooses the best move based on minimax evaluation."""
    best_move = None
    best_score = -float('inf')

    for move in [i for i, spot in enumerate(board) if spot == ' ']:
        board[move] = 'O'
        score = minimax(board, 0, False)
        board[move] = ' '
        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def board_to_state_index(board):
    """Convert a board state to a unique state index."""
    state_str = ''.join(['2' if spot == 'O' else '1' if spot == 'X' else '0' for spot in board])
    return int(state_str, 3)

def q_learning_ai_move(board, epsilon):
    """Q-Learning AI chooses the best move based on Q-learning."""
    if not hasattr(q_learning_ai_move, "Q"):
        q_learning_ai_move.Q = load_q_table()  # Load the Q-table
        q_learning_ai_move.learning_rate = 0.1
        q_learning_ai_move.discount_factor = 0.9
        q_learning_ai_move.exploration_rate = epsilon
        q_learning_ai_move.last_state = None
        q_learning_ai_move.last_action = None

    state_index = board_to_state_index(board)

    if random.random() < q_learning_ai_move.exploration_rate:
        available_moves = [i for i, spot in enumerate(board) if spot == ' ']
        action = random.choice(available_moves)
    else:
        q_values = q_learning_ai_move.Q[state_index]
        available_moves = [i for i, spot in enumerate(board) if spot == ' ']
        action = max(available_moves, key=lambda move: q_values[move])

    if q_learning_ai_move.last_state is not None:
        reward = 0
        if check_win(board, 'O'):
            reward = 10  # Win reward
        elif check_win(board, 'X'):
            reward = -10  # Lose penalty
        elif check_tie(board):
            reward = 5  # Tie reward
        else:
            reward = -1  # Small penalty for non-winning moves

        last_state_index = board_to_state_index(q_learning_ai_move.last_state)
        last_action = q_learning_ai_move.last_action

        old_q_value = q_learning_ai_move.Q[last_state_index][last_action]
        best_future_q = max(q_learning_ai_move.Q[state_index])
        new_q_value = old_q_value + q_learning_ai_move.learning_rate * (reward + q_learning_ai_move.discount_factor * best_future_q - old_q_value)
        q_learning_ai_move.Q[last_state_index][last_action] = new_q_value

    q_learning_ai_move.last_state = board.copy()
    q_learning_ai_move.last_action = action

    return action

def save_q_table(Q, filename='q_table.csv'):
    """Save the Q-table to a CSV file."""
    np.savetxt(filename, Q, delimiter=",", fmt='%s')

def load_q_table(filename='q_table.csv'):
    """Load the Q-table from a CSV file, or initialize it if the file does not exist."""
    try:
        return np.loadtxt(filename, delimiter=",")
    except FileNotFoundError:
        return np.zeros((3**9, 9))  # Initialize a new Q-table if not found

def update_q_values(board, reward):
    """Update Q-values after a game ends."""
    state_index = board_to_state_index(board)
    last_state_index = board_to_state_index(q_learning_ai_move.last_state)
    last_action = q_learning_ai_move.last_action

    old_q_value = q_learning_ai_move.Q[last_state_index][last_action]
    best_future_q = max(q_learning_ai_move.Q[state_index])
    new_q_value = old_q_value + q_learning_ai_move.learning_rate * (reward + q_learning_ai_move.discount_factor * best_future_q - old_q_value)
    q_learning_ai_move.Q[last_state_index][last_action] = new_q_value

def train_q_learning(num_games=1000, epsilon_start=1.0, epsilon_decay=0.99):
    """Train Q-learning AI by playing games."""
    for game in range(num_games):
        board = [' ' for _ in range(9)]
        q_learning_ai_move.last_state = None
        q_learning_ai_move.last_action = None

        while True:
            # AI 'O' makes a move
            move = q_learning_ai_move(board, epsilon_start)
            board[move] = 'O'

            if check_win(board, 'O'):
                update_q_values(board, reward=10)
                break
            elif check_tie(board):
                update_q_values(board, reward=5)
                break

            # Random 'X' opponent makes a move
            move = random_ai_move(board)
            board[move] = 'X'

            if check_win(board, 'X'):
                update_q_values(board, reward=-10)
                break
            elif check_tie(board):
                update_q_values(board, reward=5)
                break

        epsilon_start *= epsilon_decay  # Decay epsilon to reduce exploration over time

    save_q_table(q_learning_ai_move.Q)  # Save the Q-table after training

def get_ai_move(board, strategy, epsilon=0.1):
    """Get the move for the AI based on the chosen strategy."""
    if strategy == 'random':
        return random_ai_move(board)
    elif strategy == 'greedy':
        return greedy_ai_move(board)
    elif strategy == 'blocking':
        return blocking_ai_move(board)
    elif strategy == 'minimax':
        return minimax_ai_move(board)
    elif strategy == 'q_learning':
        return q_learning_ai_move(board, epsilon)
    else:
        raise ValueError("Invalid AI strategy")

# Define EPSILON for exploration in Q-Learning
EPSILON = 0.1
