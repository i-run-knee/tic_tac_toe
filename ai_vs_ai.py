import numpy as np
import time
import sys
from ai_player import check_win, check_tie, minimax_ai_move, q_learning_ai_move

def print_board(board):
    """Utility function to print the game board."""
    print("\nCurrent board:")
    for i in range(3):
        print(" | ".join(board[i*3:(i+1)*3]))
        if i < 2:
            print("---------")
    print()

def play_game():
    """Simulate a game between Minimax AI and Q-Learning AI."""
    board = [' ' for _ in range(9)]
    current_player = 'minimax'  # Start with Minimax AI

    while True:
        if current_player == 'minimax':
            move = minimax_ai_move(board)
            board[move] = 'O'  # Minimax is 'O'
        else:
            move = q_learning_ai_move(board, epsilon=0)  # Use epsilon=0 for exploitation
            board[move] = 'X'  # Q-Learning is 'X'

        # Uncomment the line below if you want to see the board after each move
        # print_board(board)

        if check_win(board, 'O'):
            return 'minimax'
        elif check_win(board, 'X'):
            return 'q_learning'
        elif check_tie(board):
            return 'tie'

        # Switch players
        current_player = 'q_learning' if current_player == 'minimax' else 'minimax'

def run_simulation(num_games=100):
    """Run multiple simulations and collect statistics."""
    minimax_wins = 0
    q_learning_wins = 0
    ties = 0
    total_time = 0

    for game in range(num_games):
        start_time = time.time()  # Start timing the game
        result = play_game()
        game_time = time.time() - start_time  # Calculate the time taken for this game
        total_time += game_time  # Accumulate total time

        if result == 'minimax':
            minimax_wins += 1
        elif result == 'q_learning':
            q_learning_wins += 1
        else:
            ties += 1

        # Periodic update every 10 games
        if (game + 1) % 10 == 0:
            print(f"Completed {game + 1}/{num_games} games... ", end="\r")

    avg_game_time = total_time / num_games  # Average time per game

    print(f"\nResults after {num_games} games:")
    print(f"Minimax AI wins: {minimax_wins}")
    print(f"Q-Learning AI wins: {q_learning_wins}")
    print(f"Ties: {ties}")
    print(f"Total simulation time: {total_time:.2f} seconds")
    print(f"Average time per game: {avg_game_time:.2f} seconds")

if __name__ == "__main__":
    run_simulation(100)

