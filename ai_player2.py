import random

def check_win(board, player):
    """Helper function to check if a player can win."""
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # Horizontal wins
                      (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Vertical wins
                      (0, 4, 8), (2, 4, 6)]             # Diagonal wins
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
            return True
    return False

def random_ai_move(board):
    """AI makes a random move."""
    available_moves = [i for i, spot in enumerate(board) if spot == ' ']
    return random.choice(available_moves)

def greedy_ai(board):
# get all available positions
# for every position:
# update board to board[pos] = 'O'
# check_win()
# if true return pos
# else return random_pos
    available_positions = []
    for i in range(9):
        if board[i] == ' ':
            available_positions.append(i)
    for position in available_positions:
        board[position] = 'O'
        if check_win(board, 'O'):
            board[position] = ' '
            return position
        else:
            board[position] = ' '
    return defensive_ai(board)


def defensive_ai(board):
    available_positions = []
    for i in range(9):
        if board[i] == ' ':
            available_positions.append(i)
    for position in available_positions:
        board[position] = 'X'
        if check_win(board, 'X'):
            board[position] = ' '
            return position
        else:
            board[position] = ' '
    return random_ai_move(board)