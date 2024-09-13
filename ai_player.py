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

def greedy_ai_move(board):
    """Greedy AI tries to win if possible, otherwise makes a random move."""
    available_moves = [i for i, spot in enumerate(board) if spot == ' ']

    # Step 1: Check if AI can win in the next move
    for move in available_moves:
        board[move] = 'O'  # AI is 'O'
        if check_win(board, 'O'):
            board[move] = ' '  # Reset the spot after checking
            return move  # AI wins by playing this move
        board[move] = ' '  # Reset the spot after checking

    # Step 2: No immediate win, so pick a random move
    return random.choice(available_moves)

def blocking_ai_move(board):
    """Blocking AI blocks opponent's winning move or uses greedy AI if no block is possible."""
    available_moves = [i for i, spot in enumerate(board) if spot == ' ']

    # Step 1: Check if the opponent can win in the next move
    for move in available_moves:
        board[move] = 'X'  # Opponent is 'X'
        if check_win(board, 'X'):
            board[move] = ' '  # Reset the spot after checking
            return move  # Block the opponent's winning move
        board[move] = ' '  # Reset the spot after checking

    # Step 2: If no block, use greedy AI strategy
    return greedy_ai_move(board)

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

