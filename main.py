import tkinter as tk
from tkinter import messagebox
from ai_player import random_ai_move, greedy_ai_move, blocking_ai_move, minimax_ai_move, get_ai_move, EPSILON, train_q_learning

# Function to check for a win
def check_win(board, player):
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # Horizontal wins
                      (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Vertical wins
                      (0, 4, 8), (2, 4, 6)]             # Diagonal wins
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
            return True
    return False

# Function to check for a tie
def check_tie(board):
    return ' ' not in board

# Button click handler
def on_click(index):
    global current_player

    # If the spot is already taken or the game is over, do nothing
    if board[index] != ' ' or game_over:
        return

    # Mark the board and update button text
    board[index] = current_player
    buttons[index].config(text=current_player)

    # Check for a win or tie
    if check_win(board, current_player):
        messagebox.showinfo("Game Over", f"{current_player} wins!")
        reset_game()
        return
    elif check_tie(board):
        messagebox.showinfo("Game Over", "It's a tie!")
        reset_game()
        return

    # Switch player
    current_player = 'O' if current_player == 'X' else 'X'

    # AI's turn if it's 'O'
    if current_player == 'O':
        ai_turn()

# AI's turn to play
def ai_turn():
    global current_player
    move = get_ai_move(board, ai_algorithm, EPSILON)  # Use the selected AI algorithm
    board[move] = 'O'
    buttons[move].config(text='O')

    # Check for a win or tie
    if check_win(board, 'O'):
        messagebox.showinfo("Game Over", "AI (O) wins!")
        reset_game()
    elif check_tie(board):
        messagebox.showinfo("Game Over", "It's a tie!")
        reset_game()
    else:
        current_player = 'X'

# Reset the game
def reset_game():
    global board, game_over, current_player
    board = [' ' for _ in range(9)]
    current_player = 'X'
    game_over = False
    for button in buttons:
        button.config(text='')

# Function to choose the AI algorithm
def choose_ai(algorithm):
    global ai_algorithm
    ai_algorithm = algorithm
    root.deiconify()  # Show the main window
    ai_choice_window.destroy()  # Close the AI selection window

# Training mode function
def start_training():
    train_q_learning(num_games=10000)  # Run the Q-learning training
    messagebox.showinfo("Training Complete", "Q-Learning training is complete!")
    reset_game()

# Create the AI selection window
ai_choice_window = tk.Tk()
ai_choice_window.title("Choose AI Player")

tk.Label(ai_choice_window, text="Choose the AI strategy").pack(pady=10)

random_button = tk.Button(ai_choice_window, text="Random AI", command=lambda: choose_ai('random'))
random_button.pack(pady=5)

greedy_button = tk.Button(ai_choice_window, text="Greedy AI", command=lambda: choose_ai('greedy'))
greedy_button.pack(pady=5)

blocking_button = tk.Button(ai_choice_window, text="Blocking AI", command=lambda: choose_ai('blocking'))
blocking_button.pack(pady=5)

minimax_button = tk.Button(ai_choice_window, text="Minimax AI", command=lambda: choose_ai('minimax'))
minimax_button.pack(pady=5)

q_learning_button = tk.Button(ai_choice_window, text="Q-Learning AI", command=lambda: choose_ai('q_learning'))
q_learning_button.pack(pady=5)

training_button = tk.Button(ai_choice_window, text="Train Q-Learning", command=start_training)
training_button.pack(pady=20)

# Hide the main game window until AI is selected
root = tk.Tk()
root.withdraw()

# Initialize game variables
current_player = 'X'
game_over = False
board = [' ' for _ in range(9)]

# Create the buttons for the grid
buttons = []
for i in range(9):
    button = tk.Button(root, text='', font=('normal', 40), width=5, height=2,
                       command=lambda i=i: on_click(i))
    button.grid(row=i // 3, column=i % 3)
    buttons.append(button)

# Start the AI selection window
ai_choice_window.mainloop()
