import tkinter as tk
from tkinter import messagebox
import random
import numpy as np

# Initialize the main window
root = tk.Tk()
root.title("Tic-Tac-Toe")
root.geometry("400x400")

# Define the game board
board = [[" " for _ in range(3)] for _ in range(3)]
current_player = "X"

# Q-learning or any AI function placeholder
def choose_action(board, epsilon):
    empty_spots = [(i, j) for i in range(3) for j in range(3) if board[i][j] == " "]
    return random.choice(empty_spots)

# Create a 3x3 grid of buttons
buttons = [[None for _ in range(3)] for _ in range(3)]

def create_grid():
    for i in range(3):
        for j in range(3):
            buttons[i][j] = tk.Button(root, text=" ", font=('normal', 40, 'normal'), width=5, height=2,
                                      command=lambda i=i, j=j: on_click(i, j))
            buttons[i][j].grid(row=i, column=j)

def on_click(i, j):
    global current_player

    if board[i][j] == " ":
        # Update board and button text
        board[i][j] = current_player
        buttons[i][j].config(text=current_player)

        # Check for a winner or draw
        if check_winner(current_player):
            messagebox.showinfo("Tic-Tac-Toe", f"Player {current_player} wins!")
            reset_game()
        elif check_draw():
            messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
            reset_game()
        else:
            # Switch player
            current_player = "O" if current_player == "X" else "X"
            if current_player == "O":
                ai_move()  # AI makes its move

def check_winner(player):
    for i in range(3):
        if all([board[i][j] == player for j in range(3)]) or all([board[j][i] == player for j in range(3)]):
            return True
    if board[0][0] == board[1][1] == board[2][2] == player or board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False

def check_draw():
    return all(board[i][j] != " " for i in range(3) for j in range(3))

def reset_game():
    global board, current_player
    board = [[" " for _ in range(3)] for _ in range(3)]
    current_player = "X"
    create_grid()  # Reset the buttons

def ai_move():
    global current_player

    # AI chooses its move
    row, col = choose_action(board, epsilon=0)
    board[row][col] = current_player
    buttons[row][col].config(text=current_player)

    # Check for a winner or draw
    if check_winner(current_player):
        messagebox.showinfo("Tic-Tac-Toe", f"Player {current_player} wins!")
        reset_game()
    elif check_draw():
        messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
        reset_game()
    else:
        current_player = "X"

# Initialize the grid
create_grid()

# Start the Tkinter event loop
root.mainloop()
