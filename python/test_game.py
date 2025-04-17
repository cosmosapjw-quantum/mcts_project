#!/usr/bin/env python3
# Test game using the redesigned handle-based architecture

import sys
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import mcts_py
import threading
import random
import signal
import os
import atexit

# Set debug flag
os.environ["DEBUG_MCTS"] = "1"

# Test class for neural network
class GomokuModel(nn.Module):
    def __init__(self, board_size=15, num_history_moves=3):
        super(GomokuModel, self).__init__()
        self.board_size = board_size
        self.num_history_moves = num_history_moves
        
        # Calculate input dimension
        input_dim = board_size * board_size + 1 + 2 * num_history_moves + 2
        
        # Create a simple network
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Policy head
        self.policy_head = nn.Linear(128, board_size * board_size)
        
        # Value head
        self.value_fc = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)
    
    def forward(self, x):
        # Common layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Policy head
        policy_logits = self.policy_head(x)
        
        # Value head
        value = F.relu(self.value_fc(x))
        value = torch.tanh(self.value_head(value))
        
        return policy_logits, value

# Create the model
print("Creating neural network model")
model = GomokuModel()

# Register signal handlers for graceful termination
mcts_py.register_signal_handlers()

# Create a simple configuration with minimal resources
print("Creating MCTS configuration")
cfg = mcts_py.MCTSConfig()
cfg.num_simulations = 200     # Reduced number for quick test
cfg.c_puct = 1.5
cfg.num_threads = 2           # Use 2 threads
cfg.parallel_leaf_batch_size = 8  # Small batch size for testing

# Create MCTS wrapper
print("Creating MCTS wrapper")
wrapper = mcts_py.MCTSWrapper(cfg, 15, False, False, 0, False)

# Set the model - this now uses the handle-based approach
print("Setting neural network model through handle-based interface")
wrapper.set_infer_function(model)

# Clean exit handler
def handle_exit():
    print("Clean exit handler called")
    global wrapper, model
    
    # Release wrapper first - this is critical
    if wrapper is not None:
        print("Releasing MCTS wrapper")
        temp_wrapper = wrapper
        wrapper = None
        # Make sure PythonNNProxy is properly shutdown first
        if hasattr(temp_wrapper, 'shutdown_nn'):
            print("Calling explicit shutdown_nn")
            temp_wrapper.shutdown_nn()
        # Delete the wrapper reference
        del temp_wrapper
    
    # Give C++ side time to clean up
    time.sleep(0.5)
    
    # Release model
    if model is not None:
        print("Releasing model")
        temp_model = model
        model = None
        del temp_model
    
    # Force GC multiple times to ensure cleanup
    for i in range(3):
        gc.collect()
        time.sleep(0.1)
    
    print("Clean exit completed")

# Register clean exit handler
atexit.register(handle_exit)

# Also handle SIGINT and SIGTERM for more graceful shutdown
def signal_handler(sig, frame):
    print(f"Received signal {sig}, initiating clean shutdown")
    handle_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Helper to print the board
def print_board(moves, board_size=15):
    # Create an empty board
    board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
    
    # Fill in the moves
    player = 1  # Player 1 starts
    for move in moves:
        if move < 0 or move >= board_size * board_size:
            continue
        row = move // board_size
        col = move % board_size
        
        # Set the piece
        board[row][col] = 'X' if player == 1 else 'O'
        
        # Switch player
        player = 3 - player  # 1 -> 2, 2 -> 1
    
    # Print column indices
    print("   ", end="")
    for col in range(board_size):
        print(f"{col:2d}", end=" ")
    print()
    
    # Print horizontal line
    print("  +" + "-" * (board_size * 3 + 1))
    
    # Print rows
    for row in range(board_size):
        print(f"{row:2d}|", end=" ")
        for col in range(board_size):
            print(f"{board[row][col]} ", end=" ")
        print("|")
    
    # Print horizontal line
    print("  +" + "-" * (board_size * 3 + 1))

# Play a game
moves = []
print("\nStarting a Gomoku game using the handle-based architecture...")

# Play for a maximum of 10 moves
for i in range(10):
    print(f"\nMove {i+1} (Player {wrapper.get_current_player()})")
    
    # Run search with multiple attempts if needed
    max_attempts = 3
    move = -1
    
    for attempt in range(max_attempts):
        wrapper.run_search()
        
        # Get best move
        move = wrapper.best_move()
        print(f"Best move attempt {attempt+1}: {move}")
        
        if move >= 0:
            break  # Got a valid move
        elif attempt < max_attempts - 1:
            print(f"Invalid move returned, retrying search... (attempt {attempt+1}/{max_attempts})")
            time.sleep(0.5)  # Wait a bit before retrying
    
    if move >= 0:
        # Apply the move
        wrapper.make_move(move)
        moves.append(move)
        
        # Print the board
        print_board(moves)
        
        # Check if game is over
        if wrapper.is_terminal():
            winner = wrapper.get_winner()
            print(f"\nGame over! Winner: Player {winner}" if winner > 0 else "\nGame over! Draw")
            break
    else:
        print("Failed to get a valid move after multiple attempts. Stopping game.")
        break

print("\nGame completed successfully!")

# Let Python's GC and our clean exit handler take care of shutdown
print("Exiting normally...")