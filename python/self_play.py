# self_play.py
import mcts_py
import torch
import torch.nn.functional as F
import numpy as np
import threading
import time
import sys
import re
# Import the enhanced model class
from model import EnhancedGomokuNet
# Import our new nn_proxy module
import nn_proxy

def debug_print(message):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[DEBUG {timestamp}] {message}", flush=True)

# A toy in-memory buffer for demonstration
global_data_buffer = []

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the neural network with configurable history moves parameter
board_size = 15
policy_dim = board_size * board_size
num_history_moves = 7  # Configure this as needed

# Create the enhanced network
net = EnhancedGomokuNet(board_size=board_size, policy_dim=policy_dim, num_history_moves=num_history_moves)
net.to(device)

# Store the history parameter for easy access
net.num_history_moves = num_history_moves

def self_play_game():
    debug_print("Starting self-play game")
    
    # Configure MCTS with optimized parameters for leaf parallelization
    debug_print("Configuring MCTS")
    cfg = mcts_py.MCTSConfig()
    
    # Set simulation count based on desired quality
    cfg.num_simulations = 400  # Increased for better play quality
    
    # Set exploration parameter
    cfg.c_puct = 1.5  # Slightly increased for more exploration
    
    # Set thread count based on available CPU cores
    import multiprocessing
    available_cores = multiprocessing.cpu_count()
    # Reserve 1 core for Python/neural network and 1 for system
    cfg.num_threads = 16 #max(1, min(available_cores - 2, 8))
    
    # Set batch size for leaf parallelization
    cfg.parallel_leaf_batch_size = 128  # Larger batches for better GPU utilization
    
    debug_print(f"MCTS configuration: {cfg.num_simulations} simulations, "
               f"{cfg.num_threads} threads, {cfg.parallel_leaf_batch_size} batch size, "
               f"c_puct={cfg.c_puct}")
    
    # Create wrapper
    debug_print("Creating MCTS wrapper")
    wrapper = mcts_py.MCTSWrapper(cfg, boardSize=board_size)
    
    # Pass the neural network model directly to the wrapper
    debug_print("Setting neural network model")
    wrapper.set_infer_function(net)
    
    # Set batch size
    wrapper.set_batch_size(cfg.parallel_leaf_batch_size)
    
    # Configure the history moves to match our neural network
    debug_print(f"Setting history moves to {num_history_moves}")
    wrapper.set_num_history_moves(num_history_moves)
    
    # Set exploration parameters - higher Dirichlet noise for more diverse self-play
    debug_print("Setting exploration parameters")
    wrapper.set_exploration_parameters(dirichlet_alpha=0.3, noise_weight=0.25)
    
    # Data structures for collecting training data
    states_actions = []
    attack_defense_values = []
    history_moves = []  # Track move history for training
    move_count = 0
    start_time = time.time()
    
    # Temperature schedule for move selection
    def get_temperature(move_num):
        if move_num < 15:
            return 1.0  # High temperature for first 15 moves (exploration)
        elif move_num < 30:
            return 0.5  # Medium temperature for next 15 moves
        else:
            return 0.1  # Low temperature for remaining moves (exploitation)
    
    # Main game loop
    while not wrapper.is_terminal() and move_count < board_size*board_size:
        debug_print(f"\nMove {move_count}: Running MCTS search")
        search_start = time.time()
        
        # Run MCTS search with timeout protection
        try:
            # Use a timeout to avoid hanging games
            max_search_time = 60  # seconds
            search_thread = threading.Thread(target=wrapper.run_search)
            search_thread.daemon = True
            search_thread.start()
            search_thread.join(timeout=max_search_time)
            
            if search_thread.is_alive():
                debug_print(f"WARNING: Search timeout reached ({max_search_time}s)")
                # We can't stop the search directly, but we'll proceed anyway
            
            search_elapsed = time.time() - search_start
            debug_print(f"MCTS search completed in {search_elapsed:.3f} seconds")
            
            # Print stats
            stats = wrapper.get_stats()
            debug_print("Search statistics:\n" + stats)
        except Exception as e:
            debug_print(f"Error during MCTS search: {e}")
            break
        
        # Get temperature for current move
        temp = get_temperature(move_count)
        debug_print(f"Using temperature {temp} for move selection")
        
        # Select best move with temperature
        best_move_start = time.time()
        mv = wrapper.best_move_with_temperature(temp)
        best_move_elapsed = time.time() - best_move_start
        
        if mv < 0:
            debug_print(f"Invalid move returned: {mv}, ending game")
            break
            
        # Convert to board coordinates for more readable output
        x, y = mv // board_size, mv % board_size
        debug_print(f"Best move: {mv} ({x},{y}) found in {best_move_elapsed:.3f}s")
        
        move_count += 1
        
        # Record move in history for training
        history_moves.append(mv)
        
        # Get the board state for training
        try:
            # In a real implementation, we'd get the actual board state here
            # For now, we'll just use a placeholder
            board_state = None
            
            # Ideally, we'd get attack and defense values directly from the wrapper
            # For now we'll use placeholders
            attack = 0.0
            defense = 0.0
            
            # Store the current position, move, and history data
            current_player_history = history_moves[::2][-num_history_moves:] if move_count % 2 == 1 else history_moves[1::2][-num_history_moves:]
            opponent_history = history_moves[1::2][-num_history_moves:] if move_count % 2 == 1 else history_moves[::2][-num_history_moves:]
            
            states_actions.append((board_state, mv, current_player_history, opponent_history))
            attack_defense_values.append((attack, defense))
        except Exception as e:
            debug_print(f"Error recording game state: {e}")
        
        # Apply the move
        debug_print(f"Applying move {mv}")
        apply_start = time.time()
        wrapper.apply_best_move_with_temperature(temp)
        apply_elapsed = time.time() - apply_start
        debug_print(f"Move applied in {apply_elapsed:.3f}s")
    
    # Game results statistics
    total_elapsed = time.time() - start_time
    debug_print("\nGame finished")
    w = wrapper.get_winner()
    winner_str = "BLACK" if w == 1 else "WHITE" if w == 2 else "DRAW"
    debug_print(f"Winner: {w} ({winner_str})")
    
    # Fix for division by zero - check if moves were made
    if move_count > 0:
        debug_print(f"Game completed with {move_count} moves in {total_elapsed:.1f}s "
                  f"({total_elapsed/move_count:.1f}s per move)")
    else:
        debug_print(f"Game completed with no moves in {total_elapsed:.1f}s")
    
    # Store to global data buffer with enhanced data
    debug_print("Adding game data to training buffer")
    for i, (st, mv, curr_hist, opp_hist) in enumerate(states_actions):
        attack, defense = attack_defense_values[i]
        # Include both player's move history in the training data
        global_data_buffer.append((st, mv, w, attack, defense, curr_hist, opp_hist))
    
    debug_print(f"Added {len(states_actions)} positions to training buffer")
    return w

def main():
    debug_print("Starting self-play data generation")
    
    # Configure and print system information
    import torch
    num_games = 4  # Number of games to play for training data
    
    debug_print(f"System configuration:")
    debug_print(f"  PyTorch version: {torch.__version__}")
    debug_print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        debug_print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    debug_print(f"  Device being used: {device}")
    debug_print(f"  Board size: {board_size}x{board_size}")
    debug_print(f"  History moves: {num_history_moves}")
    debug_print(f"  Number of games: {num_games}")
    
    # Reset global data buffer
    global global_data_buffer
    global_data_buffer = []
    
    # Game statistics
    results = {1: 0, 2: 0, 0: 0}  # BLACK, WHITE, DRAW
    total_moves = 0
    total_time = 0
    
    # Play games
    for i in range(num_games):
        debug_print(f"\n=========== Starting game {i+1}/{num_games} ===========")
        game_start = time.time()
        
        try:
            winner = self_play_game()
            results[winner] += 1
            
            game_elapsed = time.time() - game_start
            total_time += game_elapsed
            
            # Estimate moves made based on buffer positions from this game
            # Note: In a real implementation, we'd track moves per game directly
            game_positions = len([pos for pos in global_data_buffer if pos[2] == winner])  # Positions with this winner
            moves_made = max(1, game_positions)  # Ensure at least 1 to avoid division by zero
            total_moves += moves_made
            
            debug_print(f"Game {i+1} completed in {game_elapsed:.1f}s")
            if moves_made > 0:
                debug_print(f"Average time per move: {game_elapsed/moves_made:.1f}s")
        except Exception as e:
            debug_print(f"Error in game {i+1}: {e}")
            import traceback
            debug_print(traceback.format_exc())
    
    # Print overall statistics
    debug_print("\n=========== Self-play completed ===========")
    debug_print(f"Games played: {num_games}")
    debug_print(f"Results: BLACK wins: {results[1]}, WHITE wins: {results[2]}, Draws: {results[0]}")
    if num_games > 0:
        debug_print(f"Win rates: BLACK: {results[1]/num_games*100:.1f}%, WHITE: {results[2]/num_games*100:.1f}%, Draws: {results[0]/num_games*100:.1f}%")
    debug_print(f"Total positions collected: {len(global_data_buffer)}")
    if total_moves > 0:
        debug_print(f"Average moves per game: {total_moves/num_games:.1f}")
        debug_print(f"Average time per move: {total_time/total_moves:.2f}s")
    debug_print(f"Total time: {total_time:.1f}s")
    
    # In a real implementation, we would save the collected data to disk here
    debug_print("Data collection complete")

if __name__ == "__main__":
    main()