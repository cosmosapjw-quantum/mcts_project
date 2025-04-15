# self_play.py
import mcts_py
import torch
import torch.nn.functional as F
import numpy as np
import threading
# Import the enhanced model class
from model import EnhancedGomokuNet

import time
import sys

# Create a debug print function
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

# Store the history parameter for easy access in the callback
net.num_history_moves = num_history_moves

# Add this to the top of self_play.py
USE_DUMMY_INFERENCE = False  # Set to False later for real inference

def my_inference_callback(batch_input):
    """
    Enhanced callback for efficient batch inference with proper error handling
    
    Parameters:
    -----------
    batch_input : list of tuples
        Each tuple contains (state_str, chosen_move, attack, defense)
    
    Returns:
    --------
    list of tuples
        Each tuple contains (policy, value)
    """
    batch_size = len(batch_input)
    debug_print(f"Inference callback received batch of size {batch_size}")
    
    start_time = time.time()
    
    try:
        # If USE_DUMMY_INFERENCE is enabled, return fast dummy values
        if USE_DUMMY_INFERENCE:
            debug_print("Using dummy inference mode")
            outputs = []
            for _ in range(batch_size):
                policy = [1.0/225] * 225  # Uniform policy for 15x15 board
                value = 0.0  # Neutral value
                outputs.append((policy, value))
            
            elapsed = time.time() - start_time
            debug_print(f"Dummy inference completed in {elapsed:.3f}s")
            return outputs
        
        # Get the history moves parameter from the net
        num_history_moves = getattr(net, 'num_history_moves', 3)
        
        debug_print(f"Processing batch with {num_history_moves} history moves")
        
        # Calculate input dimension with history moves
        # board_size*board_size (board) + 1 (player flag) + 2*num_history_moves (history) + 2 (attack/defense)
        input_dim = board_size*board_size + 1 + 2*num_history_moves + 2
        
        # Build input tensor
        x_input = np.zeros((batch_size, input_dim), dtype=np.float32)
        
        # Track any parsing errors
        parsing_errors = []
        
        # Process each input in the batch
        for i, (state_str, chosen_move, attack, defense) in enumerate(batch_input):
            try:
                debug_print(f"Processing input {i}/{batch_size}: move={chosen_move}, attack={attack:.3f}, defense={defense:.3f}")
                
                # Parse the board state from state_str
                board_info = {}
                state_string = None
                current_moves_list = []
                opponent_moves_list = []
                
                # Split the string by semicolons
                parts = state_str.split(';')
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        if key == 'State':
                            state_string = value
                        elif key == 'CurrentMoves':
                            if value:
                                current_moves_list = [int(m) for m in value.split(',') if m]
                        elif key == 'OpponentMoves':
                            if value:
                                opponent_moves_list = [int(m) for m in value.split(',') if m]
                        elif key in ['Attack', 'Defense']:
                            # These are already provided as separate parameters
                            pass
                        else:
                            board_info[key] = value
                
                # Get the board size and current player
                bs = int(board_info.get('Board', str(board_size)))
                current_player = int(board_info.get('Player', '1'))
                
                # Create the board representation (flattened)
                board_array = np.zeros(bs*bs, dtype=np.float32)
                
                # Fill the board array from the state string
                if state_string and len(state_string) == bs*bs:
                    for j, c in enumerate(state_string):
                        cell_value = int(c)
                        if cell_value == current_player:
                            board_array[j] = 1.0  # Current player's stone
                        elif cell_value != 0:
                            board_array[j] = -1.0  # Opponent's stone (normalized to -1)
                
                # Fill the input tensor with board state
                x_input[i, :bs*bs] = board_array
                
                # Add player flag (1.0 for BLACK=1, 0.0 for WHITE=2)
                x_input[i, bs*bs] = 1.0 if current_player == 1 else 0.0
                
                # Add previous moves for current player (normalize positions)
                offset = bs*bs + 1
                for j, prev_move in enumerate(current_moves_list[:num_history_moves]):
                    if prev_move >= 0 and j < num_history_moves:  # Valid move and within history limit
                        # Normalize the move position
                        x_input[i, offset + j] = float(prev_move) / (bs*bs)
                
                # Add previous moves for opponent
                offset = bs*bs + 1 + num_history_moves
                for j, prev_move in enumerate(opponent_moves_list[:num_history_moves]):
                    if prev_move >= 0 and j < num_history_moves:  # Valid move and within history limit
                        # Normalize the move position
                        x_input[i, offset + j] = float(prev_move) / (bs*bs)
                
                # Add attack and defense scores (normalized)
                x_input[i, -2] = min(max(attack, -1.0), 1.0)  # Clamp to [-1, 1]
                x_input[i, -1] = min(max(defense, -1.0), 1.0)  # Clamp to [-1, 1]
                
            except Exception as e:
                debug_print(f"Error parsing input {i}: {str(e)}")
                parsing_errors.append(i)
                # Continue with next input, this one will get default values later
        
        # If we had parsing errors for all inputs, return defaults
        if len(parsing_errors) == batch_size:
            debug_print("All inputs had parsing errors, returning defaults")
            outputs = []
            for _ in range(batch_size):
                policy = [1.0/225] * 225
                value = 0.0
                outputs.append((policy, value))
            return outputs
        
        debug_print(f"Converting input tensor to PyTorch and running neural network")
        
        # Convert numpy array to PyTorch tensor and move to device
        with torch.no_grad():
            t_input = torch.from_numpy(x_input).to(device)
            
            # Forward pass through the neural network
            try:
                policy_logits, value_out = net(t_input)
                
                # Move results back to CPU for returning to C++
                policy_logits = policy_logits.cpu()
                value_out = value_out.cpu()
            except Exception as e:
                debug_print(f"Error in neural network forward pass: {str(e)}")
                # Return defaults
                outputs = []
                for _ in range(batch_size):
                    policy = [1.0/225] * 225
                    value = 0.0
                    outputs.append((policy, value))
                return outputs
        
        # Convert to probabilities using softmax
        policy_probs = F.softmax(policy_logits, dim=1).numpy()
        values = value_out.squeeze(-1).numpy()
        
        # Build output
        outputs = []
        for i in range(batch_size):
            if i in parsing_errors:
                # Use default values for inputs that had parsing errors
                policy = [1.0/225] * 225
                value = 0.0
            else:
                policy = policy_probs[i].tolist()
                value = float(values[i])
            outputs.append((policy, value))
        
        elapsed = time.time() - start_time
        debug_print(f"Batch inference completed in {elapsed:.3f}s ({elapsed/batch_size:.4f}s per input)")
        
        return outputs
    
    except Exception as e:
        # Catch-all error handler
        import traceback
        debug_print(f"Unexpected error in inference callback: {str(e)}")
        debug_print(traceback.format_exc())
        
        # Return reasonable defaults
        outputs = []
        for _ in range(batch_size):
            policy = [1.0/225] * 225
            value = 0.0
            outputs.append((policy, value))
        
        return outputs

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
    cfg.num_threads = max(1, min(available_cores - 2, 8))
    
    # Set batch size for leaf parallelization
    cfg.parallel_leaf_batch_size = 16  # Larger batches for better GPU utilization
    
    debug_print(f"MCTS configuration: {cfg.num_simulations} simulations, "
               f"{cfg.num_threads} threads, {cfg.parallel_leaf_batch_size} batch size, "
               f"c_puct={cfg.c_puct}")
    
    # Create wrapper
    debug_print("Creating MCTS wrapper")
    wrapper = mcts_py.MCTSWrapper(cfg, boardSize=board_size)
    
    # Set neural network callback and batch size
    debug_print("Setting inference function and batch size")
    wrapper.set_infer_function(my_inference_callback)
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
        
        # Record move in history for training data
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
    num_games = 10  # Number of games to play for training data
    
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