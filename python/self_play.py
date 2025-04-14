# self_play.py
import mcts_py
import torch
import torch.nn.functional as F
import random
import numpy as np

from model import SimpleGomokuNet

# A toy in-memory buffer for demonstration
global_data_buffer = []

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the neural network and move it to GPU if available
board_size = 15
policy_dim = board_size * board_size
net = SimpleGomokuNet(board_size=board_size, policy_dim=policy_dim)
net.to(device)  # Move model to GPU if available
print("Neural network initialized and moved to device")

def my_inference_callback(batch_input):
    """
    Enhanced callback using the actual game state
    """
    try:
        batch_size = len(batch_input)
        print(f"Processing batch of {batch_size} neural network inputs")
        
        # Convert input to proper format for neural network
        board_size = 15
        input_dim = board_size*board_size + 3
        
        # Build input tensor
        x_input = np.zeros((batch_size, input_dim), dtype=np.float32)
        
        for i, (stateStr, move, attack, defense) in enumerate(batch_input):
            # Parse the board state from stateStr
            board_info = {}
            state_string = None
            
            # Split the string by semicolons
            parts = stateStr.split(';')
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)  # Split on first colon only
                    if key == 'State':
                        # Get the board state string
                        state_string = value
                    else:
                        board_info[key] = value
            
            # Get the board size and current player
            bs = int(board_info.get('Board', '15'))
            current_player = int(board_info.get('Player', '1'))
            
            # Create the board representation
            # 0 = empty, 1 = current player, 2 = opponent
            board_array = np.zeros(bs*bs, dtype=np.float32)
            
            # Fill the board array from the state string
            if state_string and len(state_string) == bs*bs:
                for j, c in enumerate(state_string):
                    cell_value = int(c)
                    if cell_value == current_player:
                        board_array[j] = 1  # Current player's stone
                    elif cell_value != 0:
                        board_array[j] = 2  # Opponent's stone
            
            # Fill the input tensor
            x_input[i, :bs*bs] = board_array
            
            # Add the move, attack, and defense features
            x_input[i, -3] = float(move) / (bs*bs)  # Normalize move position
            x_input[i, -2] = attack
            x_input[i, -1] = defense
            
            # Print a summary of the board state
            stone_count = np.sum(board_array > 0)
            print(f"Input {i}: move={move}, stones={stone_count}, attack={attack:.2f}, defense={defense:.2f}")
        
        # Convert numpy array to PyTorch tensor and move to GPU
        with torch.no_grad():
            t_input = torch.from_numpy(x_input).to(device)
            
            # Forward pass through the neural network
            policy_logits, value_out = net(t_input)
            
            # Move results back to CPU for returning to C++
            policy_logits = policy_logits.cpu()
            value_out = value_out.cpu()
        
        # Convert to probabilities and prepare outputs
        policy_probs = F.softmax(policy_logits, dim=1).numpy()
        values = value_out.squeeze(-1).numpy()
        
        # Build output
        outputs = []
        for i in range(batch_size):
            outputs.append((policy_probs[i].tolist(), float(values[i])))
            print(f"Output {i}: value={float(values[i]):.4f}")
        
        return outputs
    except Exception as e:
        print(f"ERROR in neural network inference: {e}")
        import traceback
        traceback.print_exc()
        # Return reasonable defaults
        return [([1.0/225] * 225, 0.0)] * len(batch_input)

def self_play_game():
    cfg = mcts_py.MCTSConfig()
    cfg.num_simulations = 50
    cfg.c_puct = 1.0
    cfg.num_threads = 1
    wrapper = mcts_py.MCTSWrapper(cfg, boardSize=board_size)
    wrapper.set_infer_function(my_inference_callback)
    wrapper.set_batch_size(8)  # Process 8 requests at a time

    states_actions = []
    move_count = 0
    print("Starting a new game")
    
    while not wrapper.is_terminal() and move_count < 225:  # Add a move limit for safety
        # run MCTS
        print(f"Running search for move {move_count}...")
        wrapper.run_search()
        mv = wrapper.best_move()
        
        if mv < 0:
            print("Error: Invalid move returned")
            break
            
        print(f"Move {move_count}: {mv // board_size},{mv % board_size}")
        move_count += 1
        
        # record state, action for training
        # In a real implementation, we'd store the actual board state
        board_state = None  # Placeholder for actual board state
        states_actions.append((board_state, mv))
        
        wrapper.apply_best_move()

    # once done, get winner
    w = wrapper.get_winner()
    print(f"Game finished. Winner: {w} after {move_count} moves")
    
    # store to global data buffer with attack-defense information
    for (st, mv) in states_actions:
        # In a real implementation, we'd compute attack-defense for each move
        attack = 0.0  # Placeholder for attack bonus
        defense = 0.0  # Placeholder for defense bonus
        global_data_buffer.append((st, mv, w, attack, defense))

def main():
    num_games = 2  # Reduce to 2 games for testing
    for i in range(num_games):
        print(f"\n===== Game {i+1} =====")
        self_play_game()
    print(f"Collected {len(global_data_buffer)} samples in the buffer.")

if __name__ == "__main__":
    main()
