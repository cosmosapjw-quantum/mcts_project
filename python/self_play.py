# self_play.py
import mcts_py
import torch
import torch.nn.functional as F
import numpy as np
from model import SimpleGomokuNet

# A toy in-memory buffer for demonstration
global_data_buffer = []

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the neural network and move it to GPU if available
board_size = 15
policy_dim = board_size * board_size
net = SimpleGomokuNet(board_size=board_size, policy_dim=policy_dim)
net.to(device)

def my_inference_callback(batch_input):
    """
    Enhanced callback using the actual game state
    """
    try:
        batch_size = len(batch_input)
        
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
                    key, value = part.split(':', 1)
                    if key == 'State':
                        state_string = value
                    else:
                        board_info[key] = value
            
            # Get the board size and current player
            bs = int(board_info.get('Board', '15'))
            current_player = int(board_info.get('Player', '1'))
            
            # Create the board representation
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
            
        return outputs
    except Exception as e:
        # Return reasonable defaults on error
        return [([1.0/225] * 225, 0.0)] * len(batch_input)

def self_play_game():
    # Configure MCTS
    cfg = mcts_py.MCTSConfig()
    cfg.num_simulations = 200
    cfg.c_puct = 1.0
    cfg.num_threads = 4
    cfg.parallel_leaf_batch_size = 8
    
    # Create wrapper
    wrapper = mcts_py.MCTSWrapper(cfg, boardSize=board_size)
    
    # Set neural network callback and batch size
    wrapper.set_infer_function(my_inference_callback)
    wrapper.set_batch_size(8)
    
    # Set exploration parameters
    wrapper.set_exploration_parameters(dirichlet_alpha=0.3, noise_weight=0.25)
    
    states_actions = []
    attack_defense_values = []
    move_count = 0
    
    # Temperature schedule for first 30 moves
    def get_temperature(move_num):
        if move_num < 10:
            return 1.0  # High temperature for first 10 moves (exploration)
        elif move_num < 20:
            return 0.5  # Medium temperature for next 10 moves
        else:
            return 0.1  # Low temperature for remaining moves (exploitation)
    
    while not wrapper.is_terminal() and move_count < board_size*board_size:
        # Run MCTS search
        wrapper.run_search()
        
        # Get temperature for current move
        temp = get_temperature(move_count)
        mv = wrapper.best_move_with_temperature(temp)
        
        if mv < 0:
            break
            
        move_count += 1
        
        # Record state, action, and attack/defense values for training
        board_state = None  # In a full implementation, store the actual board state
        attack = 0.0  # Placeholder
        defense = 0.0  # Placeholder
        states_actions.append((board_state, mv))
        attack_defense_values.append((attack, defense))
        
        # Apply the move with temperature
        wrapper.apply_best_move_with_temperature(temp)
    
    # Game results statistics
    w = wrapper.get_winner()
    
    # Store to global data buffer
    for i, (st, mv) in enumerate(states_actions):
        attack, defense = attack_defense_values[i]
        global_data_buffer.append((st, mv, w, attack, defense))
    
    return w

def main():
    num_games = 2  # Reduce to 2 games for testing
    for i in range(num_games):
        self_play_game()

if __name__ == "__main__":
    main()