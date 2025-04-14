# self_play.py
import mcts_py
import torch
import torch.nn.functional as F
import numpy as np
# Import the enhanced model class
from model import EnhancedGomokuNet

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

# self_play.py update
def my_inference_callback(batch_input):
    """
    Enhanced callback using the actual game state with player flag and move history
    """
    try:
        batch_size = len(batch_input)
        
        # Get the history moves parameter from the net
        num_history_moves = net.num_history_moves if hasattr(net, 'num_history_moves') else 3
        
        # Calculate input dimension with history moves
        # board_size*board_size (board) + 1 (player flag) + 2*num_history_moves (history) + 2 (attack/defense)
        input_dim = board_size*board_size + 1 + 2*num_history_moves + 2
        
        # Build input tensor
        x_input = np.zeros((batch_size, input_dim), dtype=np.float32)
        
        for i, (stateStr, move, attack, defense) in enumerate(batch_input):
            # Parse the board state from stateStr
            board_info = {}
            state_string = None
            current_moves_list = []
            opponent_moves_list = []
            
            # Split the string by semicolons
            parts = stateStr.split(';')
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
            
            # Fill the input tensor with board state
            x_input[i, :bs*bs] = board_array
            
            # Add player flag (1 for BLACK=1, 0 for WHITE=2)
            x_input[i, bs*bs] = 1.0 if current_player == 1 else 0.0
            
            # Add previous moves for current player (one-hot encoding)
            offset = bs*bs + 1
            for j, prev_move in enumerate(current_moves_list[:num_history_moves]):
                if prev_move >= 0:  # Valid move
                    # Normalize the move position
                    x_input[i, offset + j] = float(prev_move) / (bs*bs)
            
            # Add previous moves for opponent
            offset = bs*bs + 1 + num_history_moves
            for j, prev_move in enumerate(opponent_moves_list[:num_history_moves]):
                if prev_move >= 0:  # Valid move
                    # Normalize the move position
                    x_input[i, offset + j] = float(prev_move) / (bs*bs)
            
            # Add attack and defense scores
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
        print(f"Error in inference callback: {e}")
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
    
    # Configure the history moves to match our neural network
    wrapper.set_num_history_moves(num_history_moves)
    
    # Set exploration parameters
    wrapper.set_exploration_parameters(dirichlet_alpha=0.3, noise_weight=0.25)
    
    states_actions = []
    attack_defense_values = []
    history_moves = []  # New array to track move history for training
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
        
        # Record move in history for training data
        if len(history_moves) < move_count:
            history_moves.append(mv)
        else:
            history_moves[move_count - 1] = mv
        
        # Record state, action, and attack/defense values for training
        board_state = None  # In a full implementation, store the actual board state
        attack = 0.0  # Placeholder (in real implementation, get from attack/defense module)
        defense = 0.0  # Placeholder
        
        # Store the current position, move, and history data
        current_player_history = history_moves[::2][-num_history_moves:] if move_count % 2 == 1 else history_moves[1::2][-num_history_moves:]
        opponent_history = history_moves[1::2][-num_history_moves:] if move_count % 2 == 1 else history_moves[::2][-num_history_moves:]
        
        states_actions.append((board_state, mv, current_player_history, opponent_history))
        attack_defense_values.append((attack, defense))
        
        # Apply the move with temperature
        wrapper.apply_best_move_with_temperature(temp)
    
    # Game results statistics
    w = wrapper.get_winner()
    
    # Store to global data buffer with enhanced data
    for i, (st, mv, curr_hist, opp_hist) in enumerate(states_actions):
        attack, defense = attack_defense_values[i]
        # Include both player's move history in the training data
        global_data_buffer.append((st, mv, w, attack, defense, curr_hist, opp_hist))
    
    return w

# Main function remains unchanged
def main():
    num_games = 2  # Reduce to 2 games for testing
    for i in range(num_games):
        self_play_game()

if __name__ == "__main__":
    main()