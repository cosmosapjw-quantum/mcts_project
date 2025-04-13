# self_play.py
import mcts_py
import torch
import torch.nn.functional as F
import random
import numpy as np

from model import SimpleGomokuNet

# A toy in-memory buffer for demonstration
global_data_buffer = []

def my_inference_callback(batch_input):
    """
    batch_input: list of (stateString, chosenMove, attack, defense)
    We'll parse them into a numeric vector, feed them to a PyTorch net, 
    and output a policy + value.
    """
    # Convert to a single batch
    # For demonstration, we won't parse "stateString" properly. We'll mock a board vector.
    # Real code: parse boardString -> [0 or 1 or 2 for each cell], or one-hot, etc.
    batch_size = len(batch_input)
    board_size = 15  # must match what you used in MCTSWrapper
    input_dim = board_size*board_size + 3

    # Build an input tensor
    x_input = np.zeros((batch_size, input_dim), dtype=np.float32)
    for i, (stateStr, move, attack, defense) in enumerate(batch_input):
        # Real code: parse the board from stateStr
        # We'll just do random for demonstration:
        board_vec = np.random.randint(0, 2, board_size*board_size)
        x_input[i, :board_size*board_size] = board_vec
        # Then chosenMove, attack, defense
        x_input[i, -3] = float(move)
        x_input[i, -2] = attack
        x_input[i, -1] = defense

    # to torch
    t_input = torch.from_numpy(x_input)  # shape [batch, input_dim]

    # Forward pass
    policy_logits, value_out = net(t_input)
    # Convert to python
    policy_np = policy_logits.detach().numpy()
    value_np  = value_out.detach().squeeze(-1).numpy()

    # We must produce a list of (policyList, value)
    # We'll do a softmax for policy
    outputs = []
    for i in range(batch_size):
        p_logits = policy_np[i]
        p_exp = np.exp(p_logits - np.max(p_logits))
        p_softmax = p_exp / np.sum(p_exp)
        val = float(value_np[i])
        # We'll just return the entire policy dimension
        outputs.append((p_softmax.tolist(), val))
    return outputs

# Create the net globally for demonstration
board_size = 15
policy_dim = board_size*board_size
net = SimpleGomokuNet(board_size=board_size, policy_dim=policy_dim)

def self_play_game():
    cfg = mcts_py.MCTSConfig()
    cfg.num_simulations = 10
    cfg.c_puct = 1.0
    cfg.num_threads = 2
    wrapper = mcts_py.MCTSWrapper(cfg, boardSize=board_size, use_omok=True)
    wrapper.set_infer_function(my_inference_callback)

    states_actions = []
    while not wrapper.is_terminal():
        # run MCTS
        wrapper.run_search()
        mv = wrapper.best_move()
        print(mv)
        # record state, action for training
        states_actions.append((None, mv))  # Real code: store actual board
        wrapper.apply_best_move()

    # once done, get winner
    w = wrapper.get_winner()
    # store to global data buffer
    for (st, mv) in states_actions:
        # label or advantage can be +1 if the winner was current player, etc.
        # This is up to your training method
        global_data_buffer.append((st, mv, w))

def main():
    for _ in range(5):
        self_play_game()
    print("Collected", len(global_data_buffer), "samples in the buffer.")

if __name__ == "__main__":
    main()
