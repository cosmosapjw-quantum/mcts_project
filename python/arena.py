# arena.py
import mcts_py
import torch
import numpy as np
from model import SimpleGomokuNet

def build_inference_fn(net):
    def callback(batch_input):
        # parse batch_input => run net => return (policy, value)
        # same approach as in self_play.py, 
        # but we might want a different net or some advanced logic
        outputs = []
        # ... same code as example ...
        return outputs
    return callback

def play_match(cfg, board_size, netA, netB):
    """
    Play a single game between netA (Player1) and netB (Player2) 
    using MCTS for each side.
    """
    # We'll create two wrappers, but we only need one if we carefully switch the net callback
    # For demonstration, let's just do a single wrapper but we change the callback each turn.
    # However, a simpler approach is to do a separate wrapper for each net, 
    # then alternate moves. We'll do the simpler approach below:

    wrapperA = mcts_py.MCTSWrapper(cfg, boardSize=board_size)
    wrapperB = mcts_py.MCTSWrapper(cfg, boardSize=board_size)
    wrapperA.set_infer_function(build_inference_fn(netA))
    wrapperB.set_infer_function(build_inference_fn(netB))

    current = wrapperA
    other   = wrapperB

    while not current.is_terminal():
        current.run_search()
        mv = current.best_move()
        current.apply_best_move()
        # we must also apply the same move to the other wrapper 
        # so they remain in sync
        other.apply_best_move() 

        # swap
        current, other = other, current

    winner = current.get_winner()  # after the last move
    return winner

def arena_example():
    cfg = mcts_py.MCTSConfig()
    cfg.num_simulations = 10
    cfg.c_puct = 1.0
    cfg.num_threads = 1

    # Suppose you have two different nets
    netA = SimpleGomokuNet(board_size=15, policy_dim=225)
    netB = SimpleGomokuNet(board_size=15, policy_dim=225)

    # you might load different weights:
    # netA.load_state_dict(torch.load("agentA.pth"))
    # netB.load_state_dict(torch.load("agentB.pth"))

    wins_for_A = 0
    nGames = 5
    for i in range(nGames):
        w = play_match(cfg, 15, netA, netB)
        if w == 1:
            wins_for_A += 1
        print(f"Game {i} ended. Winner = {w}")

    print(f"A won {wins_for_A} / {nGames}")

if __name__ == "__main__":
    arena_example()
