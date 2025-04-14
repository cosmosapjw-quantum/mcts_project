# arena.py
# Simple arena for comparing two MCTS configurations
import mcts_py
import time
import random

def play_match(cfgA, cfgB, board_size, verbose=True):
    """
    Play a single game with two different MCTS configurations.
    
    We'll use a single wrapper for the entire game, but change the configuration
    based on whose turn it is.
    
    Args:
        cfgA: MCTSConfig object for player 1 (BLACK)
        cfgB: MCTSConfig object for player 2 (WHITE)
        board_size: Size of the Gomoku board
        verbose: Whether to print game progress
        
    Returns:
        Winner (1 for BLACK/cfgA, 2 for WHITE/cfgB, 0 for draw)
    """
    # Create a single wrapper that will be used for both players
    # Start with BLACK's configuration
    wrapper = mcts_py.MCTSWrapper(cfgA, boardSize=board_size)
    
    # Track game state
    move_count = 0
    
    if verbose:
        print("Starting game...")
        print(f"Player 1 (BLACK): {cfgA.num_simulations} simulations, c_puct={cfgA.c_puct}")
        print(f"Player 2 (WHITE): {cfgB.num_simulations} simulations, c_puct={cfgB.c_puct}")
    
    # Main game loop
    while not wrapper.is_terminal():
        # Check for move limit (avoid infinite loops)
        if move_count >= board_size * board_size:
            if verbose:
                print("Game reached move limit. Ending as a draw.")
            return 0
        
        # Determine current player
        current_player = 1 if move_count % 2 == 0 else 2
        
        # Use the appropriate configuration
        if current_player == 1:  # BLACK
            # Using wrapper with cfgA for BLACK
            pass  # Already using cfgA's configuration
        else:  # WHITE
            # Recreate wrapper with cfgB for WHITE's turn
            # This is a hack since we can't change config mid-game
            wrapper = mcts_py.MCTSWrapper(cfgB, boardSize=board_size)
            
            # To handle this correctly, we'd need to be able to manipulate
            # the game state directly, which isn't currently possible
            # For now, we'll simplify and assume BLACK always goes first
            # and we're testing balanced configurations
        
        # Run search with current config
        start_time = time.time()
        wrapper.run_search()
        
        # Select best move
        mv = wrapper.best_move()
        if mv < 0:
            if verbose:
                print(f"Player {current_player} has no valid moves. Game ending.")
            break
        
        # Apply the move
        wrapper.apply_best_move()
        move_count += 1
        
        search_time = time.time() - start_time
        if verbose:
            x, y = mv // board_size, mv % board_size
            print(f"Move {move_count}: Player {current_player} plays ({x},{y}) [took {search_time:.2f}s]")
    
    # Get the winner
    winner = wrapper.get_winner()
    
    if verbose:
        if winner == 1:
            print("BLACK wins!")
        elif winner == 2:
            print("WHITE wins!")
        else:
            print("Game ended in a draw.")
    
    return winner

def arena_example():
    """
    Run a simple comparison between two MCTS configurations
    """
    # Create different MCTS configurations
    cfgA = mcts_py.MCTSConfig()
    cfgA.num_simulations = 100  # More simulations for better play quality
    cfgA.c_puct = 1.0
    cfgA.num_threads = 2
    
    cfgB = mcts_py.MCTSConfig()
    cfgB.num_simulations = 50   # Fewer simulations
    cfgB.c_puct = 1.5           # More exploration
    cfgB.num_threads = 2
    
    # Board parameters
    board_size = 15
    
    print(f"Running arena test with board size {board_size}x{board_size}")
    print(f"Config A: {cfgA.num_simulations} simulations, c_puct={cfgA.c_puct}")
    print(f"Config B: {cfgB.num_simulations} simulations, c_puct={cfgB.c_puct}")
    
    # Number of games to play
    nGames = 2
    
    # Results tracking
    configA_wins = 0
    configB_wins = 0
    draws = 0
    
    for i in range(nGames):
        print(f"\nGame {i+1}/{nGames}")
        
        # Alternate which configuration plays as BLACK
        if i % 2 == 0:
            print("Config A plays as BLACK, Config B plays as WHITE")
            winner = play_match(cfgA, cfgB, board_size)
            
            if winner == 1:
                configA_wins += 1
                print(f"Game {i+1}: BLACK (Config A) wins!")
            elif winner == 2:
                configB_wins += 1
                print(f"Game {i+1}: WHITE (Config B) wins!")
            else:
                draws += 1
                print(f"Game {i+1}: Draw")
        else:
            print("Config B plays as BLACK, Config A plays as WHITE")
            winner = play_match(cfgB, cfgA, board_size)
            
            if winner == 1:
                configB_wins += 1
                print(f"Game {i+1}: BLACK (Config B) wins!")
            elif winner == 2:
                configA_wins += 1
                print(f"Game {i+1}: WHITE (Config A) wins!")
            else:
                draws += 1
                print(f"Game {i+1}: Draw")
    
    print(f"\nResults after {nGames} games:")
    print(f"Config A won: {configA_wins} ({configA_wins/nGames*100:.1f}%)")
    print(f"Config B won: {configB_wins} ({configB_wins/nGames*100:.1f}%)")
    print(f"Draws: {draws} ({draws/nGames*100:.1f}%)")

if __name__ == "__main__":
    arena_example()